#!/usr/bin/env python3
"""Run the frozen Nutri2Cycle Belgium-bread EIPG *search stage*.

This script never reads the final-holdout choice column. It selects public-benchmark
persona generators using only Tasks 1--3 and writes selected parameters for a later,
separate Task-4 evaluation. Do not change the frozen protocol in response to search
outputs or eventual Task-4 outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from eipg.econ import MNLConfig, MultinomialLogitModel
from eipg.objectives.regularization import RegularizationConfig, regularization_report
from eipg.outeropt.evolution import mutate_generator_params
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig

ALTS = ("organic", "conventional", "circular", "none")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/public_food_benchmark.yaml"))
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--search-seeds", type=int, nargs="+", default=None)
    return p.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def calibration_panel(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Select the frozen population without touching Task-4 outcomes."""
    ds = cfg["dataset"]
    calibration_cols = [cfg["tasks"][int(t)]["column"] for t in cfg["split"]["calibration_tasks"]]
    valid_codes = {int(v) for v in ds["choice_codes"].values()}
    mask = pd.to_numeric(raw[ds["country_column"]], errors="coerce").eq(int(ds["country_code"]))
    mask &= pd.to_numeric(raw[ds["purchase_column"]], errors="coerce").eq(int(ds["purchase_required_value"]))
    for col in calibration_cols:
        mask &= pd.to_numeric(raw[col], errors="coerce").isin(valid_codes)
    panel = raw.loc[mask, [ds["respondent_id"], *calibration_cols]].copy()
    expected = int(ds["expected_complete_panels"])
    if len(panel) != expected:
        raise RuntimeError(f"expected {expected} calibration respondents; got {len(panel)}")
    return panel


def make_contexts(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in cfg["split"]["calibration_tasks"]:
        prices = cfg["tasks"][int(task)]["prices"]
        for alt in ALTS:
            p = float(prices[alt])
            rows.append({
                "context_id": f"bread_task_{int(task)}",
                "task_id": int(task),
                "alternative_id": alt,
                # Simulator-facing structured persona dimensions.
                "price_sensitivity": p,
                "organic_affinity": float(alt == "organic"),
                "conventional_affinity": float(alt == "conventional"),
                "circular_affinity": float(alt == "circular"),
                "opt_out_propensity": float(alt == "none"),
                # Inner MNL features.
                "price": p,
                "asc_organic": float(alt == "organic"),
                "asc_conventional": float(alt == "conventional"),
                "asc_circular": float(alt == "circular"),
            })
    return pd.DataFrame(rows)


def human_targets(panel: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    ds = cfg["dataset"]
    code_to_alt = {int(code): alt for alt, code in ds["choice_codes"].items()}
    shares: dict[int, dict[str, float]] = {}
    mean_price: dict[int, float] = {}
    for task in cfg["split"]["calibration_tasks"]:
        task = int(task)
        col = cfg["tasks"][task]["column"]
        choices = pd.to_numeric(panel[col], errors="raise").astype(int).map(code_to_alt)
        shares[task] = {alt: float((choices == alt).mean()) for alt in ALTS}
        prices = cfg["tasks"][task]["prices"]
        mean_price[task] = float(np.mean([float(prices[a]) for a in choices]))

    anchor = int(cfg["split"]["anchor_tasks"][0])
    interventions = [int(t) for t in cfg["split"]["calibration_intervention_tasks"]]
    responses = {
        task: {alt: shares[task][alt] - shares[anchor][alt] for alt in ALTS}
        for task in interventions
    }
    return {
        "shares": shares,
        "mean_price": mean_price,
        "anchor": anchor,
        "interventions": interventions,
        "responses": responses,
    }


def initial_params(cfg: dict[str, Any]) -> MixtureGeneratorParams:
    p = cfg["persona_generator"]
    return MixtureGeneratorParams(
        weights=np.asarray(p["initial_weights"], dtype=float),
        means=np.asarray(p["initial_means"], dtype=float),
        features=tuple(str(x) for x in p["features"]),
        within_component_std=float(p["within_component_std"]),
        segment_labels=tuple(str(x) for x in p["segment_labels"]),
    )


def softmax(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def predicted_moments(beta: np.ndarray, features: tuple[str, ...], contexts: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    task_shares: dict[int, dict[str, float]] = {}
    task_mean_price: dict[int, float] = {}
    for task, g0 in contexts.groupby("task_id", sort=True):
        g = g0.set_index("alternative_id").loc[list(ALTS)].reset_index()
        util = g[list(features)].to_numpy(dtype=float) @ np.asarray(beta, dtype=float)
        prob = softmax(util)
        task = int(task)
        task_shares[task] = {alt: float(prob[i]) for i, alt in enumerate(ALTS)}
        task_mean_price[task] = float(np.sum(prob * g["price"].to_numpy(dtype=float)))
    anchor = int(cfg["split"]["anchor_tasks"][0])
    interventions = [int(t) for t in cfg["split"]["calibration_intervention_tasks"]]
    responses = {
        task: {alt: task_shares[task][alt] - task_shares[anchor][alt] for alt in ALTS}
        for task in interventions
    }
    return {"shares": task_shares, "mean_price": task_mean_price, "responses": responses}


def block_vectors(target: dict[str, Any], pred: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    anchor = int(target["anchor"])
    ints = list(target["interventions"])
    blocks: dict[str, np.ndarray] = {}
    blocks["anchor_choice_shares"] = np.asarray(
        [pred["shares"][anchor][a] - target["shares"][anchor][a] for a in ALTS], dtype=float
    )
    blocks["anchor_mean_chosen_price"] = np.asarray(
        [pred["mean_price"][anchor] - target["mean_price"][anchor]], dtype=float
    )
    blocks["intervention_task_choice_shares"] = np.asarray(
        [pred["shares"][t][a] - target["shares"][t][a] for t in ints for a in ALTS], dtype=float
    )
    blocks["intervention_share_responses"] = np.asarray(
        [pred["responses"][t][a] - target["responses"][t][a] for t in ints for a in ALTS], dtype=float
    )
    return blocks


def weighted_score(blocks: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    weighted_sum = 0.0
    total_weight = 0.0
    diag: dict[str, float] = {}
    for name, w0 in weights.items():
        w = float(w0)
        if w <= 0:
            continue
        diff = np.asarray(blocks[name], dtype=float)
        mse = float(np.mean(diff * diff))
        diag[f"block_{name}_rmse"] = float(np.sqrt(mse))
        weighted_sum += w * mse
        total_weight += w
    if total_weight <= 0:
        raise ValueError("no active calibration blocks")
    return float(np.sqrt(weighted_sum / total_weight)), diag


def evaluate_candidate(
    params: MixtureGeneratorParams,
    *,
    method: str,
    cfg: dict[str, Any],
    contexts: pd.DataFrame,
    targets: dict[str, Any],
    persona_seed: int,
    simulator_seed: int,
) -> dict[str, Any]:
    pg = cfg["persona_generator"]
    personas = MixturePersonaGenerator(params, seed=persona_seed).sample(int(pg["n_personas"]))
    sim = RandomUtilityChoiceSimulator(SyntheticSimulatorConfig(choice_temperature=1.0), seed=simulator_seed)
    d = sim.simulate_long_dataset(
        contexts=contexts,
        personas=personas,
        n_observations=int(pg["simulation_observations"]),
        dataset_label=f"public_{method}",
    )
    im = cfg["inner_model"]
    mnl = MultinomialLogitModel(MNLConfig(
        features=tuple(im["features"]), l2=float(im["l2"]), max_iter=int(im["max_iter"])
    )).fit(d)
    pred = predicted_moments(mnl.beta, mnl.features, contexts, cfg)
    blocks = block_vectors(targets, pred, cfg)
    weights = cfg["calibration"]["anchor_only_blocks" if method == "anchor_only_eipg" else "intervention_rich_blocks"]
    cal, block_diag = weighted_score(blocks, weights)
    reg_cfg = RegularizationConfig.from_config(cfg["regularization"])
    reg_report = regularization_report(params, reg_cfg)
    reg = float(reg_report.objective())
    objective = float(cal + float(cfg["search"]["regularization_multiplier"]) * reg)
    return {
        "objective": objective,
        "weighted_calibration": cal,
        "regularization": reg,
        "mnl_nll_per_observation": float(mnl.train_nll_per_observation),
        "mnl_success": bool(mnl.success),
        "mnl_beta": mnl.beta_by_feature,
        "predicted_moments": pred,
        **block_diag,
    }


def run_search(method: str, seed: int, initial: MixtureGeneratorParams, cfg: dict[str, Any], contexts: pd.DataFrame, targets: dict[str, Any]) -> tuple[MixtureGeneratorParams, pd.DataFrame, dict[str, Any]]:
    sc = cfg["search"]
    rng = np.random.default_rng(seed + 20_000)
    center = initial
    best = initial
    best_eval: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    persona_seed = seed + 100_000
    simulator_seed = seed + 110_000

    for generation in range(int(sc["budget"])):
        scale = float(sc["sigma_decay"]) ** generation
        candidates = [center]
        while len(candidates) < int(sc["population"]):
            candidates.append(mutate_generator_params(
                center,
                rng=rng,
                mean_scale=float(sc["mean_mutation_scale"]) * scale,
                weight_logit_scale=float(sc["weight_logit_mutation_scale"]) * scale,
                mean_clip=float(sc["mean_clip"]),
            ))
        generation_evals: list[tuple[float, MixtureGeneratorParams]] = []
        for idx, params in enumerate(candidates):
            ev = evaluate_candidate(
                params, method=method, cfg=cfg, contexts=contexts, targets=targets,
                persona_seed=persona_seed, simulator_seed=simulator_seed,
            )
            rows.append({
                "method": method, "search_seed": seed, "generation": generation,
                "candidate_index": idx, "objective": ev["objective"],
                "weighted_calibration": ev["weighted_calibration"],
                "regularization": ev["regularization"],
                "mnl_nll_per_observation": ev["mnl_nll_per_observation"],
                **{k: v for k, v in ev.items() if k.startswith("block_")},
            })
            generation_evals.append((float(ev["objective"]), params))
            if best_eval is None or float(ev["objective"]) < float(best_eval["objective"]):
                best, best_eval = params, ev
        center = min(generation_evals, key=lambda x: x[0])[1]
        print(f"{method} seed={seed} gen={generation}: best={min(x[0] for x in generation_evals):.6f}", flush=True)
    assert best_eval is not None
    return best, pd.DataFrame(rows), best_eval


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    if cfg["protocol"]["status"] != "frozen_pre_eipg":
        raise RuntimeError("public benchmark protocol must be frozen before search")
    out = args.outdir or Path(cfg["output"]["dir"]) / "search_stage"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    raw = pd.read_csv(args.csv, sep=";", low_memory=False)
    panel = calibration_panel(raw, cfg)
    contexts = make_contexts(cfg)
    targets = human_targets(panel, cfg)
    (out / "human_calibration_targets.json").write_text(json.dumps(targets, indent=2, sort_keys=True), encoding="utf-8")
    contexts.to_csv(out / "calibration_contexts.csv", index=False)

    initial = initial_params(cfg)
    initial.save_json(out / "initial_generator.json")
    seeds = [int(s) for s in (args.search_seeds or cfg["search"]["search_seeds"])]
    declared = {int(s) for s in cfg["search"]["search_seeds"]}
    if any(s not in declared for s in seeds):
        raise ValueError("all search seeds must be predeclared in the frozen config")

    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for method in ("anchor_only_eipg", "intervention_rich_eipg"):
        for seed in seeds:
            best, hist, best_eval = run_search(method, seed, initial, cfg, contexts, targets)
            histories.append(hist)
            best.save_json(out / f"selected_{method}_seed_{seed}.json")
            summaries.append({
                "method": method,
                "search_seed": seed,
                "selected_objective": float(best_eval["objective"]),
                "selected_weighted_calibration": float(best_eval["weighted_calibration"]),
                "selected_regularization": float(best_eval["regularization"]),
                "selected_mnl_nll_per_observation": float(best_eval["mnl_nll_per_observation"]),
                "selected_mnl_beta": json.dumps(best_eval["mnl_beta"], sort_keys=True),
                "final_holdout_outcomes_consumed": False,
            })

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(out / "search_history.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "selected_summary.csv", index=False)
    manifest = {
        "protocol": cfg["protocol"]["name"],
        "n_respondents": int(len(panel)),
        "search_seeds": seeds,
        "methods": ["anchor_only_eipg", "intervention_rich_eipg"],
        "candidate_evaluations": int(len(history)),
        "final_holdout_column_read": False,
        "final_holdout_outcomes_consumed": False,
        "next_step": "freeze selected search artifacts, then run separate Task-4 evaluation",
    }
    (out / "search_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
