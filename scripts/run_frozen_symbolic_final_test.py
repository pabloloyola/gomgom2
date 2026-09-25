#!/usr/bin/env python3
"""Run the preregistered frozen symbolic EIPG final-test protocol.

The search stage never reads the development CF dataset and never sees final-test
outcomes. For each fresh benchmark realization, it selects generators using only
anchor/intervention calibration plus the frozen regularizer. Only after every
search replicate for that benchmark is complete is a fresh final counterfactual
truth realization generated from its predeclared independent test seed.

Do not alter the frozen config in response to the outputs of this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eipg.econ import MNLConfig, MultinomialLogitModel, predict_probabilities_long
from eipg.experiments import InterventionSpec, apply_intervention, combine_intervention_reports
from eipg.objectives import (
    CalibrationMomentConfig,
    RegularizationConfig,
    build_calibration_report,
    regularization_report,
)
from eipg.outeropt.evolution import mutate_generator_params
from eipg.outeropt.weighted_evolution import block_weighted_calibration_score
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator, PersonaLatent, params_from_config
from eipg.reporting.io import load_json, load_yaml, read_table
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig, generate_context_sets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/eipg_frozen_symbolic.yaml"))
    p.add_argument("--benchmark-seeds", type=int, nargs="+", default=None)
    p.add_argument("--search-seeds", type=int, nargs="+", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def run_checked(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def params_from_json(path: Path) -> MixtureGeneratorParams:
    d = load_json(path)
    return MixtureGeneratorParams(
        weights=np.asarray(d["weights"], dtype=float),
        means=np.asarray(d["means"], dtype=float),
        features=tuple(d["features"]),
        within_component_std=float(d.get("within_component_std", 0.30)),
        segment_labels=tuple(d.get("segment_labels", [])),
    )


def personas_from_table(df: pd.DataFrame, features: tuple[str, ...]) -> list[PersonaLatent]:
    out: list[PersonaLatent] = []
    for _, row in df.iterrows():
        out.append(
            PersonaLatent(
                persona_id=int(row["persona_id"]),
                segment_id=int(row.get("segment_id", row.get("persona_segment_id", 0))),
                segment_label=str(row.get("segment_label", row.get("persona_segment_label", "segment"))),
                z=np.asarray([float(row[f"z_{f}"]) for f in features], dtype=float),
                features=features,
            )
        )
    return out


def intervention_specs(raw: list[dict[str, Any]]) -> dict[str, InterventionSpec]:
    out: dict[str, InterventionSpec] = {}
    for d in raw:
        spec = InterventionSpec(
            name=str(d["name"]),
            attribute=str(d["attribute"]),
            mode=str(d["mode"]),
            value=float(d["value"]),
            intervened_alternative_id=int(d.get("intervened_alternative_id", 0)),
            clip_min=float(d["clip_min"]) if d.get("clip_min") is not None else None,
            clip_max=float(d["clip_max"]) if d.get("clip_max") is not None else None,
        )
        out[spec.name] = spec
    return out


def build_benchmark(base_cfg: dict[str, Any], seed: int, run_dir: Path, cfg_path: Path) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    cfg.setdefault("run", {})["seed"] = int(seed)
    cfg["run"]["timestamped"] = False
    cfg["run"]["output_dir"] = str(run_dir)
    cfg.setdefault("pipeline", {})["stop_after"] = "datasets"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_checked([
        sys.executable,
        "scripts/run_controlled_synthetic.py",
        "--config", str(cfg_path),
        "--stage", "datasets",
        "--no-timestamp",
    ])


def make_intervention_targets(
    *,
    x_h: pd.DataFrame,
    human_personas: list[PersonaLatent],
    specs: dict[str, InterventionSpec],
    n_contexts: int,
    n_observations: int,
    sim_cfg: SyntheticSimulatorConfig,
    benchmark_seed: int,
) -> dict[str, pd.DataFrame]:
    targets: dict[str, pd.DataFrame] = {}
    for idx, (name, spec) in enumerate(specs.items()):
        x_int = apply_intervention(
            x_h,
            spec,
            prefix=f"frozen_{name}",
            context_set=f"X_calib_frozen_{name}",
            n_contexts=n_contexts,
        )
        simulator = RandomUtilityChoiceSimulator(sim_cfg, seed=benchmark_seed + 50_000 + idx)
        targets[name] = simulator.simulate_long_dataset(
            contexts=x_int,
            personas=human_personas,
            n_observations=n_observations,
            dataset_label=f"D_calib_frozen_{name}",
        )
    return targets


def score_candidate(
    *,
    params: MixtureGeneratorParams,
    bundle: list[str],
    block_weights: dict[str, float],
    reg_multiplier: float,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    intervention_targets: dict[str, pd.DataFrame],
    sim_cfg: SyntheticSimulatorConfig,
    mnl_cfg: MNLConfig,
    calib_cfg: CalibrationMomentConfig,
    reg_cfg: RegularizationConfig,
    n_personas: int,
    n_obs: int,
    persona_seed: int,
    simulator_seed: int,
) -> tuple[float, float, float]:
    personas = MixturePersonaGenerator(params, seed=persona_seed).sample(n_personas)
    d_phi = RandomUtilityChoiceSimulator(sim_cfg, seed=simulator_seed).simulate_long_dataset(
        contexts=x_sim,
        personas=personas,
        n_observations=n_obs,
        dataset_label="D_phi_frozen_search",
    )
    fit = MultinomialLogitModel(mnl_cfg).fit(d_phi)
    pred_h = predict_probabilities_long(d_h, beta=fit.beta, features=fit.features)
    reports = []
    for name in bundle:
        target = intervention_targets[name]
        pred = predict_probabilities_long(target, beta=fit.beta, features=fit.features)
        reports.append((name, build_calibration_report(
            anchor_target=d_h,
            anchor_model=pred_h,
            intervention_target=target,
            intervention_model=pred,
            config=calib_cfg,
        )))
    combined = combine_intervention_reports(reports)
    weighted_cal, _ = block_weighted_calibration_score(combined, block_weights=block_weights)
    reg = float(regularization_report(params, reg_cfg).objective())
    objective = float(weighted_cal + reg_multiplier * reg)
    return objective, float(weighted_cal), reg


def frozen_search(
    *,
    initial: MixtureGeneratorParams,
    bundle: list[str],
    block_weights: dict[str, float],
    search_cfg: dict[str, Any],
    search_seed: int,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    intervention_targets: dict[str, pd.DataFrame],
    sim_cfg: SyntheticSimulatorConfig,
    mnl_cfg: MNLConfig,
    calib_cfg: CalibrationMomentConfig,
    reg_cfg: RegularizationConfig,
    n_personas: int,
    n_obs: int,
) -> tuple[MixtureGeneratorParams, pd.DataFrame]:
    budget = int(search_cfg["budget"])
    population = int(search_cfg["population"])
    rng = np.random.default_rng(search_seed + 20_000)
    center = initial
    best_params = initial
    best_obj = float("inf")
    rows: list[dict[str, Any]] = []

    for generation in range(budget):
        scale = float(search_cfg["sigma_decay"]) ** generation
        candidates = [center]
        while len(candidates) < population:
            candidates.append(mutate_generator_params(
                center,
                rng=rng,
                mean_scale=float(search_cfg["mean_mutation_scale"]) * scale,
                weight_logit_scale=float(search_cfg["weight_logit_mutation_scale"]) * scale,
                mean_clip=float(search_cfg["mean_clip"]),
            ))
        generation_results: list[tuple[float, MixtureGeneratorParams]] = []
        for candidate_index, params in enumerate(candidates):
            objective, weighted_cal, reg = score_candidate(
                params=params,
                bundle=bundle,
                block_weights=block_weights,
                reg_multiplier=float(search_cfg["regularization_multiplier"]),
                x_sim=x_sim,
                d_h=d_h,
                intervention_targets=intervention_targets,
                sim_cfg=sim_cfg,
                mnl_cfg=mnl_cfg,
                calib_cfg=calib_cfg,
                reg_cfg=reg_cfg,
                n_personas=n_personas,
                n_obs=n_obs,
                persona_seed=search_seed,
                simulator_seed=search_seed + 202,
            )
            rows.append({
                "generation": generation,
                "candidate_index": candidate_index,
                "objective": objective,
                "weighted_calibration": weighted_cal,
                "regularization": reg,
            })
            generation_results.append((objective, params))
            if objective < best_obj:
                best_obj = objective
                best_params = params
        center = min(generation_results, key=lambda x: x[0])[1]
    return best_params, pd.DataFrame(rows)


def make_final_truth(
    *,
    source_cfg: dict[str, Any],
    test_seed: int,
    sim_cfg: SyntheticSimulatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_sets = generate_context_sets(source_cfg["benchmark"], seed=test_seed)
    x_cf = context_sets.x_cf.copy()
    truth_cfg: dict[str, Any] = dict(source_cfg["persona_generator"])
    truth_cfg.update(source_cfg["synthetic_truth"])
    truth_params = params_from_config(truth_cfg)
    n_human = int(source_cfg["simulation"].get("n_human_personas", 320))
    truth_personas = MixturePersonaGenerator(truth_params, seed=test_seed + 101).sample(n_human)
    d_cf = RandomUtilityChoiceSimulator(sim_cfg, seed=test_seed + 202).simulate_long_dataset(
        contexts=x_cf,
        personas=truth_personas,
        n_observations=int(source_cfg["simulation"].get("n_cf_obs", 600)),
        dataset_label="D_cf_final_truth",
    )
    return x_cf, d_cf


def final_cf_error(
    *,
    params: MixtureGeneratorParams,
    x_sim: pd.DataFrame,
    d_cf: pd.DataFrame,
    sim_cfg: SyntheticSimulatorConfig,
    mnl_cfg: MNLConfig,
    calib_cfg: CalibrationMomentConfig,
    n_personas: int,
    n_obs: int,
    search_seed: int,
) -> float:
    personas = MixturePersonaGenerator(params, seed=search_seed).sample(n_personas)
    d_phi = RandomUtilityChoiceSimulator(sim_cfg, seed=search_seed + 202).simulate_long_dataset(
        contexts=x_sim,
        personas=personas,
        n_observations=n_obs,
        dataset_label="D_phi_final_eval",
    )
    fit = MultinomialLogitModel(mnl_cfg).fit(d_phi)
    pred = predict_probabilities_long(d_cf, beta=fit.beta, features=fit.features)
    report = build_calibration_report(anchor_target=d_cf, anchor_model=pred, config=calib_cfg)
    return float(report.summary()["l2_error"])


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    initial = results[results["method"] == "initial_static"][
        ["benchmark_seed", "search_seed", "final_cf_l2"]
    ].rename(columns={"final_cf_l2": "initial_cf_l2"})
    paired = results.merge(initial, on=["benchmark_seed", "search_seed"], how="left")
    paired["cf_change_vs_initial_fraction"] = (
        paired["final_cf_l2"] - paired["initial_cf_l2"]
    ) / paired["initial_cf_l2"].abs().clip(lower=1e-12)
    rows = []
    for method, g in paired.groupby("method", sort=True):
        rows.append({
            "method": method,
            "n_replications": int(len(g)),
            "mean_final_cf_l2": float(g["final_cf_l2"].mean()),
            "median_final_cf_l2": float(g["final_cf_l2"].median()),
            "std_final_cf_l2": float(g["final_cf_l2"].std(ddof=1)),
            "mean_cf_change_vs_initial_fraction": float(g["cf_change_vs_initial_fraction"].mean()),
            "fraction_replications_cf_improved_vs_initial": float((g["cf_change_vs_initial_fraction"] < 0).mean()),
        })
    return pd.DataFrame(rows), paired


def main() -> None:
    args = parse_args()
    frozen_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    benchmark_cfg = frozen_cfg["benchmark"]
    search_cfg = frozen_cfg["search"]
    calibration_cfg = frozen_cfg["calibration"]

    benchmark_seeds = [int(x) for x in (args.benchmark_seeds or benchmark_cfg["benchmark_seeds"])]
    search_seeds = [int(x) for x in (args.search_seeds or benchmark_cfg["search_seeds"])]
    declared_benchmarks = [int(x) for x in benchmark_cfg["benchmark_seeds"]]
    declared_tests = [int(x) for x in benchmark_cfg["final_test_seeds"]]
    test_seed_map = dict(zip(declared_benchmarks, declared_tests, strict=True))
    missing = [s for s in benchmark_seeds if s not in test_seed_map]
    if missing:
        raise ValueError(f"benchmark seeds are not preregistered in frozen config: {missing}")

    out_root = args.output_dir or Path(frozen_cfg["output"]["dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    base_cfg_path = Path(benchmark_cfg["base_config"])
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    intervention_cfg_path = Path(calibration_cfg["intervention_config"])
    intervention_cfg = yaml.safe_load(intervention_cfg_path.read_text(encoding="utf-8")) or {}
    specs = intervention_specs(intervention_cfg["interventions"])
    bundles = {str(k): [str(x) for x in v] for k, v in intervention_cfg["bundles"].items()}
    int_exp = intervention_cfg["experiment"]

    final_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for benchmark_seed in benchmark_seeds:
        print(f"\n=== FROZEN FINAL BENCHMARK {benchmark_seed} ===", flush=True)
        run_dir = out / f"benchmark_seed_{benchmark_seed}"
        generated_cfg = out / f"benchmark_seed_{benchmark_seed}.yaml"
        build_benchmark(base_cfg, benchmark_seed, run_dir, generated_cfg)
        source_cfg = load_yaml(run_dir / "config_used.yaml")
        initial = params_from_json(run_dir / "generator_params_initial.json")
        x_sim = read_table(run_dir, "X_sim")
        x_h = read_table(run_dir, "X_H")
        d_h = read_table(run_dir, "D_H")
        human_table = read_table(run_dir, "human_personas")
        human_personas = personas_from_table(human_table, initial.features)

        sim_cfg = SyntheticSimulatorConfig.from_config(source_cfg["simulation"])
        mnl_cfg = MNLConfig.from_config(source_cfg["inner_model"], default_features=initial.features)
        calib_cfg = CalibrationMomentConfig.from_config(
            source_cfg["calibration"],
            default_features=initial.features,
            intervened_alternative_id=int(source_cfg["benchmark"].get("intervened_alternative_id", 0)),
        )
        reg_cfg = RegularizationConfig.from_config(source_cfg["regularization"])
        n_personas = int(source_cfg["simulation"].get("n_personas", 160))
        n_obs = int(source_cfg["simulation"].get("n_obs", 1200))
        targets = make_intervention_targets(
            x_h=x_h,
            human_personas=human_personas,
            specs=specs,
            n_contexts=int(int_exp["n_calibration_contexts_per_intervention"]),
            n_observations=int(int_exp["n_calibration_observations_per_intervention"]),
            sim_cfg=sim_cfg,
            benchmark_seed=benchmark_seed,
        )

        selected: dict[tuple[int, str], MixtureGeneratorParams] = {}
        for search_seed in search_seeds:
            for method, bundle_key, weights in [
                ("price_only_equal", calibration_cfg["comparator_bundle"], calibration_cfg["comparator_block_weights"]),
                ("all_four_equal", calibration_cfg["primary_bundle"], calibration_cfg["primary_block_weights"]),
            ]:
                params, history = frozen_search(
                    initial=initial,
                    bundle=bundles[str(bundle_key)],
                    block_weights={str(k): float(v) for k, v in weights.items()},
                    search_cfg=search_cfg,
                    search_seed=search_seed,
                    x_sim=x_sim,
                    d_h=d_h,
                    intervention_targets=targets,
                    sim_cfg=sim_cfg,
                    mnl_cfg=mnl_cfg,
                    calib_cfg=calib_cfg,
                    reg_cfg=reg_cfg,
                    n_personas=n_personas,
                    n_obs=n_obs,
                )
                selected[(search_seed, method)] = params
                params_dir = out / "selected_params"
                params_dir.mkdir(exist_ok=True)
                params.save_json(params_dir / f"b{benchmark_seed}_s{search_seed}_{method}.json")
                h = history.copy()
                h.insert(0, "benchmark_seed", benchmark_seed)
                h.insert(1, "search_seed", search_seed)
                h.insert(2, "method", method)
                history_frames.append(h)

        # Final truth is generated only after all searches for this benchmark are complete.
        test_seed = int(test_seed_map[benchmark_seed])
        x_cf_final, d_cf_final = make_final_truth(
            source_cfg=source_cfg,
            test_seed=test_seed,
            sim_cfg=sim_cfg,
        )
        x_cf_final.to_csv(out / f"X_cf_final_b{benchmark_seed}.csv", index=False)
        d_cf_final.to_csv(out / f"D_cf_final_truth_b{benchmark_seed}.csv", index=False)

        for search_seed in search_seeds:
            methods = {
                "initial_static": initial,
                "price_only_equal": selected[(search_seed, "price_only_equal")],
                "all_four_equal": selected[(search_seed, "all_four_equal")],
            }
            for method, params in methods.items():
                cf_l2 = final_cf_error(
                    params=params,
                    x_sim=x_sim,
                    d_cf=d_cf_final,
                    sim_cfg=sim_cfg,
                    mnl_cfg=mnl_cfg,
                    calib_cfg=calib_cfg,
                    n_personas=n_personas,
                    n_obs=n_obs,
                    search_seed=search_seed,
                )
                final_rows.append({
                    "benchmark_seed": benchmark_seed,
                    "final_test_seed": test_seed,
                    "search_seed": search_seed,
                    "method": method,
                    "final_cf_l2": cf_l2,
                })
                pd.DataFrame(final_rows).to_csv(out / "final_test_results.csv", index=False)

    results = pd.DataFrame(final_rows)
    summary, paired = aggregate(results)
    results.to_csv(out / "final_test_results.csv", index=False)
    paired.to_csv(out / "final_test_paired.csv", index=False)
    summary.to_csv(out / "final_test_summary.csv", index=False)
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(out / "search_history.csv", index=False)

    manifest = {
        "protocol": frozen_cfg["protocol"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config": str(args.config),
        "benchmark_seeds": benchmark_seeds,
        "final_test_seed_map": {str(k): int(test_seed_map[k]) for k in benchmark_seeds},
        "search_seeds": search_seeds,
        "selection_reads_development_cf": False,
        "selection_reads_final_cf": False,
        "final_truth_generated_after_selection": True,
        "methods": frozen_cfg["methods"],
        "note": "Final-test metrics are for reporting only. Do not tune the frozen method using these values.",
    }
    (out / "final_test_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== FROZEN SYMBOLIC FINAL TEST SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nArtifacts: {out}")
    print("IMPORTANT: do not use final-test outcomes for further method tuning.")


if __name__ == "__main__":
    main()
