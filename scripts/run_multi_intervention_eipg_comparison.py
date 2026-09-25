#!/usr/bin/env python3
"""Compare EIPG outer-loop search under richer calibration interventions.

This is a development experiment. Candidate selection uses calibration moments
and regularization only. CF-dev is evaluated after each candidate has been fit
solely to diagnose whether the calibration objective supplies a better search
signal; it is never part of the candidate-level objective.

The default comparison is:
  A. price-only calibration, equal block weights;
  B. all-four interventions, equal block weights;
  C. all-four interventions, substitution block weight 4.

Each benchmark seed regenerates contexts, synthetic-human personas, anchor data,
intervention targets, and CF-dev data. Search seeds vary mutation randomness and
candidate simulation CRNs within each benchmark realization.
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
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator, PersonaLatent
from eipg.reporting.io import load_json, load_yaml, read_table
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multi_intervention_eipg_comparison.yaml"),
    )
    p.add_argument("--benchmark-seeds", type=int, nargs="+", default=None)
    p.add_argument("--search-seeds", type=int, nargs="+", default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--population", type=int, default=None)
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
    specs: dict[str, InterventionSpec] = {}
    for d in raw:
        spec = InterventionSpec(
            name=str(d["name"]),
            attribute=str(d["attribute"]),
            mode=str(d["mode"]),
            value=float(d["value"]),
            intervened_alternative_id=int(d.get("intervened_alternative_id", 0)),
            clip_min=(float(d["clip_min"]) if d.get("clip_min") is not None else None),
            clip_max=(float(d["clip_max"]) if d.get("clip_max") is not None else None),
        )
        specs[spec.name] = spec
    return specs


def safe_fraction(delta: float, baseline: float) -> float:
    return float(delta / max(abs(float(baseline)), 1.0e-12))


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    value = x.corr(y)
    return float(value) if pd.notna(value) else float("nan")


def spearman(x: pd.Series, y: pd.Series) -> float:
    return safe_corr(x.rank(method="average"), y.rank(method="average"))


def write_table(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def build_benchmark(
    *,
    base_cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    config_path: Path,
) -> None:
    seed_cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    seed_cfg.setdefault("run", {})["seed"] = int(seed)
    seed_cfg["run"]["timestamped"] = False
    seed_cfg["run"]["output_dir"] = str(run_dir)
    seed_cfg.setdefault("pipeline", {})["stop_after"] = "datasets"
    config_path.write_text(yaml.safe_dump(seed_cfg, sort_keys=False), encoding="utf-8")
    run_checked(
        [
            sys.executable,
            "scripts/run_controlled_synthetic.py",
            "--config",
            str(config_path),
            "--stage",
            "datasets",
            "--no-timestamp",
        ]
    )


def generate_intervention_targets(
    *,
    x_h: pd.DataFrame,
    human_personas: list[PersonaLatent],
    specs: dict[str, InterventionSpec],
    n_contexts: int,
    n_observations: int,
    simulator_config: SyntheticSimulatorConfig,
    benchmark_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    contexts: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.DataFrame] = {}
    for idx, (name, spec) in enumerate(specs.items()):
        x_int = apply_intervention(
            x_h,
            spec,
            prefix=f"outer_{name}",
            context_set=f"X_calib_outer_{name}",
            n_contexts=n_contexts,
        )
        simulator = RandomUtilityChoiceSimulator(
            simulator_config,
            seed=int(benchmark_seed + 50_000 + idx),
        )
        d_int = simulator.simulate_long_dataset(
            contexts=x_int,
            personas=human_personas,
            n_observations=n_observations,
            dataset_label=f"D_calib_outer_{name}",
        )
        contexts[name] = x_int
        targets[name] = d_int
    return contexts, targets


def evaluate_candidate(
    *,
    params: MixtureGeneratorParams,
    candidate_name: str,
    bundle_names: list[str],
    block_weights: dict[str, float],
    regularization_multiplier: float,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    d_cf: pd.DataFrame,
    intervention_targets: dict[str, pd.DataFrame],
    simulator_config: SyntheticSimulatorConfig,
    mnl_config: MNLConfig,
    calibration_config: CalibrationMomentConfig,
    regularization_config: RegularizationConfig,
    n_personas: int,
    n_observations: int,
    persona_seed: int,
    simulator_seed: int,
) -> dict[str, Any]:
    generator = MixturePersonaGenerator(params, seed=persona_seed)
    personas = generator.sample(n_personas)
    simulator = RandomUtilityChoiceSimulator(simulator_config, seed=simulator_seed)
    d_phi = simulator.simulate_long_dataset(
        contexts=x_sim,
        personas=personas,
        n_observations=n_observations,
        dataset_label=f"D_phi_{candidate_name}",
    )
    fit = MultinomialLogitModel(mnl_config).fit(d_phi)

    pred_h = predict_probabilities_long(d_h, beta=fit.beta, features=fit.features)
    pred_cf = predict_probabilities_long(d_cf, beta=fit.beta, features=fit.features)
    cf_report = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=pred_cf,
        config=calibration_config,
    )
    cf_l2 = float(cf_report.summary()["l2_error"])

    per_intervention = {}
    for name in bundle_names:
        target = intervention_targets[name]
        pred = predict_probabilities_long(target, beta=fit.beta, features=fit.features)
        per_intervention[name] = build_calibration_report(
            anchor_target=d_h,
            anchor_model=pred_h,
            intervention_target=target,
            intervention_model=pred,
            config=calibration_config,
        )

    combined = combine_intervention_reports(
        (name, per_intervention[name]) for name in bundle_names
    )
    weighted_calibration, diagnostics = block_weighted_calibration_score(
        combined,
        block_weights=block_weights,
    )
    reg_report = regularization_report(params, regularization_config)
    reg_value = float(reg_report.objective())
    objective = float(weighted_calibration + regularization_multiplier * reg_value)

    return {
        "params": params,
        "objective": objective,
        "weighted_calibration": float(weighted_calibration),
        "raw_calibration_l2": float(combined.summary()["l2_error"]),
        "cf_dev_l2": cf_l2,
        "regularization": reg_value,
        "dispersion": float(reg_report.terms["avg_pairwise_mean_distance"]),
        "train_nll_per_observation": float(fit.train_nll_per_observation),
        "block_level_shares_rmse": float(diagnostics.get("block_level_shares_rmse", np.nan)),
        "block_level_attributes_rmse": float(diagnostics.get("block_level_attributes_rmse", np.nan)),
        "block_substitution_rmse": float(diagnostics.get("block_substitution_rmse", np.nan)),
        "block_intervention_attributes_rmse": float(
            diagnostics.get("block_intervention_attributes_rmse", np.nan)
        ),
    }


def run_condition_search(
    *,
    condition_name: str,
    bundle_names: list[str],
    block_weights: dict[str, float],
    initial_params: MixtureGeneratorParams,
    search_seed: int,
    benchmark_seed: int,
    budget: int,
    population: int,
    mean_mutation_scale: float,
    weight_logit_mutation_scale: float,
    mean_clip: float,
    sigma_decay: float,
    regularization_multiplier: float,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    d_cf: pd.DataFrame,
    intervention_targets: dict[str, pd.DataFrame],
    simulator_config: SyntheticSimulatorConfig,
    mnl_config: MNLConfig,
    calibration_config: CalibrationMomentConfig,
    regularization_config: RegularizationConfig,
    n_personas: int,
    n_observations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(search_seed + 20_000))
    persona_seed = int(search_seed)
    simulator_seed = int(search_seed + 202)
    center = initial_params
    history_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    total_evals = budget * population
    completed = 0
    for generation in range(budget):
        scale = float(sigma_decay) ** generation
        candidates: list[MixtureGeneratorParams] = [center]
        while len(candidates) < population:
            candidates.append(
                mutate_generator_params(
                    center,
                    rng=rng,
                    mean_scale=float(mean_mutation_scale) * scale,
                    weight_logit_scale=float(weight_logit_mutation_scale) * scale,
                    mean_clip=float(mean_clip),
                )
            )

        generation_results: list[dict[str, Any]] = []
        for candidate_index, params in enumerate(candidates[:population]):
            candidate_name = (
                f"{condition_name}_b{benchmark_seed}_s{search_seed}_"
                f"g{generation:03d}_c{candidate_index:03d}"
            )
            result = evaluate_candidate(
                params=params,
                candidate_name=candidate_name,
                bundle_names=bundle_names,
                block_weights=block_weights,
                regularization_multiplier=regularization_multiplier,
                x_sim=x_sim,
                d_h=d_h,
                d_cf=d_cf,
                intervention_targets=intervention_targets,
                simulator_config=simulator_config,
                mnl_config=mnl_config,
                calibration_config=calibration_config,
                regularization_config=regularization_config,
                n_personas=n_personas,
                n_observations=n_observations,
                persona_seed=persona_seed,
                simulator_seed=simulator_seed,
            )
            completed += 1
            row = {
                "benchmark_seed": benchmark_seed,
                "search_seed": search_seed,
                "condition": condition_name,
                "generation": generation,
                "candidate_index": candidate_index,
                "candidate_name": candidate_name,
                "objective": result["objective"],
                "weighted_calibration": result["weighted_calibration"],
                "raw_calibration_l2": result["raw_calibration_l2"],
                "cf_dev_l2": result["cf_dev_l2"],
                "regularization": result["regularization"],
                "dispersion": result["dispersion"],
                "train_nll_per_observation": result["train_nll_per_observation"],
                "block_level_shares_rmse": result["block_level_shares_rmse"],
                "block_level_attributes_rmse": result["block_level_attributes_rmse"],
                "block_substitution_rmse": result["block_substitution_rmse"],
                "block_intervention_attributes_rmse": result[
                    "block_intervention_attributes_rmse"
                ],
            }
            history_rows.append(row)
            generation_results.append({**result, **row})
            if best is None or float(result["objective"]) < float(best["objective"]):
                best = {**result, **row}
            print(
                f"    [{completed:03d}/{total_evals:03d}] g={generation} c={candidate_index} "
                f"J={result['objective']:.5f} cal={result['weighted_calibration']:.5f} "
                f"CF-dev={result['cf_dev_l2']:.5f}",
                flush=True,
            )

        generation_best = min(generation_results, key=lambda r: float(r["objective"]))
        center = generation_best["params"]

    assert best is not None
    history = pd.DataFrame(history_rows)
    initial = history[(history["generation"] == 0) & (history["candidate_index"] == 0)].iloc[0]
    summary = {
        "benchmark_seed": benchmark_seed,
        "search_seed": search_seed,
        "condition": condition_name,
        "bundle": "+".join(bundle_names),
        "n_interventions": len(bundle_names),
        "initial_weighted_calibration": float(initial["weighted_calibration"]),
        "best_weighted_calibration": float(best["weighted_calibration"]),
        "weighted_calibration_improvement_fraction": safe_fraction(
            float(initial["weighted_calibration"]) - float(best["weighted_calibration"]),
            float(initial["weighted_calibration"]),
        ),
        "initial_cf_dev_l2": float(initial["cf_dev_l2"]),
        "selected_cf_dev_l2": float(best["cf_dev_l2"]),
        "cf_dev_change_fraction": safe_fraction(
            float(best["cf_dev_l2"]) - float(initial["cf_dev_l2"]),
            float(initial["cf_dev_l2"]),
        ),
        "selected_objective": float(best["objective"]),
        "selected_dispersion": float(best["dispersion"]),
        "initial_substitution_rmse": float(initial["block_substitution_rmse"]),
        "selected_substitution_rmse": float(best["block_substitution_rmse"]),
        "pearson_calibration_vs_cf": safe_corr(
            history["weighted_calibration"], history["cf_dev_l2"]
        ),
        "spearman_calibration_vs_cf": spearman(
            history["weighted_calibration"], history["cf_dev_l2"]
        ),
        "cf_worsened": bool(float(best["cf_dev_l2"]) > float(initial["cf_dev_l2"])),
        "selected_candidate": str(best["candidate_name"]),
    }
    return history, summary


def aggregate_summary(rep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, g in rep.groupby("condition", sort=True):
        rows.append(
            {
                "condition": condition,
                "n_replications": int(len(g)),
                "mean_weighted_calibration_improvement_fraction": float(
                    g["weighted_calibration_improvement_fraction"].mean()
                ),
                "mean_cf_dev_change_fraction": float(g["cf_dev_change_fraction"].mean()),
                "median_cf_dev_change_fraction": float(g["cf_dev_change_fraction"].median()),
                "fraction_runs_cf_worsened": float(g["cf_worsened"].astype(float).mean()),
                "mean_pearson_calibration_vs_cf": float(g["pearson_calibration_vs_cf"].mean()),
                "mean_spearman_calibration_vs_cf": float(g["spearman_calibration_vs_cf"].mean()),
                "mean_initial_cf_dev_l2": float(g["initial_cf_dev_l2"].mean()),
                "mean_selected_cf_dev_l2": float(g["selected_cf_dev_l2"].mean()),
                "mean_initial_substitution_rmse": float(g["initial_substitution_rmse"].mean()),
                "mean_selected_substitution_rmse": float(g["selected_substitution_rmse"].mean()),
                "mean_selected_dispersion": float(g["selected_dispersion"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    exp = cfg.get("experiment", {})
    search = cfg.get("search", {})
    conditions = cfg.get("conditions", {})
    if not conditions:
        raise ValueError("config must define at least one condition")

    benchmark_seeds = [int(x) for x in (args.benchmark_seeds or exp.get("benchmark_seeds", []))]
    search_seeds = [int(x) for x in (args.search_seeds or exp.get("search_seeds", []))]
    if not benchmark_seeds or not search_seeds:
        raise ValueError("benchmark_seeds and search_seeds must be non-empty")

    budget = int(args.budget or search.get("budget", 6))
    population = int(args.population or search.get("population", 4))
    base_cfg_path = Path(exp.get("base_benchmark_config", "configs/controlled_synthetic_paperlike.yaml"))
    intervention_cfg_path = Path(
        exp.get("intervention_config", "configs/multi_intervention_identification.yaml")
    )
    out_root = args.output_dir or Path(
        exp.get("output_dir", "outputs/multi_intervention_eipg_comparison")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = out_root / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    intervention_cfg = yaml.safe_load(intervention_cfg_path.read_text(encoding="utf-8")) or {}
    specs = intervention_specs(intervention_cfg.get("interventions", []))
    bundles = {
        str(k): [str(v) for v in values]
        for k, values in (intervention_cfg.get("bundles", {}) or {}).items()
    }
    int_exp = intervention_cfg.get("experiment", {})
    n_cal_contexts = int(int_exp.get("n_calibration_contexts_per_intervention", 120))
    n_cal_obs = int(int_exp.get("n_calibration_observations_per_intervention", 600))

    history_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    total_replications = len(benchmark_seeds) * len(search_seeds) * len(conditions)
    replication_counter = 0

    for benchmark_seed in benchmark_seeds:
        print(f"\n=== BENCHMARK seed={benchmark_seed} ===", flush=True)
        benchmark_dir = batch_dir / f"benchmark_seed_{benchmark_seed}"
        benchmark_cfg_path = batch_dir / f"benchmark_seed_{benchmark_seed}.yaml"
        build_benchmark(
            base_cfg=base_cfg,
            seed=benchmark_seed,
            run_dir=benchmark_dir,
            config_path=benchmark_cfg_path,
        )

        source_cfg = load_yaml(benchmark_dir / "config_used.yaml")
        initial = params_from_json(benchmark_dir / "generator_params_initial.json")
        x_sim = read_table(benchmark_dir, "X_sim")
        x_h = read_table(benchmark_dir, "X_H")
        d_h = read_table(benchmark_dir, "D_H")
        d_cf = read_table(benchmark_dir, "D_cf_truth")
        human_table = read_table(benchmark_dir, "human_personas")
        human_personas = personas_from_table(human_table, initial.features)

        sim_cfg = SyntheticSimulatorConfig.from_config(source_cfg["simulation"])
        mnl_cfg = MNLConfig.from_config(
            source_cfg["inner_model"], default_features=initial.features
        )
        calib_cfg = CalibrationMomentConfig.from_config(
            source_cfg["calibration"],
            default_features=initial.features,
            intervened_alternative_id=int(
                source_cfg["benchmark"].get("intervened_alternative_id", 0)
            ),
        )
        reg_cfg = RegularizationConfig.from_config(source_cfg["regularization"])
        n_personas = int(source_cfg["simulation"].get("n_personas", 160))
        n_obs = int(source_cfg["simulation"].get("n_obs", 1200))

        _, intervention_targets = generate_intervention_targets(
            x_h=x_h,
            human_personas=human_personas,
            specs=specs,
            n_contexts=n_cal_contexts,
            n_observations=n_cal_obs,
            simulator_config=sim_cfg,
            benchmark_seed=benchmark_seed,
        )

        for search_seed in search_seeds:
            for condition_name, condition in conditions.items():
                replication_counter += 1
                bundle_name = str(condition["bundle"])
                if bundle_name not in bundles:
                    raise ValueError(
                        f"condition {condition_name} references unknown bundle {bundle_name}"
                    )
                bundle_names = bundles[bundle_name]
                block_weights = {
                    str(k): float(v)
                    for k, v in (condition.get("block_weights", {}) or {}).items()
                }
                print(
                    f"\n--- REPLICATION {replication_counter}/{total_replications}: "
                    f"benchmark={benchmark_seed} search={search_seed} "
                    f"condition={condition_name} ---",
                    flush=True,
                )
                history, summary = run_condition_search(
                    condition_name=str(condition_name),
                    bundle_names=bundle_names,
                    block_weights=block_weights,
                    initial_params=initial,
                    search_seed=search_seed,
                    benchmark_seed=benchmark_seed,
                    budget=budget,
                    population=population,
                    mean_mutation_scale=float(search.get("mean_mutation_scale", 0.25)),
                    weight_logit_mutation_scale=float(
                        search.get("weight_logit_mutation_scale", 0.20)
                    ),
                    mean_clip=float(search.get("mean_clip", 4.0)),
                    sigma_decay=float(search.get("sigma_decay", 0.90)),
                    regularization_multiplier=float(
                        search.get("regularization_multiplier", 0.05)
                    ),
                    x_sim=x_sim,
                    d_h=d_h,
                    d_cf=d_cf,
                    intervention_targets=intervention_targets,
                    simulator_config=sim_cfg,
                    mnl_config=mnl_cfg,
                    calibration_config=calib_cfg,
                    regularization_config=reg_cfg,
                    n_personas=n_personas,
                    n_observations=n_obs,
                )
                history_frames.append(history)
                summaries.append(summary)

                # Checkpoint after every completed benchmark/search/condition replicate.
                pd.DataFrame(summaries).to_csv(
                    batch_dir / "replication_results.csv", index=False
                )
                pd.concat(history_frames, ignore_index=True).to_csv(
                    batch_dir / "candidate_history.csv", index=False
                )

    rep_df = pd.DataFrame(summaries)
    history_df = pd.concat(history_frames, ignore_index=True)
    aggregate = aggregate_summary(rep_df)
    rep_path = write_table(rep_df, batch_dir / "replication_results.parquet")
    history_path = write_table(history_df, batch_dir / "candidate_history.parquet")
    aggregate_path = write_table(aggregate, batch_dir / "condition_summary.parquet")
    rep_df.to_csv(batch_dir / "replication_results.csv", index=False)
    history_df.to_csv(batch_dir / "candidate_history.csv", index=False)
    aggregate.to_csv(batch_dir / "condition_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development_outer_loop_multi_intervention_comparison",
        "cf_role": "CF-dev only; never used in candidate-level selection",
        "benchmark_seeds": benchmark_seeds,
        "search_seeds": search_seeds,
        "budget": budget,
        "population": population,
        "regularization_multiplier": float(search.get("regularization_multiplier", 0.05)),
        "conditions": conditions,
        "intervention_config": str(intervention_cfg_path),
        "base_benchmark_config": str(base_cfg_path),
        "n_candidate_fits": int(
            len(benchmark_seeds) * len(search_seeds) * len(conditions) * budget * population
        ),
        "artifacts": {
            "replication_results": str(rep_path),
            "candidate_history": str(history_path),
            "condition_summary": str(aggregate_path),
        },
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (batch_dir / "config_used.yaml").write_text(
        args.config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== MULTI-INTERVENTION EIPG COMPARISON ===")
    print(aggregate.to_string(index=False))
    print(f"\nArtifacts: {batch_dir}")


if __name__ == "__main__":
    main()
