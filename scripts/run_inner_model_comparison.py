#!/usr/bin/env python
"""Compare EIPG with homogeneous-MNL and panel latent-class-MNL feedback.

The script reuses one existing controlled-synthetic run (normally the latest
paper-like run) for X_sim, D_H, D_calib_int_truth and D_cf_truth, then launches
two *new* outer searches with identical search budgets, mutation settings and
common random numbers.

Held-out counterfactual data are reported only; neither search optimizes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eipg.config import load_config
from eipg.econ import (
    MNLConfig,
    latent_class_coefficient_table,
    latent_class_panel_posteriors,
)
from eipg.objectives import CalibrationMomentConfig, RegularizationConfig
from eipg.outeropt import EvolutionSearchConfig, run_evolutionary_search
from eipg.outeropt.latent_class_evolution import (
    LatentClassInnerConfig,
    run_latent_class_evolutionary_search,
)
from eipg.personas import MixturePersonaGenerator, params_from_config
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare homogeneous and panel-latent-class EIPG inner models."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/controlled_synthetic_paperlike"),
        help="Parent experiment directory containing LATEST_RUN.txt.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific existing timestamped run. Overrides --experiment-dir.",
    )
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--lc-classes", type=int, default=None)
    parser.add_argument("--lc-restarts", type=int, default=2)
    parser.add_argument("--lc-em-max-iter", type=int, default=25)
    parser.add_argument("--lc-em-tol", type=float, default=1e-5)
    parser.add_argument("--lc-mstep-max-iter", type=int, default=80)
    parser.add_argument("--lc-init-scale", type=float, default=0.8)
    parser.add_argument("--panel-id-column", type=str, default="persona_id")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="inner_model_comparison_v1_10",
    )
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run = args.run_dir.resolve()
    else:
        pointer = args.experiment_dir / "LATEST_RUN.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"LATEST_RUN.txt not found: {pointer}")
        run = Path(pointer.read_text().strip()).resolve()
    if not run.exists():
        raise FileNotFoundError(run)
    return run


def _read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    parquet = run_dir / f"{stem}.parquet"
    csv = run_dir / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Could not find {parquet} or {csv}")


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def _initial_and_best_rows(
    history: pd.DataFrame,
    *,
    inner_model: str,
) -> list[dict[str, Any]]:
    if history.empty:
        raise ValueError("outer-search history is empty")
    initial = history.sort_values(["generation", "candidate_index"]).iloc[0]
    best = history.loc[history["objective_value"].idxmin()]
    rows = []
    for stage, row in [("initial", initial), ("best", best)]:
        rows.append(
            {
                "inner_model": inner_model,
                "stage": stage,
                "candidate": row["candidate"],
                "generation": int(row["generation"]),
                "candidate_index": int(row["candidate_index"]),
                "objective_value": float(row["objective_value"]),
                "calibration_l2_error": float(row["calibration_l2_error"]),
                "calibration_rmse": float(row["calibration_rmse"]),
                "cf_l2_error": float(row["cf_l2_error"]),
                "cf_rmse": float(row["cf_rmse"]),
                "anchor_nll_per_observation": float(row["anchor_nll_per_observation"]),
                "cf_nll_per_observation": float(row["cf_nll_per_observation"]),
                "mixture_entropy": float(row["mixture_entropy"]),
                "avg_pairwise_mean_distance": float(row["avg_pairwise_mean_distance"]),
                "regularization_objective": float(row["regularization_objective"]),
            }
        )
    return rows


def _delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for inner_model, g in summary.groupby("inner_model", sort=False):
        initial = g.loc[g["stage"] == "initial"].iloc[0]
        best = g.loc[g["stage"] == "best"].iloc[0]
        rows.append(
            {
                "inner_model": inner_model,
                "initial_calibration_l2": float(initial["calibration_l2_error"]),
                "best_calibration_l2": float(best["calibration_l2_error"]),
                "delta_calibration_l2": float(
                    best["calibration_l2_error"] - initial["calibration_l2_error"]
                ),
                "initial_cf_l2": float(initial["cf_l2_error"]),
                "best_cf_l2": float(best["cf_l2_error"]),
                "delta_cf_l2": float(best["cf_l2_error"] - initial["cf_l2_error"]),
                "best_objective": float(best["objective_value"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args)
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = load_config(config_path)

    gen_section = cfg.require_section("persona_generator")
    sim_section = cfg.require_section("simulation")
    benchmark_section = cfg.require_section("benchmark")
    inner_section = cfg.require_section("inner_model")
    calibration_section = cfg.require_section("calibration")
    regularization_section = cfg.require_section("regularization")
    outer_section = dict(cfg.require_section("outer_optimizer"))

    params = params_from_config(gen_section)
    simulator_cfg = SyntheticSimulatorConfig.from_config(sim_section)
    mnl_cfg = MNLConfig.from_config(inner_section, default_features=params.features)
    calibration_cfg = CalibrationMomentConfig.from_config(
        calibration_section,
        default_features=mnl_cfg.features,
        intervened_alternative_id=int(benchmark_section.get("intervened_alternative_id", 0)),
    )
    regularization_cfg = RegularizationConfig.from_config(regularization_section)

    # Equalize search budget/population across both inner models while keeping
    # all other outer-search hyperparameters identical to the paper-like config.
    outer_section["budget"] = int(args.budget)
    outer_section["population"] = int(args.population)
    outer_section["population_size"] = int(args.population)
    search_cfg = EvolutionSearchConfig.from_config(outer_section)

    lc_cfg = LatentClassInnerConfig(
        n_classes=int(args.lc_classes or gen_section.get("k_components", 3)),
        panel_id_column=str(args.panel_id_column),
        n_restarts=int(args.lc_restarts),
        em_max_iter=int(args.lc_em_max_iter),
        em_tol=float(args.lc_em_tol),
        mstep_max_iter=int(args.lc_mstep_max_iter),
        init_scale=float(args.lc_init_scale),
        min_class_weight=1e-3,
        seed=int(cfg.seed + 30_000),
    )

    x_sim = _read_table(run_dir, "X_sim")
    d_h = _read_table(run_dir, "D_H")
    d_calib_int = _read_table(run_dir, "D_calib_int_truth")
    d_cf = _read_table(run_dir, "D_cf_truth")

    n_personas = int(sim_section.get("n_personas", 160))
    n_observations = int(sim_section.get("n_obs", 1200))
    persona_seed = int(cfg.seed)
    simulator_seed = int(cfg.seed + 202)
    mutation_seed = int(cfg.seed + 20_000)

    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running fair inner-model comparison")
    print(f"  source run: {run_dir}")
    print(f"  budget: {search_cfg.budget}")
    print(f"  population: {search_cfg.population_size}")
    print(f"  evaluations/model: {search_cfg.budget * search_cfg.population_size}")
    print(f"  LC classes: {lc_cfg.n_classes}")
    print(f"  LC restarts: {lc_cfg.n_restarts}")
    print(f"  LC EM max iter: {lc_cfg.em_max_iter}")

    print("\n[1/2] Homogeneous MNL EIPG search...")
    homogeneous = run_evolutionary_search(
        initial_params=params,
        x_sim=x_sim,
        d_h=d_h,
        d_calib_int=d_calib_int,
        d_cf=d_cf,
        simulator_config=simulator_cfg,
        mnl_config=mnl_cfg,
        calibration_config=calibration_cfg,
        regularization_config=regularization_cfg,
        search_config=search_cfg,
        seed=mutation_seed,
        n_personas=n_personas,
        n_observations=n_observations,
        evaluation_persona_seed=persona_seed,
        evaluation_simulator_seed=simulator_seed,
    )

    print("[2/2] Panel latent-class MNL EIPG search...")
    latent = run_latent_class_evolutionary_search(
        initial_params=params,
        x_sim=x_sim,
        d_h=d_h,
        d_calib_int=d_calib_int,
        d_cf=d_cf,
        simulator_config=simulator_cfg,
        mnl_config=mnl_cfg,
        calibration_config=calibration_cfg,
        regularization_config=regularization_cfg,
        search_config=search_cfg,
        lc_config=lc_cfg,
        seed=mutation_seed,
        n_personas=n_personas,
        n_observations=n_observations,
        evaluation_persona_seed=persona_seed,
        evaluation_simulator_seed=simulator_seed,
    )

    homogeneous_history = homogeneous.history.copy()
    homogeneous_history["inner_model"] = "homogeneous_mnl"
    latent_history = latent.history.copy()

    homogeneous_history_path = _write_table(
        homogeneous_history, output_dir / "homogeneous_mnl_history.parquet"
    )
    latent_history_path = _write_table(
        latent_history, output_dir / "panel_latent_class_mnl_history.parquet"
    )
    homogeneous_result_path = homogeneous.save_json(
        output_dir / "homogeneous_mnl_search_result.json"
    )
    latent_result_path = latent.save_json(
        output_dir / "panel_latent_class_mnl_search_result.json"
    )
    homogeneous_best_path = homogeneous.best.result.save_json(
        output_dir / "homogeneous_mnl_best.json"
    )
    latent_best_path = latent.best.result.save_json(
        output_dir / "panel_latent_class_mnl_best.json"
    )
    homogeneous.best.params.save_json(output_dir / "generator_params_eipg_homogeneous_mnl.json")
    latent.best.params.save_json(output_dir / "generator_params_eipg_panel_latent_class_mnl.json")
    latent.best.result.fit.save_json(output_dir / "panel_latent_class_mnl_best_fit.json")
    _write_table(
        latent_class_coefficient_table(latent.best.result.fit),
        output_dir / "panel_latent_class_mnl_best_coefficients.parquet",
    )

    # Recreate the best LC training dataset exactly to expose panel posteriors.
    best_generator = MixturePersonaGenerator(latent.best.params, seed=persona_seed)
    best_personas = best_generator.sample(n_personas)
    best_simulator = RandomUtilityChoiceSimulator(simulator_cfg, seed=simulator_seed)
    best_d_phi = best_simulator.simulate_long_dataset(
        contexts=x_sim,
        personas=best_personas,
        n_observations=n_observations,
        dataset_label="D_phi_eipg_panel_latent_class_best",
    )
    _write_table(
        latent_class_panel_posteriors(
            best_d_phi,
            latent.best.result.fit,
            panel_id_column=lc_cfg.panel_id_column,
        ),
        output_dir / "panel_latent_class_mnl_best_panel_posteriors.parquet",
    )

    comparison_rows = []
    comparison_rows.extend(
        _initial_and_best_rows(homogeneous_history, inner_model="homogeneous_mnl")
    )
    comparison_rows.extend(
        _initial_and_best_rows(latent_history, inner_model="panel_latent_class_mnl")
    )
    comparison = pd.DataFrame(comparison_rows)
    deltas = _delta_summary(comparison)
    comparison_path = _write_table(
        comparison, output_dir / "inner_model_eipg_comparison.parquet"
    )
    deltas_path = _write_table(
        deltas, output_dir / "inner_model_eipg_deltas.parquet"
    )
    summary_json_path = _write_json(
        {
            "source_run": str(run_dir),
            "search_config": search_cfg.to_dict(),
            "latent_class_config": lc_cfg.to_dict(),
            "comparison_rows": comparison.to_dict(orient="records"),
            "delta_rows": deltas.to_dict(orient="records"),
        },
        output_dir / "inner_model_eipg_comparison.json",
    )

    print("\n=== EIPG INNER-MODEL COMPARISON ===")
    print(
        comparison[
            [
                "inner_model",
                "stage",
                "calibration_l2_error",
                "cf_l2_error",
                "anchor_nll_per_observation",
                "cf_nll_per_observation",
                "avg_pairwise_mean_distance",
                "objective_value",
            ]
        ].to_string(index=False)
    )
    print("\n=== CHANGE FROM INITIAL TO BEST ===")
    print(deltas.to_string(index=False))

    print("\nArtifacts:")
    print(f"  output_dir: {output_dir}")
    print(f"  homogeneous history: {homogeneous_history_path}")
    print(f"  latent history: {latent_history_path}")
    print(f"  homogeneous result: {homogeneous_result_path}")
    print(f"  latent result: {latent_result_path}")
    print(f"  homogeneous best: {homogeneous_best_path}")
    print(f"  latent best: {latent_best_path}")
    print(f"  comparison: {comparison_path}")
    print(f"  deltas: {deltas_path}")
    print(f"  json: {summary_json_path}")


if __name__ == "__main__":
    main()
