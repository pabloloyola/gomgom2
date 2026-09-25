#!/usr/bin/env python
"""Run the controlled synthetic benchmark.

The controlled synthetic runner supports staged execution, timestamped run folders, and the full symbolic EIPG loop. The runner now:

- loads a paper-aligned config;
- constructs the initial persona generator ``G_phi``;
- samples candidate personas ``z ~ p_phi(z)``;
- generates the context sets ``X_sim``, ``X_H``, ``X_calib_int``, and ``X_cf``;
- samples a ground-truth synthetic-human population;
- simulates choices from a random-utility simulator for ``D_phi``, ``D_H``,
  calibration interventions, and held-out counterfactual contexts;
- fits an MNL model ``m_{beta*(phi)}`` to ``D_phi_initial``;
- scores the fitted MNL on anchor and held-out datasets;
- computes calibration moments comparing model-implied behavior to target behavior;
- computes generator regularization diagnostics;
- evaluates a simple controlled-synthetic baseline ladder;
- runs an evolutionary outer loop that updates phi.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

import pandas as pd

# Allow running the script before editable installation, while still working with `uv run`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eipg.baselines import (
    baseline_results_table,
    default_baseline_candidates,
    evaluate_baseline_candidate,
)
from eipg.config import load_config
from eipg.diagnostics import (
    build_economic_model_capacity_diagnostic,
    direct_oracle_probability_rows,
)
from eipg.econ import (
    MNLConfig,
    MultinomialLogitModel,
    evaluate_fitted_mnl,
    fit_latent_class_mnl,
    fit_oracle_segmented_mnl,
    latent_class_coefficient_table,
    latent_class_panel_posteriors,
    predict_latent_class_mnl_long,
    predict_oracle_mixture_mnl_long,
    predict_oracle_segmented_mnl_long,
    predict_probabilities_long,
    segmented_coefficient_table,
)
from eipg.objectives import (
    CalibrationMomentConfig,
    RegularizationConfig,
    build_calibration_report,
    compare_calibration_reports,
    regularization_report,
)
from eipg.outeropt import EvolutionSearchConfig, run_evolutionary_search
from eipg.personas import MixturePersonaGenerator, PersonaRenderer, params_from_config
from eipg.simulators import (
    RandomUtilityChoiceSimulator,
    SyntheticSimulatorConfig,
    generate_context_sets,
    summarize_choice_dataset,
)

REQUIRED_SECTIONS = [
    "run",
    "benchmark",
    "persona_generator",
    "simulation",
    "synthetic_truth",
    "inner_model",
    "calibration",
    "regularization",
    "outer_optimizer",
]

STAGE_ORDER = [
    "setup",
    "personas",
    "contexts",
    "datasets",
    "mnl",
    "calibration",
    "baselines",
    "outer",
]


def _target_stage(cfg, cli_stage: str | None) -> str:
    """Resolve the final stage to run."""

    if cli_stage and cli_stage != "all":
        return cli_stage
    pipeline_section = cfg.data.get("pipeline", {})
    configured = str(pipeline_section.get("stop_after", "all"))
    if configured == "all":
        return "outer"
    if configured not in STAGE_ORDER:
        raise ValueError(
            f"Unknown pipeline.stop_after={configured!r}; expected one of {STAGE_ORDER} or 'all'."
        )
    return configured


def _stage_reached(current_stage: str, target_stage: str) -> bool:
    return STAGE_ORDER.index(current_stage) >= STAGE_ORDER.index(target_stage)


def _resolve_output_dir(cfg, args: argparse.Namespace) -> tuple[Path, str | None]:
    """Resolve output directory, optionally adding a timestamped run subfolder."""

    base_out = args.output_dir or cfg.output_dir
    run_section = cfg.data.get("run", {})
    timestamp_enabled = bool(run_section.get("timestamped", False))
    if args.timestamp is not None:
        timestamp_enabled = bool(args.timestamp)

    timestamp = None
    if timestamp_enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(base_out) / timestamp, timestamp
    return Path(base_out), timestamp


def _write_config_copy(cfg, out_dir: Path) -> Path:
    """Write the exact loaded config into the run folder."""

    path = out_dir / "config_used.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.data, sort_keys=False), encoding="utf-8")
    return path


def _write_latest_pointer(out_dir: Path) -> Path | None:
    """Write a pointer to the newest timestamped run beside the timestamp folders."""

    parent = out_dir.parent
    if parent == out_dir:
        return None
    pointer = parent / "LATEST_RUN.txt"
    pointer.write_text(str(out_dir.resolve()) + "\n", encoding="utf-8")
    return pointer


def _print_stop(stage: str, out_dir: Path, extra: dict[str, Path] | None = None) -> None:
    print(f"EIPG clean v1.7 stopped after stage: {stage}")
    print(f"  output_dir: {out_dir}")
    if extra:
        for label, path in extra.items():
            print(f"  {label}: {path}")
    print("\nResume by running the same config with a later --stage value or --stage all.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled synthetic EIPG benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_synthetic_smoke.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--stage",
        choices=["setup", "personas", "contexts", "datasets", "mnl", "calibration", "baselines", "outer", "all"],
        default=None,
        help=(
            "Run through this stage and then stop. "
            "Use 'datasets' to generate X_* and D_* files without fitting MNL or optimizing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from the config.",
    )
    timestamp_group = parser.add_mutually_exclusive_group()
    timestamp_group.add_argument(
        "--timestamp",
        action="store_true",
        default=None,
        help="Force creation of a timestamped subdirectory inside the configured output_dir.",
    )
    timestamp_group.add_argument(
        "--no-timestamp",
        action="store_false",
        dest="timestamp",
        help="Write directly into output_dir, without creating a timestamped subdirectory.",
    )
    return parser.parse_args()


def _write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame, preferring parquet and falling back to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    for section in REQUIRED_SECTIONS:
        cfg.require_section(section)

    target_stage = _target_stage(cfg, args.stage)
    out_dir, run_timestamp = _resolve_output_dir(cfg, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = _write_config_copy(cfg, out_dir)
    latest_pointer_path = _write_latest_pointer(out_dir) if run_timestamp else None
    manifest_path = cfg.write_manifest(out_dir)
    run_metadata_path = _write_json(
        {
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "stage_requested": args.stage or cfg.data.get("pipeline", {}).get("stop_after", "all"),
            "stage_resolved": target_stage,
            "timestamp": run_timestamp,
            "output_dir": str(out_dir),
            "config_path": str(args.config),
            "config_copy_path": str(config_copy_path),
            "latest_pointer_path": str(latest_pointer_path) if latest_pointer_path else None,
        },
        out_dir / "run_metadata.json",
    )

    if _stage_reached("setup", target_stage):
        _print_stop(
            "setup",
            out_dir,
            {
                "manifest": manifest_path,
                "metadata": run_metadata_path,
                "config_used": config_copy_path,
            },
        )
        return

    gen_section = cfg.require_section("persona_generator")
    truth_section = cfg.require_section("synthetic_truth")
    benchmark_section = cfg.require_section("benchmark")
    sim_section = cfg.require_section("simulation")

    n_personas = int(sim_section.get("n_personas", 40))
    n_human_personas = int(sim_section.get("n_human_personas", 80))
    n_obs = int(sim_section.get("n_obs", 100))
    n_anchor_obs = int(sim_section.get("n_anchor_obs", n_obs))
    n_calib_obs = int(sim_section.get("n_calib_obs", n_anchor_obs))
    n_cf_obs = int(sim_section.get("n_cf_obs", n_anchor_obs))

    # ------------------------------------------------------------------
    # 1. Initial candidate generator G_phi.
    # ------------------------------------------------------------------
    params = params_from_config(gen_section)
    generator = MixturePersonaGenerator(params=params, seed=cfg.seed)
    personas = generator.sample(n_personas)
    prototypes = generator.prototypes()
    renderer = PersonaRenderer()

    params_path = params.save_json(out_dir / "generator_params_initial.json")

    personas_df = pd.DataFrame([p.to_record() for p in personas])
    personas_path = _write_dataframe(personas_df, out_dir / "personas.parquet")

    prototype_records = []
    for p in prototypes:
        rec = p.to_record()
        rec["rendered_profile"] = renderer.render_profile(p)
        prototype_records.append(rec)
    prototypes_df = pd.DataFrame(prototype_records)
    prototypes_path = _write_dataframe(prototypes_df, out_dir / "persona_prototypes.parquet")

    preview_path = out_dir / "persona_preview.txt"
    preview_text = "\n\n".join(renderer.render_structured(p) for p in prototypes)
    preview_path.write_text(preview_text + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 2. Context sets X_sim, X_H, X_calib_int, X_cf.
    # ------------------------------------------------------------------
    context_sets = generate_context_sets(benchmark_section, seed=cfg.seed)
    x_sim_path = _write_dataframe(context_sets.x_sim, out_dir / "X_sim.parquet")
    x_h_path = _write_dataframe(context_sets.x_h, out_dir / "X_H.parquet")
    x_calib_path = _write_dataframe(context_sets.x_calib_int, out_dir / "X_calib_int.parquet")
    x_cf_path = _write_dataframe(context_sets.x_cf, out_dir / "X_cf.parquet")
    x_cf_base_path = _write_dataframe(context_sets.x_cf_base, out_dir / "X_cf_base.parquet")
    context_summary_path = context_sets.save_summary(out_dir / "choice_context_summary.json")

    if _stage_reached("contexts", target_stage):
        _print_stop(
            "contexts",
            out_dir,
            {
                "X_sim": x_sim_path,
                "X_H": x_h_path,
                "X_calib_int": x_calib_path,
                "X_cf_base": x_cf_base_path,
                "X_cf": x_cf_path,
                "context_summary": context_summary_path,
            },
        )
        return

    # ------------------------------------------------------------------
    # 3. Ground-truth synthetic-human population and simulator pi_theta.
    # ------------------------------------------------------------------
    truth_config = dict(gen_section)
    truth_config.update(truth_section)
    truth_params = params_from_config(truth_config)
    truth_generator = MixturePersonaGenerator(params=truth_params, seed=cfg.seed + 101)
    human_personas = truth_generator.sample(n_human_personas)
    human_prototypes = truth_generator.prototypes()

    truth_params_path = truth_params.save_json(out_dir / "generator_params_truth.json")
    human_personas_df = pd.DataFrame([p.to_record() for p in human_personas])
    human_personas_path = _write_dataframe(human_personas_df, out_dir / "human_personas.parquet")

    human_proto_records = []
    for p in human_prototypes:
        rec = p.to_record()
        rec["rendered_profile"] = renderer.render_profile(p)
        human_proto_records.append(rec)
    human_prototypes_path = _write_dataframe(
        pd.DataFrame(human_proto_records), out_dir / "human_persona_prototypes.parquet"
    )

    simulator_cfg = SyntheticSimulatorConfig.from_config(sim_section)
    simulator = RandomUtilityChoiceSimulator(config=simulator_cfg, seed=cfg.seed + 202)

    d_phi = simulator.simulate_long_dataset(
        contexts=context_sets.x_sim,
        personas=personas,
        n_observations=n_obs,
        dataset_label="D_phi_initial",
    )
    d_h = simulator.simulate_long_dataset(
        contexts=context_sets.x_h,
        personas=human_personas,
        n_observations=n_anchor_obs,
        dataset_label="D_H",
    )
    d_calib_int = simulator.simulate_long_dataset(
        contexts=context_sets.x_calib_int,
        personas=human_personas,
        n_observations=n_calib_obs,
        dataset_label="D_calib_int_truth",
    )
    d_cf = simulator.simulate_long_dataset(
        contexts=context_sets.x_cf,
        personas=human_personas,
        n_observations=n_cf_obs,
        dataset_label="D_cf_truth",
    )

    d_phi_path = _write_dataframe(d_phi, out_dir / "D_phi_initial.parquet")
    d_h_path = _write_dataframe(d_h, out_dir / "D_H.parquet")
    d_calib_path = _write_dataframe(d_calib_int, out_dir / "D_calib_int_truth.parquet")
    d_cf_path = _write_dataframe(d_cf, out_dir / "D_cf_truth.parquet")

    simulator_summary = {
        "simulator": "synthetic_random_utility",
        "config": {"choice_temperature": simulator_cfg.choice_temperature},
        "datasets": {
            "D_phi_initial": summarize_choice_dataset(d_phi),
            "D_H": summarize_choice_dataset(d_h),
            "D_calib_int_truth": summarize_choice_dataset(d_calib_int),
            "D_cf_truth": summarize_choice_dataset(d_cf),
        },
    }
    simulator_summary_path = _write_json(simulator_summary, out_dir / "synthetic_choice_summary.json")

    if _stage_reached("datasets", target_stage):
        _print_stop(
            "datasets",
            out_dir,
            {
                "D_phi_initial": d_phi_path,
                "D_H": d_h_path,
                "D_calib_int_truth": d_calib_path,
                "D_cf_truth": d_cf_path,
                "synthetic_choice_summary": simulator_summary_path,
            },
        )
        return

    # ------------------------------------------------------------------
    # 4. Inner economic model m_beta fitted to D_phi_initial.
    # ------------------------------------------------------------------
    inner_section = cfg.require_section("inner_model")
    mnl_cfg = MNLConfig.from_config(inner_section, default_features=params.features)
    mnl_model = MultinomialLogitModel(mnl_cfg)
    mnl_fit = mnl_model.fit(d_phi)
    mnl_fit_path = mnl_fit.save_json(out_dir / "mnl_fit_initial.json")

    mnl_eval = evaluate_fitted_mnl(
        mnl_fit,
        {
            "D_phi_initial": d_phi,
            "D_H": d_h,
            "D_calib_int_truth": d_calib_int,
            "D_cf_truth": d_cf,
        },
    )
    mnl_eval_path = _write_json(mnl_eval, out_dir / "mnl_evaluation_initial.json")

    mnl_pred_phi = predict_probabilities_long(
        d_phi, beta=mnl_fit.beta, features=mnl_fit.features
    )
    mnl_pred_h = predict_probabilities_long(
        d_h, beta=mnl_fit.beta, features=mnl_fit.features
    )
    mnl_pred_calib = predict_probabilities_long(
        d_calib_int, beta=mnl_fit.beta, features=mnl_fit.features
    )
    mnl_pred_cf = predict_probabilities_long(
        d_cf, beta=mnl_fit.beta, features=mnl_fit.features
    )
    mnl_pred_phi_path = _write_dataframe(
        mnl_pred_phi, out_dir / "mnl_probabilities_D_phi_initial.parquet"
    )
    mnl_pred_h_path = _write_dataframe(
        mnl_pred_h, out_dir / "mnl_probabilities_D_H.parquet"
    )
    mnl_pred_calib_path = _write_dataframe(
        mnl_pred_calib, out_dir / "mnl_probabilities_D_calib_int_truth.parquet"
    )
    mnl_pred_cf_path = _write_dataframe(
        mnl_pred_cf, out_dir / "mnl_probabilities_D_cf_truth.parquet"
    )

    if _stage_reached("mnl", target_stage):
        _print_stop(
            "mnl",
            out_dir,
            {
                "mnl_fit": mnl_fit_path,
                "mnl_evaluation": mnl_eval_path,
                "mnl_probabilities_D_phi_initial": mnl_pred_phi_path,
                "mnl_probabilities_D_H": mnl_pred_h_path,
            },
        )
        return

    # ------------------------------------------------------------------
    # 5. Calibration moments M_phi versus M_tar.
    # ------------------------------------------------------------------
    calibration_section = cfg.require_section("calibration")
    calibration_cfg = CalibrationMomentConfig.from_config(
        calibration_section,
        default_features=mnl_fit.features,
        intervened_alternative_id=int(benchmark_section.get("intervened_alternative_id", 0)),
    )
    calibration_report = build_calibration_report(
        anchor_target=d_h,
        anchor_model=mnl_pred_h,
        intervention_target=d_calib_int,
        intervention_model=mnl_pred_calib,
        config=calibration_cfg,
        weighting=str(calibration_section.get("weights", "diagonal_uniform")),
    )
    calibration_report_path = calibration_report.save_json(
        out_dir / "calibration_moments_initial.json"
    )
    calibration_table_path = _write_dataframe(
        calibration_report.table, out_dir / "calibration_moment_table_initial.parquet"
    )

    if _stage_reached("calibration", target_stage):
        _print_stop(
            "calibration",
            out_dir,
            {
                "calibration_moments": calibration_report_path,
                "calibration_table": calibration_table_path,
            },
        )
        return

    # ------------------------------------------------------------------
    # 6. Regularization diagnostics and simple baseline ladder.
    # ------------------------------------------------------------------
    regularization_section = cfg.require_section("regularization")
    regularization_cfg = RegularizationConfig.from_config(regularization_section)
    regularization_initial = regularization_report(params, regularization_cfg)
    regularization_initial_path = regularization_initial.save_json(
        out_dir / "regularization_initial.json"
    )
    evaluation_randomness_path = _write_json(
        {
            "strategy": "common_random_numbers",
            "persona_seed": int(cfg.seed),
            "simulator_seed": int(cfg.seed + 202),
            "description": (
                "All baseline and outer-loop candidates are evaluated with the same "
                "persona-generator RNG seed and simulator RNG seed. The outer-search mutation "
                "RNG remains separate."
            ),
        },
        out_dir / "evaluation_randomness.json",
    )

    diversity_radius = float(regularization_section.get("diversity_radius", 2.5))
    baseline_candidates = default_baseline_candidates(
        initial_params=params,
        truth_params=truth_params,
        diversity_radius=diversity_radius,
    )
    baseline_results = []
    for idx, candidate in enumerate(baseline_candidates):
        result = evaluate_baseline_candidate(
            candidate,
            x_sim=context_sets.x_sim,
            d_h=d_h,
            d_calib_int=d_calib_int,
            d_cf=d_cf,
            simulator_config=simulator_cfg,
            mnl_config=mnl_cfg,
            calibration_config=calibration_cfg,
            regularization_config=regularization_cfg,
            seed=cfg.seed,
            n_personas=n_personas,
            n_observations=n_obs,
            persona_seed=cfg.seed,
            simulator_seed=cfg.seed + 202,
        )
        result.save_json(out_dir / f"baseline_{candidate.name}.json")
        result.calibration_report.save_json(
            out_dir / f"calibration_moments_{candidate.name}.json"
        )
        _write_dataframe(
            result.calibration_report.table,
            out_dir / f"calibration_moment_table_{candidate.name}.parquet",
        )
        candidate.params.save_json(out_dir / f"generator_params_{candidate.name}.json")
        baseline_results.append(result)

    baseline_table = baseline_results_table(baseline_results)
    baseline_results_path = _write_dataframe(baseline_table, out_dir / "baseline_results.parquet")
    baseline_results_json_path = _write_json(
        {
            "candidates": [result.to_dict() for result in baseline_results],
            "summary_rows": baseline_table.to_dict(orient="records"),
        },
        out_dir / "baseline_results.json",
    )

    if _stage_reached("baselines", target_stage):
        _print_stop(
            "baselines",
            out_dir,
            {
                "regularization_initial": regularization_initial_path,
                "baseline_results": baseline_results_path,
                "baseline_results_json": baseline_results_json_path,
            },
        )
        return

    # ------------------------------------------------------------------
    # 7. Full EIPG outer optimizer: update phi.
    # ------------------------------------------------------------------
    outer_section = cfg.require_section("outer_optimizer")
    search_cfg = EvolutionSearchConfig.from_config(outer_section)
    eipg_search = run_evolutionary_search(
        initial_params=params,
        x_sim=context_sets.x_sim,
        d_h=d_h,
        d_calib_int=d_calib_int,
        d_cf=d_cf,
        simulator_config=simulator_cfg,
        mnl_config=mnl_cfg,
        calibration_config=calibration_cfg,
        regularization_config=regularization_cfg,
        search_config=search_cfg,
        seed=cfg.seed + 20_000,
        n_personas=n_personas,
        n_observations=n_obs,
        evaluation_persona_seed=cfg.seed,
        evaluation_simulator_seed=cfg.seed + 202,
    )
    eipg_history = eipg_search.history
    eipg_history_path = _write_dataframe(
        eipg_history, out_dir / "outer_optimizer_history.parquet"
    )
    eipg_history_json_path = _write_json(
        {"history_rows": eipg_history.to_dict(orient="records")},
        out_dir / "outer_optimizer_history.json",
    )
    eipg_result_path = eipg_search.save_json(out_dir / "outer_optimizer_result.json")
    eipg_best_path = eipg_search.best.result.save_json(out_dir / "eipg_result.json")
    eipg_params_path = eipg_search.best.params.save_json(out_dir / "generator_params_eipg.json")
    eipg_calibration_path = eipg_search.best.result.calibration_report.save_json(
        out_dir / "calibration_moments_eipg.json"
    )
    eipg_calibration_table_path = _write_dataframe(
        eipg_search.best.result.calibration_report.table,
        out_dir / "calibration_moment_table_eipg.parquet",
    )

    # Moment-level before/after diagnostic. All candidates are evaluated against
    # the same D_H and D_calib_int targets with common random numbers, so these
    # rows isolate how generator calibration changes each behavioral moment.
    reports_for_comparison = {
        result.candidate.name: result.calibration_report for result in baseline_results
    }
    reports_for_comparison["eipg"] = eipg_search.best.result.calibration_report
    calibration_comparison = compare_calibration_reports(reports_for_comparison)
    calibration_comparison_path = _write_dataframe(
        calibration_comparison, out_dir / "calibration_moment_comparison.parquet"
    )
    calibration_comparison_json_path = _write_json(
        {"rows": calibration_comparison.to_dict(orient="records")},
        out_dir / "calibration_moment_comparison.json",
    )

    # ------------------------------------------------------------------
    # 8. Economic-model capacity diagnostic.
    # ------------------------------------------------------------------
    # v1.8 adds an oracle segmented MNL between the direct simulator oracle and
    # the homogeneous oracle-through-MNL path.  Because this is a controlled
    # benchmark, we can use the true mixture-component labels to fit one MNL per
    # segment and route each target task through its known segment.  This is not
    # a deployable estimator; it is a capacity diagnostic for preference
    # heterogeneity.
    oracle_result = next(
        result for result in baseline_results if result.candidate.name == "oracle_truth"
    )
    static_result = next(
        result for result in baseline_results if result.candidate.name == "static"
    )

    oracle_segment_generator = MixturePersonaGenerator(truth_params, seed=cfg.seed)
    oracle_segment_personas = oracle_segment_generator.sample(n_personas)
    oracle_segment_simulator = RandomUtilityChoiceSimulator(
        simulator_cfg, seed=cfg.seed + 202
    )
    d_phi_oracle_segmented = oracle_segment_simulator.simulate_long_dataset(
        contexts=context_sets.x_sim,
        personas=oracle_segment_personas,
        n_observations=n_obs,
        dataset_label="D_phi_oracle_segmented",
    )
    oracle_segmented_fit = fit_oracle_segmented_mnl(
        d_phi_oracle_segmented,
        mnl_cfg,
        segment_column="persona_segment_id",
    )
    oracle_segmented_pred_h = predict_oracle_segmented_mnl_long(
        d_h, oracle_segmented_fit
    )
    oracle_segmented_pred_calib = predict_oracle_segmented_mnl_long(
        d_calib_int, oracle_segmented_fit
    )
    oracle_segmented_pred_cf = predict_oracle_segmented_mnl_long(
        d_cf, oracle_segmented_fit
    )

    # v1.9.2 fair oracle-mixture comparator: retain oracle training labels when
    # fitting the segment-specific MNLs, but withhold target labels at prediction
    # time. Every target task is predicted by every segment MNL and the results
    # are averaged using training-population segment weights.
    capacity_cfg = cfg.data.get("capacity_diagnostic", {})
    oracle_mixture_weight_basis = str(
        capacity_cfg.get("oracle_mixture_weight_basis", "persona")
    )
    oracle_mixture_pred_h = predict_oracle_mixture_mnl_long(
        d_h, oracle_segmented_fit, weight_basis=oracle_mixture_weight_basis
    )
    oracle_mixture_pred_calib = predict_oracle_mixture_mnl_long(
        d_calib_int, oracle_segmented_fit, weight_basis=oracle_mixture_weight_basis
    )
    oracle_mixture_pred_cf = predict_oracle_mixture_mnl_long(
        d_cf, oracle_segmented_fit, weight_basis=oracle_mixture_weight_basis
    )

    oracle_segmented_calibration = build_calibration_report(
        anchor_target=d_h,
        anchor_model=oracle_segmented_pred_h,
        intervention_target=d_calib_int,
        intervention_model=oracle_segmented_pred_calib,
        config=calibration_cfg,
    )
    oracle_segmented_counterfactual = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=oracle_segmented_pred_cf,
        config=calibration_cfg,
    )
    oracle_mixture_calibration = build_calibration_report(
        anchor_target=d_h,
        anchor_model=oracle_mixture_pred_h,
        intervention_target=d_calib_int,
        intervention_model=oracle_mixture_pred_calib,
        config=calibration_cfg,
    )
    oracle_mixture_counterfactual = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=oracle_mixture_pred_cf,
        config=calibration_cfg,
    )

    d_phi_oracle_segmented_path = _write_dataframe(
        d_phi_oracle_segmented, out_dir / "D_phi_oracle_segmented.parquet"
    )
    oracle_segmented_fit_path = oracle_segmented_fit.save_json(
        out_dir / "oracle_segmented_mnl_fit.json"
    )
    oracle_segmented_coef_path = _write_dataframe(
        segmented_coefficient_table(oracle_segmented_fit),
        out_dir / "oracle_segmented_mnl_coefficients.parquet",
    )
    oracle_segmented_h_path = _write_dataframe(
        oracle_segmented_pred_h,
        out_dir / "oracle_segmented_mnl_probabilities_D_H.parquet",
    )
    oracle_segmented_calib_pred_path = _write_dataframe(
        oracle_segmented_pred_calib,
        out_dir / "oracle_segmented_mnl_probabilities_D_calib_int.parquet",
    )
    oracle_segmented_cf_pred_path = _write_dataframe(
        oracle_segmented_pred_cf,
        out_dir / "oracle_segmented_mnl_probabilities_D_cf.parquet",
    )
    oracle_segmented_calibration_path = oracle_segmented_calibration.save_json(
        out_dir / "oracle_segmented_mnl_calibration_moments.json"
    )
    oracle_segmented_calibration_table_path = _write_dataframe(
        oracle_segmented_calibration.table,
        out_dir / "oracle_segmented_mnl_calibration_moment_table.parquet",
    )
    oracle_segmented_cf_path = oracle_segmented_counterfactual.save_json(
        out_dir / "oracle_segmented_mnl_counterfactual_moments.json"
    )
    oracle_mixture_h_path = _write_dataframe(
        oracle_mixture_pred_h,
        out_dir / "oracle_mixture_mnl_probabilities_D_H.parquet",
    )
    oracle_mixture_calib_pred_path = _write_dataframe(
        oracle_mixture_pred_calib,
        out_dir / "oracle_mixture_mnl_probabilities_D_calib_int.parquet",
    )
    oracle_mixture_cf_pred_path = _write_dataframe(
        oracle_mixture_pred_cf,
        out_dir / "oracle_mixture_mnl_probabilities_D_cf.parquet",
    )
    oracle_mixture_calibration_path = oracle_mixture_calibration.save_json(
        out_dir / "oracle_mixture_mnl_calibration_moments.json"
    )
    oracle_mixture_calibration_table_path = _write_dataframe(
        oracle_mixture_calibration.table,
        out_dir / "oracle_mixture_mnl_calibration_moment_table.parquet",
    )
    oracle_mixture_cf_path = oracle_mixture_counterfactual.save_json(
        out_dir / "oracle_mixture_mnl_counterfactual_moments.json"
    )

    # v1.9: estimate the same number of preference classes without using the
    # ground-truth segment labels. This is a deployable-style capacity
    # diagnostic: class membership is latent and only aggregate class weights
    # plus class-specific MNL coefficients are estimated from D_phi.
    latent_class_k = int(
        capacity_cfg.get(
            "latent_class_k",
            cfg.data.get("persona_generator", {}).get("k_components", 3),
        )
    )
    latent_class_restarts = int(capacity_cfg.get("latent_class_restarts", 6))
    latent_class_em_max_iter = int(capacity_cfg.get("latent_class_em_max_iter", 60))
    latent_class_init_scale = float(capacity_cfg.get("latent_class_init_scale", 0.8))
    latent_class_panel_id_column = str(
        capacity_cfg.get("latent_class_panel_id_column", "persona_id")
    )

    # A homogeneous fit provides a stable center for multi-start class
    # initialization. True segment labels are never supplied to the latent-class
    # estimator or its prediction path.
    latent_class_initial_fit = MultinomialLogitModel(mnl_cfg).fit(
        d_phi_oracle_segmented
    )
    estimated_latent_class_fit = fit_latent_class_mnl(
        d_phi_oracle_segmented,
        mnl_cfg,
        n_classes=latent_class_k,
        seed=cfg.seed + 1909,
        n_restarts=latent_class_restarts,
        em_max_iter=latent_class_em_max_iter,
        init_scale=latent_class_init_scale,
        initial_beta=latent_class_initial_fit.beta,
        panel_id_column=latent_class_panel_id_column,
    )
    estimated_latent_class_pred_h = predict_latent_class_mnl_long(
        d_h, estimated_latent_class_fit
    )
    estimated_latent_class_pred_calib = predict_latent_class_mnl_long(
        d_calib_int, estimated_latent_class_fit
    )
    estimated_latent_class_pred_cf = predict_latent_class_mnl_long(
        d_cf, estimated_latent_class_fit
    )
    estimated_latent_class_calibration = build_calibration_report(
        anchor_target=d_h,
        anchor_model=estimated_latent_class_pred_h,
        intervention_target=d_calib_int,
        intervention_model=estimated_latent_class_pred_calib,
        config=calibration_cfg,
    )
    estimated_latent_class_counterfactual = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=estimated_latent_class_pred_cf,
        config=calibration_cfg,
    )

    estimated_latent_class_fit_path = estimated_latent_class_fit.save_json(
        out_dir / "estimated_latent_class_mnl_fit.json"
    )
    estimated_latent_class_coef_path = _write_dataframe(
        latent_class_coefficient_table(estimated_latent_class_fit),
        out_dir / "estimated_latent_class_mnl_coefficients.parquet",
    )
    estimated_latent_class_posterior_path = _write_dataframe(
        latent_class_panel_posteriors(
            d_phi_oracle_segmented,
            estimated_latent_class_fit,
            panel_id_column=latent_class_panel_id_column,
        ),
        out_dir / "estimated_latent_class_mnl_panel_posteriors.parquet",
    )
    estimated_latent_class_h_path = _write_dataframe(
        estimated_latent_class_pred_h,
        out_dir / "estimated_latent_class_mnl_probabilities_D_H.parquet",
    )
    estimated_latent_class_calib_pred_path = _write_dataframe(
        estimated_latent_class_pred_calib,
        out_dir / "estimated_latent_class_mnl_probabilities_D_calib_int.parquet",
    )
    estimated_latent_class_cf_pred_path = _write_dataframe(
        estimated_latent_class_pred_cf,
        out_dir / "estimated_latent_class_mnl_probabilities_D_cf.parquet",
    )
    estimated_latent_class_calibration_path = estimated_latent_class_calibration.save_json(
        out_dir / "estimated_latent_class_mnl_calibration_moments.json"
    )
    estimated_latent_class_calibration_table_path = _write_dataframe(
        estimated_latent_class_calibration.table,
        out_dir / "estimated_latent_class_mnl_calibration_moment_table.parquet",
    )
    estimated_latent_class_cf_path = estimated_latent_class_counterfactual.save_json(
        out_dir / "estimated_latent_class_mnl_counterfactual_moments.json"
    )

    capacity = build_economic_model_capacity_diagnostic(
        d_h=d_h,
        d_calib_int=d_calib_int,
        d_cf=d_cf,
        oracle_segmented_mnl_calibration=oracle_segmented_calibration,
        oracle_segmented_mnl_counterfactual=oracle_segmented_counterfactual,
        oracle_mixture_mnl_calibration=oracle_mixture_calibration,
        oracle_mixture_mnl_counterfactual=oracle_mixture_counterfactual,
        estimated_latent_class_mnl_calibration=estimated_latent_class_calibration,
        estimated_latent_class_mnl_counterfactual=estimated_latent_class_counterfactual,
        oracle_through_mnl_calibration=oracle_result.calibration_report,
        oracle_through_mnl_counterfactual=oracle_result.counterfactual_report,
        config=calibration_cfg,
        extra_reports={
            "static": static_result.calibration_report,
            "eipg": eipg_search.best.result.calibration_report,
        },
    )
    direct_h = direct_oracle_probability_rows(
        d_h, output_probability_column=calibration_cfg.probability_column
    )
    direct_calib = direct_oracle_probability_rows(
        d_calib_int, output_probability_column=calibration_cfg.probability_column
    )
    direct_cf = direct_oracle_probability_rows(
        d_cf, output_probability_column=calibration_cfg.probability_column
    )
    direct_h_path = _write_dataframe(
        direct_h, out_dir / "direct_oracle_probabilities_D_H.parquet"
    )
    direct_calib_path = _write_dataframe(
        direct_calib, out_dir / "direct_oracle_probabilities_D_calib_int.parquet"
    )
    direct_cf_path = _write_dataframe(
        direct_cf, out_dir / "direct_oracle_probabilities_D_cf.parquet"
    )
    direct_calibration_path = capacity.direct_oracle_calibration.save_json(
        out_dir / "direct_oracle_calibration_moments.json"
    )
    direct_calibration_table_path = _write_dataframe(
        capacity.direct_oracle_calibration.table,
        out_dir / "direct_oracle_calibration_moment_table.parquet",
    )
    direct_cf_report_path = capacity.direct_oracle_counterfactual.save_json(
        out_dir / "direct_oracle_counterfactual_moments.json"
    )
    capacity_comparison_path = _write_dataframe(
        capacity.moment_comparison,
        out_dir / "economic_model_capacity_moment_comparison.parquet",
    )
    capacity_summary_path = _write_json(
        capacity.summary(), out_dir / "economic_model_capacity_summary.json"
    )

    baseline_plus_eipg = pd.concat(
        [baseline_table, pd.DataFrame([eipg_search.best.result.summary_row()])],
        ignore_index=True,
    )
    baseline_plus_eipg.loc[baseline_plus_eipg.index[-1], "candidate"] = "full_eipg"
    baseline_plus_eipg_path = _write_dataframe(
        baseline_plus_eipg, out_dir / "baseline_plus_eipg_results.parquet"
    )
    baseline_plus_eipg_json_path = _write_json(
        {"summary_rows": baseline_plus_eipg.to_dict(orient="records")},
        out_dir / "baseline_plus_eipg_results.json",
    )

    print("EIPG clean v1.9.2 is working.")
    print(f"  run_name:       {cfg.run_name}")
    print(f"  seed:           {cfg.seed}")
    print(f"  output_dir:     {out_dir}")
    print(f"  manifest:       {manifest_path}")
    print(f"  config_used:    {config_copy_path}")
    print(f"  run_metadata:   {run_metadata_path}")
    print(f"  params:         {params_path}")
    print(f"  truth params:   {truth_params_path}")
    print(f"  personas:       {personas_path}  ({len(personas)} rows)")
    print(f"  human personas: {human_personas_path}  ({len(human_personas)} rows)")
    print(f"  prototypes:     {prototypes_path}  ({len(prototypes)} rows)")
    print(f"  human protos:   {human_prototypes_path}  ({len(human_prototypes)} rows)")
    print(f"  preview:        {preview_path}")
    print(f"  X_sim:          {x_sim_path}  ({context_sets.x_sim['context_id'].nunique()} contexts)")
    print(f"  X_H:            {x_h_path}  ({context_sets.x_h['context_id'].nunique()} contexts)")
    print(f"  X_calib_int:    {x_calib_path}  ({context_sets.x_calib_int['context_id'].nunique()} contexts)")
    print(f"  X_cf_base:      {x_cf_base_path}  ({context_sets.x_cf_base['context_id'].nunique()} contexts)")
    print(f"  X_cf:           {x_cf_path}  ({context_sets.x_cf['context_id'].nunique()} contexts)")
    print(f"  context summary:{context_summary_path}")
    print(f"  D_phi_initial:  {d_phi_path}  ({d_phi['observation_id'].nunique()} observations)")
    print(f"  D_H:            {d_h_path}  ({d_h['observation_id'].nunique()} observations)")
    print(f"  D_calib_int:    {d_calib_path}  ({d_calib_int['observation_id'].nunique()} observations)")
    print(f"  D_cf_truth:     {d_cf_path}  ({d_cf['observation_id'].nunique()} observations)")
    print(f"  choice summary: {simulator_summary_path}")
    print(f"  MNL fit:        {mnl_fit_path}")
    print(f"  MNL eval:       {mnl_eval_path}")
    print(f"  MNL probs phi:  {mnl_pred_phi_path}")
    print(f"  MNL probs D_H:  {mnl_pred_h_path}")
    print(f"  MNL probs calib:{mnl_pred_calib_path}")
    print(f"  MNL probs cf:   {mnl_pred_cf_path}")
    print(f"  calib moments:  {calibration_report_path}")
    print(f"  calib table:    {calibration_table_path}")
    print(f"  regularization: {regularization_initial_path}")
    print(f"  eval randomness:{evaluation_randomness_path}")
    print(f"  baseline table: {baseline_results_path}")
    print(f"  baseline json:  {baseline_results_json_path}")
    print(f"  EIPG history:   {eipg_history_path}")
    print(f"  EIPG hist json: {eipg_history_json_path}")
    print(f"  EIPG result:    {eipg_result_path}")
    print(f"  EIPG best:      {eipg_best_path}")
    print(f"  EIPG params:    {eipg_params_path}")
    print(f"  EIPG calib:     {eipg_calibration_path}")
    print(f"  EIPG calib tbl: {eipg_calibration_table_path}")
    print(f"  moment compare: {calibration_comparison_path}")
    print(f"  moment cmp json:{calibration_comparison_json_path}")
    print(f"  oracle seg Dphi:{d_phi_oracle_segmented_path}")
    print(f"  oracle seg fit: {oracle_segmented_fit_path}")
    print(f"  oracle seg coef:{oracle_segmented_coef_path}")
    print(f"  oracle seg H:   {oracle_segmented_h_path}")
    print(f"  oracle seg I:   {oracle_segmented_calib_pred_path}")
    print(f"  oracle seg C:   {oracle_segmented_cf_pred_path}")
    print(f"  oracle seg calib:{oracle_segmented_calibration_path}")
    print(f"  oracle seg tbl: {oracle_segmented_calibration_table_path}")
    print(f"  oracle seg CF:  {oracle_segmented_cf_path}")
    print(f"  oracle mix H:   {oracle_mixture_h_path}")
    print(f"  oracle mix I:   {oracle_mixture_calib_pred_path}")
    print(f"  oracle mix C:   {oracle_mixture_cf_pred_path}")
    print(f"  oracle mix calib:{oracle_mixture_calibration_path}")
    print(f"  oracle mix tbl: {oracle_mixture_calibration_table_path}")
    print(f"  oracle mix CF:  {oracle_mixture_cf_path}")
    print(f"  latent cls fit: {estimated_latent_class_fit_path}")
    print(f"  latent cls coef:{estimated_latent_class_coef_path}")
    print(f"  latent cls post:{estimated_latent_class_posterior_path}")
    print(f"  latent cls H:   {estimated_latent_class_h_path}")
    print(f"  latent cls I:   {estimated_latent_class_calib_pred_path}")
    print(f"  latent cls C:   {estimated_latent_class_cf_pred_path}")
    print(f"  latent cls calib:{estimated_latent_class_calibration_path}")
    print(f"  latent cls tbl: {estimated_latent_class_calibration_table_path}")
    print(f"  latent cls CF:  {estimated_latent_class_cf_path}")
    print(f"  direct oracle H:{direct_h_path}")
    print(f"  direct oracle I:{direct_calib_path}")
    print(f"  direct oracle C:{direct_cf_path}")
    print(f"  direct calib:   {direct_calibration_path}")
    print(f"  direct calib tbl:{direct_calibration_table_path}")
    print(f"  direct CF:      {direct_cf_report_path}")
    print(f"  capacity cmp:   {capacity_comparison_path}")
    print(f"  capacity summary:{capacity_summary_path}")
    print(f"  comparison:     {baseline_plus_eipg_path}")
    print(f"  comp json:      {baseline_plus_eipg_json_path}")
    print("\nMNL beta estimates fitted on D_phi_initial:")
    for feature, value in mnl_fit.beta_by_feature.items():
        print(f"  - {feature}: {value: .4f}")
    print("\nRendered prototype examples:")
    for p in prototypes:
        print(f"  - {renderer.render_profile(p)}")
    print("\nCalibration summary:")
    summary = calibration_report.summary()
    print(f"  - n_moments:     {summary['n_moments']}")
    print(f"  - l2_error:      {summary['l2_error']:.4f}")
    print(f"  - rmse:          {summary['rmse']:.4f}")
    print(f"  - max_abs_error: {summary['max_abs_error']:.4f}")
    print("\nRegularization summary for initial generator:")
    for key, value in regularization_initial.terms.items():
        print(f"  - {key}: {value:.4f}")
    print("\nBaseline + EIPG summary:")
    display_cols = [
        "candidate",
        "calibration_l2_error",
        "cf_l2_error",
        "anchor_nll_per_observation",
        "cf_nll_per_observation",
        "mixture_entropy",
        "avg_pairwise_mean_distance",
    ]
    for _, row in baseline_plus_eipg[display_cols].iterrows():
        print(
            "  - {candidate}: calib_l2={calibration_l2_error:.4f}, "
            "cf_l2={cf_l2_error:.4f}, anchor_nll={anchor_nll_per_observation:.4f}, "
            "cf_nll={cf_nll_per_observation:.4f}, H={mixture_entropy:.4f}, "
            "disp={avg_pairwise_mean_distance:.4f}".format(**row.to_dict())
        )
    print("\nOuter optimizer summary:")
    print(f"  - evaluations: {len(eipg_search.evaluations)}")
    print(f"  - best candidate: {eipg_search.best.candidate_name}")
    print(f"  - best objective: {eipg_search.best.objective_value:.4f}")
    for key, value in eipg_search.best.objective_components.items():
        print(f"  - {key}: {value:.4f}")
    print("\nEconomic-model capacity diagnostic:")
    capacity_summary = capacity.summary()
    print(
        "  - direct oracle: calib_l2={:.4f}, cf_l2={:.4f}".format(
            capacity_summary["direct_oracle"]["calibration_l2_error"],
            capacity_summary["direct_oracle"]["cf_l2_error"],
        )
    )
    print(
        "  - oracle segmented MNL: calib_l2={:.4f}, cf_l2={:.4f}".format(
            capacity_summary["oracle_segmented_mnl"]["calibration_l2_error"],
            capacity_summary["oracle_segmented_mnl"]["cf_l2_error"],
        )
    )
    print(
        "  - oracle mixture MNL (no target labels): calib_l2={:.4f}, cf_l2={:.4f}".format(
            capacity_summary["oracle_mixture_mnl"]["calibration_l2_error"],
            capacity_summary["oracle_mixture_mnl"]["cf_l2_error"],
        )
    )
    print(
        "  - estimated panel latent-class MNL: calib_l2={:.4f}, cf_l2={:.4f}".format(
            capacity_summary["estimated_latent_class_mnl"]["calibration_l2_error"],
            capacity_summary["estimated_latent_class_mnl"]["cf_l2_error"],
        )
    )
    print(
        "  - oracle through homogeneous MNL: calib_l2={:.4f}, cf_l2={:.4f}".format(
            capacity_summary["oracle_through_mnl"]["calibration_l2_error"],
            capacity_summary["oracle_through_mnl"]["cf_l2_error"],
        )
    )
    print(
        "  - segmented excess over direct: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["excess_l2_after_segmented_mnl"]["calibration"],
            capacity_summary["excess_l2_after_segmented_mnl"]["counterfactual"],
        )
    )
    print(
        "  - oracle mixture excess over direct: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["excess_l2_after_oracle_mixture_mnl"]["calibration"],
            capacity_summary["excess_l2_after_oracle_mixture_mnl"]["counterfactual"],
        )
    )
    print(
        "  - target-routing advantage (mixture minus routed): calib={:.4f}, cf={:.4f}".format(
            capacity_summary["oracle_mixture_minus_routed_segmented_l2"]["calibration"],
            capacity_summary["oracle_mixture_minus_routed_segmented_l2"]["counterfactual"],
        )
    )
    print(
        "  - estimated LC excess over direct: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["excess_l2_after_estimated_latent_class_mnl"]["calibration"],
            capacity_summary["excess_l2_after_estimated_latent_class_mnl"]["counterfactual"],
        )
    )
    print(
        "  - homogeneous minus estimated LC: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["homogeneous_minus_estimated_latent_class_l2"]["calibration"],
            capacity_summary["homogeneous_minus_estimated_latent_class_l2"]["counterfactual"],
        )
    )
    print(
        "  - estimated LC minus oracle mixture: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["estimated_minus_oracle_mixture_l2"]["calibration"],
            capacity_summary["estimated_minus_oracle_mixture_l2"]["counterfactual"],
        )
    )
    print(
        "  - estimated LC minus routed oracle segmented: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["estimated_minus_oracle_segmented_l2"]["calibration"],
            capacity_summary["estimated_minus_oracle_segmented_l2"]["counterfactual"],
        )
    )
    print(
        "  - homogeneous minus segmented: calib={:.4f}, cf={:.4f}".format(
            capacity_summary["homogeneous_minus_segmented_l2"]["calibration"],
            capacity_summary["homogeneous_minus_segmented_l2"]["counterfactual"],
        )
    )
    print("\nNext diagnostic: build Appendix E and compare routed-oracle, oracle-mixture, estimated latent-class, and homogeneous-MNL substitution responses.")

if __name__ == "__main__":
    main()
