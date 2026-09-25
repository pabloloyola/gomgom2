#!/usr/bin/env python3
"""Test whether richer calibration interventions improve identification.

This is a development diagnostic, not a final paper-test evaluator.  A fixed
candidate bank is generated once.  Every candidate is fit through the same
homogeneous MNL and then scored under increasingly informative calibration
bundles.  We ask whether calibration scores better rank candidates by CF-dev
error as interventions are added.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eipg.econ import MNLConfig, MultinomialLogitModel, predict_probabilities_long
from eipg.experiments import InterventionSpec, apply_intervention, combine_intervention_reports
from eipg.objectives import CalibrationMomentConfig, RegularizationConfig, build_calibration_report
from eipg.outeropt.evolution import mutate_generator_params
from eipg.outeropt.weighted_evolution import block_weighted_calibration_score
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator, PersonaLatent
from eipg.reporting.io import load_json, load_yaml, read_table, resolve_run_dir
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None)
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/controlled_synthetic_paperlike"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multi_intervention_identification.yaml"),
    )
    p.add_argument("--n-candidates", type=int, default=None)
    return p.parse_args()


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


def read_any(run_dir: Path, stem: str) -> pd.DataFrame:
    return read_table(run_dir, stem)


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
        if spec.name in specs:
            raise ValueError(f"duplicate intervention name: {spec.name}")
        specs[spec.name] = spec
    return specs


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    value = x.corr(y)
    return float(value) if pd.notna(value) else float("nan")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def main() -> None:
    args = parse_args()
    source = resolve_run_dir(run=args.run, experiment_dir=args.experiment_dir)
    source_cfg = load_yaml(source / "config_used.yaml")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    exp = cfg.get("experiment", {})

    seed = int(exp.get("seed", 707))
    n_candidates = int(args.n_candidates or exp.get("n_candidates", 48))
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    n_cal_contexts = int(exp.get("n_calibration_contexts_per_intervention", 120))
    n_cal_obs = int(exp.get("n_calibration_observations_per_intervention", 600))

    initial = params_from_json(source / "generator_params_initial.json")
    x_sim = read_any(source, "X_sim")
    x_h = read_any(source, "X_H")
    d_h = read_any(source, "D_H")
    d_cf = read_any(source, "D_cf_truth")
    human_table = read_any(source, "human_personas")
    human_personas = personas_from_table(human_table, initial.features)

    sim_cfg = SyntheticSimulatorConfig.from_config(source_cfg["simulation"])
    mnl_cfg = MNLConfig.from_config(source_cfg["inner_model"], default_features=initial.features)
    calib_cfg = CalibrationMomentConfig.from_config(
        source_cfg["calibration"],
        default_features=initial.features,
        intervened_alternative_id=int(source_cfg["benchmark"].get("intervened_alternative_id", 0)),
    )
    reg_cfg = RegularizationConfig.from_config(source_cfg["regularization"])
    _ = reg_cfg  # kept explicit: this experiment intentionally does not select on regularization.

    specs = intervention_specs(cfg.get("interventions", []))
    bundles: dict[str, list[str]] = {
        str(k): [str(v) for v in values]
        for k, values in (cfg.get("bundles", {}) or {}).items()
    }
    profiles: dict[str, dict[str, float]] = {
        str(k): {str(b): float(w) for b, w in values.items()}
        for k, values in (cfg.get("scoring_profiles", {}) or {}).items()
    }
    if not specs or not bundles or not profiles:
        raise ValueError("config must define interventions, bundles, and scoring_profiles")
    for bundle, names in bundles.items():
        missing = [n for n in names if n not in specs]
        if missing:
            raise ValueError(f"bundle {bundle} references unknown interventions: {missing}")

    # Generate new calibration-development interventions from the same anchor
    # contexts and the same synthetic-human population. These are additional
    # development data, not final-test data.
    intervention_contexts: dict[str, pd.DataFrame] = {}
    intervention_targets: dict[str, pd.DataFrame] = {}
    for idx, (name, spec) in enumerate(specs.items()):
        x_int = apply_intervention(
            x_h,
            spec,
            prefix=f"multi_{name}",
            context_set=f"X_calib_multi_{name}",
            n_contexts=n_cal_contexts,
        )
        simulator = RandomUtilityChoiceSimulator(sim_cfg, seed=seed + 1_000 + idx)
        d_int = simulator.simulate_long_dataset(
            contexts=x_int,
            personas=human_personas,
            n_observations=n_cal_obs,
            dataset_label=f"D_calib_multi_{name}",
        )
        intervention_contexts[name] = x_int
        intervention_targets[name] = d_int

    # Fixed candidate bank: one initial generator plus independent mutations of
    # that same center. This isolates score informativeness from search-path effects.
    rng = np.random.default_rng(seed)
    bank: list[tuple[str, MixtureGeneratorParams]] = [("candidate_000_initial", initial)]
    for i in range(1, n_candidates):
        bank.append(
            (
                f"candidate_{i:03d}",
                mutate_generator_params(
                    initial,
                    rng=rng,
                    mean_scale=float(exp.get("candidate_mean_mutation_scale", 0.40)),
                    weight_logit_scale=float(exp.get("candidate_weight_logit_mutation_scale", 0.30)),
                    mean_clip=float(source_cfg.get("outer_optimizer", {}).get("mean_clip", 4.0)),
                ),
            )
        )

    n_personas = int(source_cfg["simulation"].get("n_personas", 160))
    n_obs = int(source_cfg["simulation"].get("n_obs", 1200))
    rows: list[dict[str, Any]] = []
    moment_rows: list[pd.DataFrame] = []

    for i, (candidate_name, params) in enumerate(bank, start=1):
        print(f"[{i:03d}/{len(bank):03d}] {candidate_name}", flush=True)
        generator = MixturePersonaGenerator(params, seed=seed + 20_000)
        personas = generator.sample(n_personas)
        simulator = RandomUtilityChoiceSimulator(sim_cfg, seed=seed + 30_000)
        d_phi = simulator.simulate_long_dataset(
            contexts=x_sim,
            personas=personas,
            n_observations=n_obs,
            dataset_label=f"D_phi_{candidate_name}",
        )
        fit = MultinomialLogitModel(mnl_cfg).fit(d_phi)
        pred_h = predict_probabilities_long(d_h, beta=fit.beta, features=fit.features)
        pred_cf = predict_probabilities_long(d_cf, beta=fit.beta, features=fit.features)
        cf_report = build_calibration_report(
            anchor_target=d_cf,
            anchor_model=pred_cf,
            config=calib_cfg,
        )
        cf_l2 = float(cf_report.summary()["l2_error"])

        per_intervention = {}
        for name, target in intervention_targets.items():
            pred = predict_probabilities_long(target, beta=fit.beta, features=fit.features)
            report = build_calibration_report(
                anchor_target=d_h,
                anchor_model=pred_h,
                intervention_target=target,
                intervention_model=pred,
                config=calib_cfg,
            )
            per_intervention[name] = report

        for bundle_name, names in bundles.items():
            combined = combine_intervention_reports((name, per_intervention[name]) for name in names)
            mt = combined.table.copy()
            mt.insert(0, "candidate", candidate_name)
            mt.insert(1, "bundle", bundle_name)
            moment_rows.append(mt)
            for profile_name, weights in profiles.items():
                score, diagnostics = block_weighted_calibration_score(
                    combined, block_weights=weights
                )
                rows.append(
                    {
                        "candidate": candidate_name,
                        "candidate_index": i - 1,
                        "bundle": bundle_name,
                        "n_interventions": len(names),
                        "interventions": "+".join(names),
                        "scoring_profile": profile_name,
                        "calibration_score": float(score),
                        "raw_calibration_l2": float(combined.summary()["l2_error"]),
                        "cf_dev_l2": cf_l2,
                        "train_nll_per_observation": float(fit.train_nll_per_observation),
                        **{
                            k: float(v)
                            for k, v in diagnostics.items()
                            if k.startswith("block_") and k.endswith("_rmse")
                        },
                    }
                )

    candidate_scores = pd.DataFrame(rows)
    moments = pd.concat(moment_rows, ignore_index=True)

    summaries: list[dict[str, Any]] = []
    for (bundle, profile), g in candidate_scores.groupby(["bundle", "scoring_profile"], sort=True):
        g = g.sort_values("calibration_score").reset_index(drop=True)
        top_k = max(1, int(np.ceil(0.10 * len(g))))
        summaries.append(
            {
                "bundle": bundle,
                "scoring_profile": profile,
                "n_interventions": int(g["n_interventions"].iloc[0]),
                "n_candidates": int(len(g)),
                "pearson_calibration_vs_cf": safe_corr(g["calibration_score"], g["cf_dev_l2"]),
                "spearman_calibration_vs_cf": spearman(g["calibration_score"], g["cf_dev_l2"]),
                "best_calibration_candidate": str(g.iloc[0]["candidate"]),
                "best_calibration_score": float(g.iloc[0]["calibration_score"]),
                "cf_of_best_calibration_candidate": float(g.iloc[0]["cf_dev_l2"]),
                "top10pct_mean_cf_dev_l2": float(g.head(top_k)["cf_dev_l2"].mean()),
                "candidate_bank_mean_cf_dev_l2": float(g["cf_dev_l2"].mean()),
                "candidate_bank_min_cf_dev_l2": float(g["cf_dev_l2"].min()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["scoring_profile", "n_interventions", "bundle"]
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = source / str(cfg.get("output", {}).get("dirname", "multi_intervention_identification")) / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    score_path = write_table(candidate_scores, out_root / "candidate_scores.parquet")
    moment_path = write_table(moments, out_root / "moment_table.parquet")
    summary_path = write_table(summary, out_root / "identification_summary.parquet")
    summary.to_csv(out_root / "identification_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source),
        "role": "development_identification_diagnostic",
        "cf_role": "CF-dev; not final untouched test",
        "seed": seed,
        "n_candidates": n_candidates,
        "n_calibration_contexts_per_intervention": n_cal_contexts,
        "n_calibration_observations_per_intervention": n_cal_obs,
        "interventions": [spec.__dict__ for spec in specs.values()],
        "bundles": bundles,
        "scoring_profiles": profiles,
        "artifacts": {
            "candidate_scores": str(score_path),
            "moment_table": str(moment_path),
            "identification_summary": str(summary_path),
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "config_used.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    print("\n=== MULTI-INTERVENTION IDENTIFICATION SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nArtifacts: {out_root}")


if __name__ == "__main__":
    main()
