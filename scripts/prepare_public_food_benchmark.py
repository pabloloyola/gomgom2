#!/usr/bin/env python3
"""Prepare the frozen Nutri2Cycle public benchmark and fit the human reference MNL.

This stage is intentionally pre-EIPG. It reads only the frozen calibration tasks when
computing human targets and fitting the reference model. The final holdout choice column
is not read into any target, diagnostic, or metric in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eipg.econ.mnl import FittedMNL, MNLConfig, MultinomialLogitModel, predict_probabilities_long


ALTERNATIVES = ("organic", "conventional", "circular", "none")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", low_memory=False)


def make_long(
    respondents: pd.DataFrame,
    cfg: dict[str, Any],
    task_ids: list[int],
    *,
    include_choices: bool,
) -> pd.DataFrame:
    dataset_cfg = cfg["dataset"]
    task_cfg = cfg["tasks"]
    choice_codes = {str(k): int(v) for k, v in dataset_cfg["choice_codes"].items()}
    records: list[dict[str, Any]] = []

    for task_id in task_ids:
        t = task_cfg[int(task_id)]
        choice_col = str(t["column"])
        prices = {str(k): float(v) for k, v in t["prices"].items()}
        chosen_codes = pd.to_numeric(respondents[choice_col], errors="coerce") if include_choices else None

        for row_pos, (_, r) in enumerate(respondents.iterrows()):
            respondent_id = r[dataset_cfg["respondent_id"]]
            observation_id = f"{respondent_id}__bread_t{task_id}"
            chosen_code = int(chosen_codes.iloc[row_pos]) if include_choices else None
            for alt_idx, alt in enumerate(ALTERNATIVES):
                rec = {
                    "respondent_id": respondent_id,
                    "observation_id": observation_id,
                    "task_id": int(task_id),
                    "alternative_id": alt,
                    "alternative_index": alt_idx,
                    "price": prices[alt],
                    "asc_organic": 1.0 if alt == "organic" else 0.0,
                    "asc_circular": 1.0 if alt == "circular" else 0.0,
                    "asc_none": 1.0 if alt == "none" else 0.0,
                }
                if include_choices:
                    rec["chosen"] = int(chosen_code == choice_codes[alt])
                records.append(rec)
    return pd.DataFrame.from_records(records)


def shares(long_df: pd.DataFrame) -> dict[str, float]:
    chosen = long_df.loc[long_df["chosen"].astype(int).eq(1)]
    s = chosen["alternative_id"].value_counts(normalize=True)
    return {alt: float(s.get(alt, 0.0)) for alt in ALTERNATIVES}


def chosen_mean_price(long_df: pd.DataFrame) -> float:
    chosen = long_df.loc[long_df["chosen"].astype(int).eq(1)]
    return float(chosen["price"].mean())


def response_vector(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {alt: float(a[alt] - b[alt]) for alt in ALTERNATIVES}


def human_moments(calib: pd.DataFrame, anchor_task: int = 2) -> dict[str, Any]:
    pooled = shares(calib)
    task_shares: dict[str, dict[str, float]] = {}
    for task_id, g in calib.groupby("task_id", sort=True):
        task_shares[str(int(task_id))] = shares(g)
    anchor = task_shares[str(anchor_task)]
    task_responses = {
        task: response_vector(s, anchor)
        for task, s in task_shares.items()
        if int(task) != anchor_task
    }
    return {
        "pooled_choice_shares": pooled,
        "pooled_mean_chosen_price": chosen_mean_price(calib),
        "task_choice_shares": task_shares,
        "anchor_task": anchor_task,
        "task_minus_anchor_share_responses": task_responses,
    }


def model_implied_shares(pred: pd.DataFrame) -> dict[str, float]:
    # Equal weight per observed choice occasion, matching empirical share aggregation.
    obs_alt = pred.groupby(["observation_id", "alternative_id"], as_index=False)["mnl_prob"].first()
    out = obs_alt.groupby("alternative_id")["mnl_prob"].mean()
    return {alt: float(out.get(alt, 0.0)) for alt in ALTERNATIVES}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/public_food_benchmark.yaml"))
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("outputs/public_food_benchmark_v1/pre_eipg"))
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    raw = read_raw_csv(args.csv)
    country = raw.loc[pd.to_numeric(raw[dcfg["country_column"]], errors="coerce").eq(int(dcfg["country_code"]))].copy()
    purchasers = country.loc[pd.to_numeric(country[dcfg["purchase_column"]], errors="coerce").eq(int(dcfg["purchase_required_value"]))].copy()

    calibration_tasks = [int(x) for x in cfg["split"]["calibration_tasks"]]
    calibration_cols = [cfg["tasks"][t]["column"] for t in calibration_tasks]
    final_tasks = [int(x) for x in cfg["split"]["final_holdout_tasks"]]
    final_cols = [cfg["tasks"][t]["column"] for t in final_tasks]
    valid_codes = set(int(v) for v in dcfg["choice_codes"].values())

    # Inclusion for calibration depends only on calibration-task validity.
    calibration_valid = purchasers[calibration_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").isin(valid_codes)
    ).all(axis=1)
    calibration_people = purchasers.loc[calibration_valid].copy()

    # We may report how many records would be evaluable later, but never inspect their choices.
    final_valid = purchasers[final_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").isin(valid_codes)
    ).all(axis=1)
    n_final_evaluable = int((calibration_valid & final_valid).sum())

    calib_long = make_long(calibration_people, cfg, calibration_tasks, include_choices=True)
    final_contexts = make_long(calibration_people, cfg, final_tasks, include_choices=False)

    # Canonical data products. The final artifact deliberately contains no chosen label.
    calib_long.to_parquet(out / "human_calibration_long.parquet", index=False)
    final_contexts.to_parquet(out / "final_holdout_contexts_no_choices.parquet", index=False)

    features = ("price", "asc_organic", "asc_circular", "asc_none")
    model = MultinomialLogitModel(MNLConfig(features=features, l2=1.0e-4, max_iter=500))
    fit = model.fit(calib_long)
    fit.save_json(out / "human_reference_mnl.json")
    pred = predict_probabilities_long(calib_long, beta=fit.beta, features=features)
    pred.to_parquet(out / "human_reference_calibration_predictions.parquet", index=False)

    moments = human_moments(calib_long, anchor_task=2)
    (out / "human_calibration_moments.json").write_text(json.dumps(moments, indent=2), encoding="utf-8")

    fitted = FittedMNL(fit)
    empirical_by_task = moments["task_choice_shares"]
    implied_by_task: dict[str, dict[str, float]] = {}
    for task_id, g in pred.groupby("task_id", sort=True):
        implied_by_task[str(int(task_id))] = model_implied_shares(g)

    beta_price = float(fit.beta_by_feature["price"])
    sanity = {
        "protocol": cfg["protocol"]["name"],
        "country": dcfg["country_name"],
        "product": dcfg["product"],
        "n_country_rows": int(len(country)),
        "n_purchasers": int(len(purchasers)),
        "n_calibration_respondents": int(len(calibration_people)),
        "n_calibration_observations": int(calib_long["observation_id"].nunique()),
        "n_final_evaluable_later": n_final_evaluable,
        "final_choices_consumed": False,
        "mnl_success": bool(fit.success),
        "mnl_message": fit.message,
        "beta_by_feature": fit.beta_by_feature,
        "price_coefficient_negative": bool(beta_price < 0),
        "calibration_nll_per_observation": float(fitted.nll_per_observation(calib_long)),
        "calibration_accuracy": float(fitted.accuracy(calib_long)),
        "empirical_task_shares": empirical_by_task,
        "model_implied_task_shares": implied_by_task,
    }
    (out / "pre_eipg_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    lines = [
        "# Nutri2Cycle public benchmark: pre-EIPG sanity gate",
        "",
        f"- Protocol: `{sanity['protocol']}`",
        f"- Country/product: **{sanity['country']} / {sanity['product']}**",
        f"- Calibration respondents: **{sanity['n_calibration_respondents']:,}**",
        f"- Calibration choice occasions: **{sanity['n_calibration_observations']:,}**",
        f"- Final-evaluable respondents later: **{sanity['n_final_evaluable_later']:,}**",
        "- Final holdout choices consumed in this stage: **NO**",
        f"- MNL converged: **{sanity['mnl_success']}**",
        f"- Price coefficient: **{beta_price:.4f}**",
        f"- Calibration NLL / observation: **{sanity['calibration_nll_per_observation']:.4f}**",
        f"- Calibration top-choice accuracy: **{sanity['calibration_accuracy']:.3f}**",
        "",
        "## Human MNL coefficients",
        "",
    ]
    for k, v in fit.beta_by_feature.items():
        lines.append(f"- `{k}`: {v:.6f}")
    lines += ["", "## Gate", ""]
    if fit.success and beta_price < 0:
        lines.append("**PASS:** reference MNL converged and the generic price coefficient is negative. The public benchmark can proceed to generator initialization and EIPG implementation without consulting Task-4 outcomes.")
    else:
        lines.append("**REVIEW REQUIRED:** do not run EIPG until the reference-model issue is understood. No Task-4 outcome should be consulted while resolving it.")
    (out / "PRE_EIPG_SANITY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # Hard fail only on structural/model sanity; never on final-test behavior.
    if not fit.success:
        raise SystemExit("Reference MNL did not converge")
    if beta_price >= 0:
        raise SystemExit(f"Expected negative price coefficient; obtained {beta_price:.6f}")


if __name__ == "__main__":
    main()
