#!/usr/bin/env python3
"""Build the frozen Nutri2Cycle public-food calibration dataset and human MNL reference.

This script is intentionally *pre-EIPG*. It reads only calibration-task choice outcomes
when constructing targets and fitting the human reference MNL. The final-holdout task
is represented only by its predeclared design in the protocol; its observed choices are
not summarized or evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eipg.econ.mnl import FittedMNL, MNLConfig, MultinomialLogitModel


ALTERNATIVES = ("organic", "conventional", "circular", "none")
FEATURES = ("price", "asc_organic", "asc_conventional", "asc_circular")


def _load_protocol(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", low_memory=False)


def _eligible_respondents(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    ds = cfg["dataset"]
    choice_cols = list(ds["complete_case_choice_columns"])
    choice_codes = {int(v) for v in ds["choice_codes"].values()}

    country = pd.to_numeric(raw[ds["country_column"]], errors="coerce").eq(int(ds["country_code"]))
    purchaser = pd.to_numeric(raw[ds["purchase_column"]], errors="coerce").eq(
        int(ds["purchase_required_value"])
    )
    valid_panel = raw[choice_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce").isin(choice_codes)
    ).all(axis=1)
    return raw.loc[country & purchaser & valid_panel].copy()


def _choice_code_map(cfg: dict[str, Any]) -> dict[int, str]:
    return {int(code): name for name, code in cfg["dataset"]["choice_codes"].items()}


def build_long_calibration(panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["dataset"]["respondent_id"]
    product = cfg["dataset"]["product"]
    code_to_alt = _choice_code_map(cfg)
    records: list[dict[str, Any]] = []

    for task in cfg["split"]["calibration_tasks"]:
        spec = cfg["tasks"][int(task)]
        col = spec["column"]
        prices = spec["prices"]
        for _, row in panel.iterrows():
            respondent_id = row[id_col]
            chosen_alt = code_to_alt[int(row[col])]
            observation_id = f"{respondent_id}::{product}::{int(task)}"
            for alt in ALTERNATIVES:
                records.append(
                    {
                        "respondent_id": respondent_id,
                        "task_id": int(task),
                        "observation_id": observation_id,
                        "alternative_id": alt,
                        "chosen": int(alt == chosen_alt),
                        "price": float(prices[alt]),
                        "asc_organic": float(alt == "organic"),
                        "asc_conventional": float(alt == "conventional"),
                        "asc_circular": float(alt == "circular"),
                    }
                )
    out = pd.DataFrame(records)
    out["task_id"] = out["task_id"].astype(int)
    return out


def _shares(df: pd.DataFrame) -> dict[str, float]:
    chosen = df.loc[df["chosen"].eq(1), "alternative_id"]
    return {alt: float((chosen == alt).mean()) for alt in ALTERNATIVES}


def _mean_chosen_price(df: pd.DataFrame) -> float:
    return float(df.loc[df["chosen"].eq(1), "price"].mean())


def human_moments(calib: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    pooled = _shares(calib)
    by_task = {
        str(t): _shares(calib[calib["task_id"].eq(int(t))])
        for t in cfg["split"]["calibration_tasks"]
    }

    anchor_tasks = list(cfg["split"].get("anchor_tasks", []))
    if len(anchor_tasks) != 1:
        raise ValueError(f"public v2 expects exactly one anchor task; got {anchor_tasks}")
    anchor_task = int(anchor_tasks[0])
    intervention_tasks = [int(t) for t in cfg["split"].get("calibration_intervention_tasks", [])]

    responses: dict[str, dict[str, float]] = {}
    for task in intervention_tasks:
        responses[f"task_{task}_minus_{anchor_task}"] = {
            alt: float(by_task[str(task)][alt] - by_task[str(anchor_task)][alt])
            for alt in ALTERNATIVES
        }

    return {
        "n_respondents": int(calib["respondent_id"].nunique()),
        "n_observations": int(calib["observation_id"].nunique()),
        "anchor_task": anchor_task,
        "calibration_intervention_tasks": intervention_tasks,
        "pooled_choice_shares": pooled,
        "pooled_mean_chosen_price": _mean_chosen_price(calib),
        "task_choice_shares": by_task,
        "pairwise_share_responses": responses,
        "pairwise_substitution_responses": responses,
    }


def predicted_task_shares(pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for task, g in pred.groupby("task_id"):
        out[str(int(task))] = {
            alt: float(g.loc[g["alternative_id"].eq(alt), "mnl_prob"].mean())
            for alt in ALTERNATIVES
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/public_food_benchmark.yaml"))
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    cfg = _load_protocol(args.config)
    outdir = args.outdir or Path(cfg["output"]["dir"]) / "reference_gate"
    outdir.mkdir(parents=True, exist_ok=True)

    ds = cfg["dataset"]
    country_slug = str(ds["country_name"]).strip().lower().replace(" ", "_")
    product_slug = str(ds["product"]).strip().lower().replace(" ", "_")
    stem = f"{country_slug}_{product_slug}"

    raw = _read_raw(args.csv)
    panel = _eligible_respondents(raw, cfg)
    calibration = build_long_calibration(panel, cfg)

    expected_n = int(ds["expected_complete_panels"])
    if len(panel) != expected_n:
        raise RuntimeError(
            f"Frozen protocol expected {expected_n} complete {ds['country_name']}-{ds['product']} panels; "
            f"got {len(panel)}"
        )

    calibration.to_parquet(outdir / f"{stem}_calibration_long.parquet", index=False)
    calibration.to_csv(outdir / f"{stem}_calibration_long.csv", index=False)

    moments = human_moments(calibration, cfg)
    (outdir / "human_calibration_moments.json").write_text(
        json.dumps(moments, indent=2, sort_keys=True), encoding="utf-8"
    )

    configured_features = tuple(cfg.get("inner_model", {}).get("features", FEATURES))
    model = MultinomialLogitModel(MNLConfig(features=configured_features, l2=1.0e-4, max_iter=500))
    fit = model.fit(calibration)
    fit.save_json(outdir / "human_reference_mnl.json")
    fitted = FittedMNL(fit)
    pred = fitted.predict_long(calibration)
    pred.to_parquet(outdir / "human_reference_mnl_calibration_predictions.parquet", index=False)

    holdout_design = {
        "tasks": {
            str(t): {
                "role": cfg["tasks"][int(t)]["role"],
                "prices": cfg["tasks"][int(t)]["prices"],
                "choice_column": cfg["tasks"][int(t)]["column"],
            }
            for t in cfg["split"]["final_holdout_tasks"]
        },
        "note": "Observed final-holdout choices are intentionally not read into any summary or metric at this pre-EIPG gate.",
    }
    (outdir / "final_holdout_design_manifest.json").write_text(
        json.dumps(holdout_design, indent=2, sort_keys=True), encoding="utf-8"
    )

    beta = fit.beta_by_feature
    price_beta = float(beta["price"])
    sanity = {
        "protocol": cfg["protocol"]["name"],
        "country": ds["country_name"],
        "product": ds["product"],
        "n_eligible_respondents": int(len(panel)),
        "n_calibration_observations": int(calibration["observation_id"].nunique()),
        "n_calibration_rows": int(len(calibration)),
        "mnl_success": bool(fit.success),
        "mnl_train_nll_per_observation": float(fit.train_nll_per_observation),
        "beta_by_feature": beta,
        "price_coefficient_negative": bool(price_beta < 0),
        "empirical_task_choice_shares": moments["task_choice_shares"],
        "mnl_predicted_task_choice_shares": predicted_task_shares(pred),
        "calibration_top_choice_accuracy": float(fitted.accuracy(calibration)),
        "final_holdout_outcomes_consumed": False,
    }
    sanity["gate_pass"] = bool(fit.success and price_beta < 0 and len(panel) == expected_n)
    (outdir / "reference_gate_summary.json").write_text(
        json.dumps(sanity, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        f"# Nutri2Cycle {ds['country_name']}-{ds['product']} human reference gate",
        "",
        f"- Eligible complete respondents: **{len(panel):,}**",
        f"- Calibration observations: **{calibration['observation_id'].nunique():,}**",
        f"- MNL convergence: **{fit.success}**",
        f"- Price coefficient: **{price_beta:.4f}**",
        f"- Organic ASC vs none: **{beta['asc_organic']:.4f}**",
        f"- Conventional ASC vs none: **{beta['asc_conventional']:.4f}**",
        f"- Circular ASC vs none: **{beta['asc_circular']:.4f}**",
        f"- Calibration NLL / observation: **{fit.train_nll_per_observation:.4f}**",
        f"- Gate pass: **{sanity['gate_pass']}**",
        "- Final Task-4 observed outcomes consumed: **False**",
        "",
        "## Empirical calibration choice shares",
        "",
    ]
    for task, shares in moments["task_choice_shares"].items():
        lines.append(
            f"- Task {task}: " + ", ".join(f"{a}={shares[a]:.3f}" for a in ALTERNATIVES)
        )
    (outdir / "reference_gate_report.md").write_text("\n".join(lines), encoding="utf-8")

    print((outdir / "reference_gate_report.md").read_text(encoding="utf-8"))
    if not sanity["gate_pass"]:
        raise SystemExit("Human reference sanity gate failed; do not run public EIPG.")


if __name__ == "__main__":
    main()
