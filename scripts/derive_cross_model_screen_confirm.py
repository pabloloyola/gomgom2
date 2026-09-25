#!/usr/bin/env python3
"""Derive a frozen screen-then-confirm strategy from existing calibration choices.

No new model calls are made. The script reconstructs partial-context calibration
objectives from the already-generated exhaustive one-sweep artifacts, applies
the predeclared screen/confirm policy, and writes a cohort freeze manifest.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_llm_prompt_refinement_calibration_v2 import (
    calibration_slates,
    residual_packet,
    simulated_moments,
    target_moments,
    target_probability_table,
    weighted_rmse,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hash_jsonable(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def _variant_mask(series: pd.Series, variants: list[int]) -> pd.Series:
    text = series.astype(str)
    mask = pd.Series(False, index=series.index)
    for variant in variants:
        mask = mask | text.str.contains(f"_v{variant:02d}_", regex=False)
    return mask


def _partial_target(
    *,
    seed: int,
    variants_per_intervention: int,
    selected_variants: list[int],
) -> dict[str, float]:
    slates = calibration_slates(seed, variants_per_intervention)
    keep = {
        cid: frame
        for cid, frame in slates.items()
        if any(f"_v{v:02d}_" in cid for v in selected_variants)
    }
    return target_moments(target_probability_table(keep))


def _objective_for_choices(
    choices: pd.DataFrame,
    *,
    target: dict[str, float],
    variants: list[int],
) -> float:
    subset = choices.loc[_variant_mask(choices["context_id"], variants)].copy()
    simulated = simulated_moments(subset)
    return float(weighted_rmse(residual_packet(target, simulated)))


def _load_eval_artifacts(seed_dir: Path) -> list[dict]:
    history_path = seed_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = []
    for row in history:
        if "evaluation_index" not in row:
            continue
        idx = int(row["evaluation_index"])
        choices_path = seed_dir / "evaluations" / f"eval_{idx:02d}" / "choices.csv"
        if not choices_path.exists():
            raise FileNotFoundError(choices_path)
        rows.append({
            "evaluation_index": idx,
            "full_objective": float(row["objective"]),
            "personas": row["personas"],
            "status": str(row.get("status", "")),
            "persona_id": row.get("persona_id"),
            "feature": row.get("feature"),
            "signed_step": row.get("signed_step"),
            "choices_path": str(choices_path),
            "choices": pd.read_csv(choices_path),
        })
    rows.sort(key=lambda x: x["evaluation_index"])
    expected = list(range(31))
    observed = [row["evaluation_index"] for row in rows]
    if observed != expected:
        raise RuntimeError(
            f"Expected exactly evaluations 0..30 in {seed_dir}; observed {observed}"
        )
    return rows


def _selection_record(
    *,
    label: str,
    budget_calls: int,
    candidates: list[dict],
) -> dict:
    best = min(
        candidates,
        key=lambda row: (float(row["full_objective"]), int(row["evaluation_index"])),
    )
    return {
        "strategy": "screen_confirm",
        "budget_label": label,
        "logical_simulator_call_budget": int(budget_calls),
        "selected_evaluation_index": int(best["evaluation_index"]),
        "selected_full_calibration_objective": float(best["full_objective"]),
        "selected_personas": best["personas"],
        "selected_move": {
            "persona_id": best.get("persona_id"),
            "feature": best.get("feature"),
            "signed_step": best.get("signed_step"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_screen_confirm_budget.yaml",
    )
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_screen_confirm_budget_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_root = Path(cfg["experiment"]["source_output_root"])
    out_root = Path(args.output_root)
    freeze_path = out_root / "strategy_cohort_freeze.json"
    if freeze_path.exists():
        raise SystemExit(f"Refusing to overwrite existing strategy freeze: {freeze_path}")

    calibration_seed = int(cfg["experiment"]["calibration_seed"])
    source_cfg_path = Path("configs/cross_model_anchored_core.yaml")
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    variants_per_intervention = int(
        source_cfg["experiment"]["context_variants_per_intervention"]
    )

    stage_a_variants = [int(x) for x in cfg["screen_confirm"]["stage_a_variants"]]
    stage_c_variants = [int(x) for x in cfg["screen_confirm"]["stage_c_variants"]]
    stage_b_top = int(cfg["screen_confirm"]["stage_b_top_neighbors"])
    stage_c_additional = int(
        cfg["screen_confirm"]["stage_c_additional_top_neighbors"]
    )

    target_stage_a = _partial_target(
        seed=calibration_seed,
        variants_per_intervention=variants_per_intervention,
        selected_variants=stage_a_variants,
    )
    target_stage_c = _partial_target(
        seed=calibration_seed,
        variants_per_intervention=variants_per_intervention,
        selected_variants=stage_c_variants,
    )

    model_entries = {}
    for model_key in cfg["experiment"]["model_keys"]:
        seed_dir = (
            source_root / str(model_key) / "calibration" / f"seed_{calibration_seed}"
        )
        rows = _load_eval_artifacts(seed_dir)
        baseline = rows[0]
        neighbors = rows[1:]

        for row in rows:
            row["stage_a_objective"] = _objective_for_choices(
                row["choices"],
                target=target_stage_a,
                variants=stage_a_variants,
            )
            row["stage_c_objective"] = _objective_for_choices(
                row["choices"],
                target=target_stage_c,
                variants=stage_c_variants,
            )

        first_ranked = sorted(
            neighbors,
            key=lambda row: (
                float(row["stage_a_objective"]),
                int(row["evaluation_index"]),
            ),
        )
        first_confirmed = first_ranked[:stage_b_top]
        stage_b_confirmed = [baseline, *first_confirmed]
        stage_b_selection = _selection_record(
            label="screen_confirm_1320",
            budget_calls=int(cfg["screen_confirm"]["stage_b_budget_calls"]),
            candidates=stage_b_confirmed,
        )

        first_ids = {int(row["evaluation_index"]) for row in first_confirmed}
        remaining = [
            row for row in neighbors
            if int(row["evaluation_index"]) not in first_ids
        ]
        second_ranked = sorted(
            remaining,
            key=lambda row: (
                float(row["stage_c_objective"]),
                int(row["evaluation_index"]),
            ),
        )
        second_confirmed = second_ranked[:stage_c_additional]
        stage_c_confirmed = [baseline, *first_confirmed, *second_confirmed]
        stage_c_selection = _selection_record(
            label="screen_confirm_2496",
            budget_calls=int(cfg["screen_confirm"]["stage_c_budget_calls"]),
            candidates=stage_c_confirmed,
        )

        full_selection = _selection_record(
            label="screen_confirm_3720",
            budget_calls=int(cfg["screen_confirm"]["full_budget_calls"]),
            candidates=rows,
        )

        source_budget_trace = json.loads(
            (seed_dir / "budget_trace.json").read_text(encoding="utf-8")
        )
        fixed_by_label = {
            str(item["label"]): item
            for item in source_budget_trace["checkpoints"]
        }
        fixed_records = []
        for label in cfg["comparison"]["fixed_prefix_budget_labels"]:
            item = fixed_by_label[str(label)]
            fixed_records.append({
                "strategy": "fixed_prefix",
                "budget_label": str(label),
                "logical_simulator_call_budget": int(
                    cfg["comparison"]["fixed_prefix_budget_calls"][label]
                ),
                "selected_evaluation_index": (
                    None
                    if item.get("selected_evaluation_index") is None
                    else int(item["selected_evaluation_index"])
                ),
                "selected_full_calibration_objective": float(
                    item["selected_objective"]
                ),
                "selected_personas": item["selected_personas"],
            })

        first_rank_by_eval = {
            int(row["evaluation_index"]): rank
            for rank, row in enumerate(first_ranked, start=1)
        }
        second_ids = {int(x["evaluation_index"]) for x in second_confirmed}
        screen_rows = [
            {
                "evaluation_index": int(row["evaluation_index"]),
                "full_objective": float(row["full_objective"]),
                "stage_a_objective": float(row["stage_a_objective"]),
                "stage_c_objective": float(row["stage_c_objective"]),
                "first_stage_rank": first_rank_by_eval.get(
                    int(row["evaluation_index"])
                ),
                "selected_stage_b_confirmation": (
                    int(row["evaluation_index"]) in first_ids
                ),
                "selected_stage_c_additional_confirmation": (
                    int(row["evaluation_index"]) in second_ids
                ),
                "persona_id": row.get("persona_id"),
                "feature": row.get("feature"),
                "signed_step": row.get("signed_step"),
            }
            for row in rows
        ]

        model_out = out_root / str(model_key)
        model_out.mkdir(parents=True, exist_ok=True)
        strategy_payload = {
            "model_key": str(model_key),
            "calibration_seed": calibration_seed,
            "source_seed_dir": str(seed_dir),
            "source_history_sha256": _sha256_file(seed_dir / "history.json"),
            "source_budget_trace_sha256": _sha256_file(
                seed_dir / "budget_trace.json"
            ),
            "screen_confirm_selections": [
                stage_b_selection,
                stage_c_selection,
                full_selection,
            ],
            "fixed_prefix_selections": fixed_records,
            "screening_diagnostics": screen_rows,
            "logical_budget_derivation": {
                "stage_a_all_31_on_one_variant": 31 * 24,
                "stage_b_confirm_baseline_plus_top5": 6 * 96,
                "stage_b_total": 1320,
                "stage_c_add_second_variant_to_remaining25": 25 * 24,
                "stage_c_confirm_additional8": 8 * 72,
                "stage_c_total": 2496,
                "full_total": 3720,
            },
        }
        strategy_path = model_out / "strategy_selection.json"
        strategy_path.write_text(
            json.dumps(strategy_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_entries[str(model_key)] = {
            "strategy_selection_sha256": _sha256_file(strategy_path),
            "stage_b_selected_evaluation_index": int(
                stage_b_selection["selected_evaluation_index"]
            ),
            "stage_c_selected_evaluation_index": int(
                stage_c_selection["selected_evaluation_index"]
            ),
            "full_selected_evaluation_index": int(
                full_selection["selected_evaluation_index"]
            ),
        }

    out_root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "status": "screen_confirm_strategy_cohort_frozen",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "source_protocol_config_sha256": _sha256_file(source_cfg_path),
        "source_output_root": str(source_root),
        "calibration_seed": calibration_seed,
        "fresh_heldout_seeds_reserved_but_not_evaluated": [
            int(x) for x in cfg["fresh_heldout"]["seeds"]
        ],
        "model_entries": model_entries,
        "new_calibration_model_calls": 0,
        "selection_uses_previous_cross_model_heldout": False,
        "warning": (
            "The strategy design was motivated after observing the first heldout "
            "experiment, so only the fresh seeds in this config may evaluate it."
        ),
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
