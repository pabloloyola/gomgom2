#!/usr/bin/env python3
"""Freeze all adaptive-search selections before fourth heldout evaluation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import yaml


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_adaptive_search.yaml")
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_adaptive_outer_search_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_root = Path(args.output_root)
    freeze_path = out_root / "adaptive_cohort_freeze.json"
    if freeze_path.exists():
        raise SystemExit(f"Refusing to overwrite existing freeze: {freeze_path}")

    random_root = Path(cfg["comparison"]["source_random_subset_root"])
    screen_root = Path(cfg["comparison"]["source_screen_confirm_root"])

    models = {}
    for model_key in cfg["experiment"]["model_keys"]:
        model_key = str(model_key)
        residual_path = out_root / model_key / "residual_linucb_selection.json"
        llm_path = out_root / model_key / "history_llm_selection.json"
        prior_path = random_root / model_key / "baseline_selection.json"
        screen_path = screen_root / model_key / "strategy_selection.json"

        for p in (residual_path, llm_path, prior_path, screen_path):
            if not p.exists():
                raise FileNotFoundError(p)

        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        comparison_records = prior["comparison_records"]
        fixed = next(r for r in comparison_records if r["strategy"] == "fixed_prefix")
        random_records = [r for r in comparison_records if r["strategy"] == "random_subset"]

        screen = json.loads(screen_path.read_text(encoding="utf-8"))
        screen_1320 = next(
            r for r in screen["screen_confirm_selections"]
            if r["budget_label"] == "screen_confirm_1320"
        )

        residual = json.loads(residual_path.read_text(encoding="utf-8"))
        llm = json.loads(llm_path.read_text(encoding="utf-8"))

        models[model_key] = {
            "artifacts": {
                "residual_linucb": {
                    "path": str(residual_path),
                    "sha256": _sha256_file(residual_path),
                },
                "history_llm": {
                    "path": str(llm_path),
                    "sha256": _sha256_file(llm_path),
                },
                "prior_equal_budget": {
                    "path": str(prior_path),
                    "sha256": _sha256_file(prior_path),
                },
                "screen_confirm": {
                    "path": str(screen_path),
                    "sha256": _sha256_file(screen_path),
                },
            },
            "selected_evaluation_indices": {
                "fixed_prefix": int(fixed["selected_evaluation_index"]),
                "screen_confirm": int(screen_1320["selected_evaluation_index"]),
                "residual_linucb": int(residual["selected_evaluation_index"]),
                "history_llm": int(llm["selected_evaluation_index"]),
                "random_subset": {
                    str(r["subset_seed"]): int(r["selected_evaluation_index"])
                    for r in random_records
                },
            },
            "history_llm_proposer_calls": int(llm["proposer_calls"]),
            "history_llm_fallback_count": int(llm["fallback_count"]),
        }

    out_root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "status": "adaptive_outer_search_cohort_frozen",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "calibration_seed": int(cfg["experiment"]["calibration_seed"]),
        "logical_simulator_choice_query_budget": int(
            cfg["budget"]["logical_simulator_choice_queries"]
        ),
        "fresh_heldout_seeds_reserved_but_not_evaluated": [
            int(x) for x in cfg["fresh_heldout"]["seeds"]
        ],
        "models": models,
        "heldout_evaluated": False,
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
