#!/usr/bin/env python3
"""Freeze equal-budget random-subset baselines from existing calibration artifacts.

No new calibration calls are made. For each model and predeclared subset seed,
sample 10 of the 30 fully evaluated coordinate neighbors, add the baseline, and
select the lowest full calibration objective among those 11 populations.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _prompt_fingerprint(rows: list[dict]) -> str:
    raw = json.dumps(
        [
            {
                "persona_id": str(row["persona_id"]),
                "segment_label": str(row["segment_label"]),
                "prompt": str(row["prompt"]),
            }
            for row in rows
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _history_by_eval(seed_dir: Path) -> dict[int, dict]:
    history = json.loads((seed_dir / "history.json").read_text(encoding="utf-8"))
    rows = {
        int(row["evaluation_index"]): row
        for row in history
        if "evaluation_index" in row
    }
    if set(rows) != set(range(31)):
        raise RuntimeError(
            f"Expected evaluation indices 0..30 in {seed_dir}; got {sorted(rows)}"
        )
    return rows


def _selection(row: dict, *, strategy: str, label: str, budget: int, extra: dict | None = None) -> dict:
    payload = {
        "strategy": strategy,
        "budget_label": label,
        "logical_simulator_call_budget": int(budget),
        "selected_evaluation_index": int(row["evaluation_index"]),
        "selected_full_calibration_objective": float(row["objective"]),
        "selected_personas": row["personas"],
        "selected_prompt_fingerprint": _prompt_fingerprint(row["personas"]),
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_random_subset_baseline.yaml",
    )
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_random_subset_baseline_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_root = Path(cfg["experiment"]["source_output_root"])
    screen_root = Path(cfg["experiment"]["screen_confirm_output_root"])
    out_root = Path(args.output_root)
    freeze_path = out_root / "baseline_cohort_freeze.json"
    if freeze_path.exists():
        raise SystemExit(f"Refusing to overwrite existing freeze: {freeze_path}")

    calibration_seed = int(cfg["experiment"]["calibration_seed"])
    subset_seeds = [int(x) for x in cfg["random_subset"]["subset_seeds"]]
    candidate_indices = [int(x) for x in cfg["random_subset"]["candidate_evaluation_indices"]]
    n_neighbors = int(cfg["budget"]["random_neighbors_per_repeat"])
    budget = int(cfg["budget"]["logical_simulator_call_budget"])

    if len(candidate_indices) != 30 or set(candidate_indices) != set(range(1, 31)):
        raise SystemExit("Random-subset candidate indices must be exactly 1..30")
    expected_budget = (
        int(cfg["budget"]["baseline_evaluations"]) + n_neighbors
    ) * int(cfg["budget"]["full_calls_per_population"])
    if expected_budget != budget:
        raise SystemExit(
            f"Budget mismatch: config says {budget}, derivation gives {expected_budget}"
        )

    model_entries: dict[str, Any] = {}

    for model_key in cfg["experiment"]["model_keys"]:
        seed_dir = source_root / str(model_key) / "calibration" / f"seed_{calibration_seed}"
        by_eval = _history_by_eval(seed_dir)

        source_budget = json.loads(
            (seed_dir / "budget_trace.json").read_text(encoding="utf-8")
        )
        fixed_1320 = next(
            item for item in source_budget["checkpoints"]
            if str(item["label"]) == "eval_budget_11"
        )
        fixed_row = by_eval[int(fixed_1320["selected_evaluation_index"])]

        screen_path = screen_root / str(model_key) / "strategy_selection.json"
        if not screen_path.exists():
            raise FileNotFoundError(
                f"Missing frozen screen-confirm selection for {model_key}: {screen_path}"
            )
        screen_payload = json.loads(screen_path.read_text(encoding="utf-8"))
        screen_1320 = next(
            item for item in screen_payload["screen_confirm_selections"]
            if str(item["budget_label"]) == "screen_confirm_1320"
        )
        screen_row = by_eval[int(screen_1320["selected_evaluation_index"])]

        random_records = []
        for subset_seed in subset_seeds:
            rng = random.Random(subset_seed)
            sampled = sorted(rng.sample(candidate_indices, n_neighbors))
            eligible = [0, *sampled]
            best = min(
                (by_eval[idx] for idx in eligible),
                key=lambda row: (
                    float(row["objective"]),
                    int(row["evaluation_index"]),
                ),
            )
            random_records.append(
                _selection(
                    best,
                    strategy="random_subset",
                    label=f"random_subset_seed_{subset_seed}",
                    budget=budget,
                    extra={
                        "subset_seed": subset_seed,
                        "sampled_neighbor_evaluation_indices": sampled,
                        "eligible_evaluation_indices": eligible,
                    },
                )
            )

        comparison_records = [
            _selection(
                fixed_row,
                strategy="fixed_prefix",
                label="fixed_prefix_1320",
                budget=budget,
            ),
            _selection(
                screen_row,
                strategy="screen_confirm",
                label="screen_confirm_1320",
                budget=budget,
            ),
            *random_records,
        ]

        model_out = out_root / str(model_key)
        model_out.mkdir(parents=True, exist_ok=True)
        selection = {
            "model_key": str(model_key),
            "calibration_seed": calibration_seed,
            "source_history_sha256": _sha256_file(seed_dir / "history.json"),
            "source_budget_trace_sha256": _sha256_file(seed_dir / "budget_trace.json"),
            "screen_confirm_selection_sha256": _sha256_file(screen_path),
            "comparison_records": comparison_records,
            "new_calibration_calls": 0,
        }
        selection_path = model_out / "baseline_selection.json"
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_entries[str(model_key)] = {
            "baseline_selection_sha256": _sha256_file(selection_path),
            "random_subset_selected_evaluation_indices": {
                str(row["subset_seed"]): int(row["selected_evaluation_index"])
                for row in random_records
            },
            "screen_confirm_selected_evaluation_index": int(
                screen_row["evaluation_index"]
            ),
            "fixed_prefix_selected_evaluation_index": int(
                fixed_row["evaluation_index"]
            ),
        }

    out_root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "status": "random_subset_baseline_cohort_frozen",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "source_output_root": str(source_root),
        "screen_confirm_output_root": str(screen_root),
        "calibration_seed": calibration_seed,
        "fresh_heldout_seeds_reserved_but_not_evaluated": [
            int(x) for x in cfg["fresh_heldout"]["seeds"]
        ],
        "subset_seeds": subset_seeds,
        "model_entries": model_entries,
        "new_calibration_model_calls": 0,
        "warning": (
            "This baseline was designed after observing the second heldout experiment. "
            "Only the fresh heldout seeds in this config may evaluate it."
        ),
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
