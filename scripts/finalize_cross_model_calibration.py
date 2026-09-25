#!/usr/bin/env python3
"""Freeze the entire cross-model calibration cohort before held-out evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

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
    payload = [
        {
            "persona_id": str(row["persona_id"]),
            "segment_label": str(row["segment_label"]),
            "prompt": str(row["prompt"]),
        }
        for row in rows
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_anchored_core.yaml",
    )
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_anchored_core",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    freeze_path = output_root / "calibration_cohort_freeze.json"
    if freeze_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing cohort freeze: {freeze_path}"
        )

    models = {}
    for model_key in cfg["experiment"]["model_keys"]:
        model_out = output_root / str(model_key)
        gate_path = model_out / "transfer_gate.json"
        if not gate_path.exists():
            raise SystemExit(
                f"Missing transfer gate for {model_key}: {gate_path}. "
                "Every frozen model must be gated before cohort freeze."
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        passed = bool(gate.get("passed", False))
        entry = {
            "model_key": str(model_key),
            "model": cfg["models"][model_key]["model"],
            "family": cfg["models"][model_key]["family"],
            "gate_passed": passed,
            "gate_sha256": _sha256_file(gate_path),
        }

        if passed:
            summary_path = model_out / "calibration_summary.json"
            if not summary_path.exists():
                raise SystemExit(
                    f"{model_key} passed the gate but calibration is incomplete: "
                    f"missing {summary_path}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seeds = [int(x) for x in cfg["experiment"]["calibration_seeds"]]
            by_seed = {int(row["seed"]): row for row in summary["per_seed"]}
            if set(by_seed) != set(seeds):
                raise SystemExit(
                    f"{model_key} calibration seeds do not match frozen protocol"
                )

            frozen_seeds = {}
            for seed in seeds:
                seed_dir = model_out / "calibration" / f"seed_{seed}"
                budget_path = seed_dir / "budget_trace.json"
                seed_summary_path = seed_dir / "summary.json"
                if not budget_path.exists() or not seed_summary_path.exists():
                    raise SystemExit(
                        f"Missing seed artifacts for {model_key}/{seed}"
                    )
                selected = by_seed[seed]["selected_personas"]
                frozen_seeds[str(seed)] = {
                    "selected_prompt_fingerprint": _prompt_fingerprint(selected),
                    "seed_summary_sha256": _sha256_file(seed_summary_path),
                    "budget_trace_sha256": _sha256_file(budget_path),
                    "final_objective": float(by_seed[seed]["final_objective"]),
                    "logical_simulator_calls": int(
                        by_seed[seed]["logical_simulator_calls"]
                    ),
                }

            entry.update({
                "calibration_summary_sha256": _sha256_file(summary_path),
                "calibration_seeds": frozen_seeds,
            })
        else:
            entry["calibration_status"] = "not_run_by_frozen_protocol"

        models[str(model_key)] = entry

    heldout_seeds = [int(x) for x in cfg["cross_model_holdout"]["seeds"]]
    heldout_calibration_seed = int(cfg["cross_model_holdout"]["calibration_seed"])
    if heldout_calibration_seed not in [
        int(x) for x in cfg["experiment"]["calibration_seeds"]
    ]:
        raise SystemExit(
            "cross_model_holdout.calibration_seed must be one of the frozen calibration seeds"
        )
    payload = {
        "status": "cross_model_calibration_cohort_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "heldout_constructed": False,
        "heldout_seeds_reserved_but_not_evaluated": heldout_seeds,
        "heldout_calibration_seed": heldout_calibration_seed,
        "model_entries": models,
        "passed_model_keys": [
            key for key, value in models.items() if value["gate_passed"]
        ],
        "failed_gate_model_keys": [
            key for key, value in models.items() if not value["gate_passed"]
        ],
        "calibration_locked_after_this_manifest": True,
        "warning": (
            "Do not modify model-specific calibration, prompts, search settings, "
            "budget checkpoints, or held-out seeds after this freeze."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
