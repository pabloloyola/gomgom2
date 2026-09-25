#!/usr/bin/env python3
"""Gate and calibrate one model in the frozen cross-model anchored protocol.

This runner never constructs cross-model held-out contexts. After all models
finish gating/calibration, run finalize_cross_model_calibration.py to freeze the
cohort before any held-out evaluation is allowed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eipg.simulators.huggingface_local import (
    HuggingFaceLocalChatClient,
    HuggingFaceLocalConfig,
)
from scripts.run_local_hf_gemma12_anchored_mu_seed_robustness import run_seed
from scripts.run_local_hf_gemma12_eipg import _run_transfer_gate


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _hash_jsonable(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def _backend(cfg: dict, model_key: str) -> dict:
    section = dict(cfg["backend_common"])
    section.update(dict(cfg["models"][model_key]))
    for key in ("family", "nominal_parameters_b"):
        section.pop(key, None)
    return section


def _budget_trace(seed_out: Path, checkpoints: list[int]) -> dict:
    history = json.loads((seed_out / "history.json").read_text(encoding="utf-8"))
    summary = json.loads((seed_out / "summary.json").read_text(encoding="utf-8"))
    ordered = sorted(
        history,
        key=lambda row: int(row.get("evaluation_index", 10**9)),
    )
    if not ordered or "evaluation_index" not in ordered[0]:
        raise RuntimeError(
            f"{seed_out}/history.json does not contain budget accounting fields"
        )

    records = []
    for checkpoint in checkpoints:
        eligible = [
            row for row in ordered
            if int(row["evaluation_index"]) + 1 <= int(checkpoint)
        ]
        if not eligible:
            continue
        max_available = int(ordered[-1]["evaluation_index"]) + 1
        if checkpoint > max_available:
            continue
        best = min(
            eligible,
            key=lambda row: (float(row["objective"]), int(row["evaluation_index"])),
        )
        records.append({
            "label": f"eval_budget_{checkpoint}",
            "evaluation_budget": int(checkpoint),
            "logical_simulator_call_budget": (
                int(checkpoint) * int(summary["logical_calls_per_evaluation"])
            ),
            "selected_evaluation_index": int(best["evaluation_index"]),
            "selected_objective": float(best["objective"]),
            "selected_personas": best["personas"],
        })

    records.append({
        "label": "full_search",
        "evaluation_budget": int(summary["evaluation_calls"]),
        "logical_simulator_call_budget": int(summary["logical_simulator_calls"]),
        "selected_evaluation_index": None,
        "selected_objective": float(summary["final_objective"]),
        "selected_personas": summary["selected_personas"],
    })

    payload = {
        "seed": int(summary["seed"]),
        "logical_calls_per_evaluation": int(summary["logical_calls_per_evaluation"]),
        "full_search_evaluations": int(summary["evaluation_calls"]),
        "full_search_logical_simulator_calls": int(summary["logical_simulator_calls"]),
        "checkpoints": records,
        "definition": (
            "At each prefix budget, select the lowest calibration objective among "
            "all complete population evaluations observed by that point in the "
            "fixed coordinate-enumeration order. Cache hits do not reduce budget."
        ),
    }
    (seed_out / "budget_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_anchored_core.yaml",
    )
    ap.add_argument("--model-key", required=True)
    ap.add_argument(
        "--stage",
        choices=("gate", "calibration", "both"),
        default="gate",
    )
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_anchored_core",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already completed gate/seeds instead of refusing to overwrite.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_key = str(args.model_key)
    if model_key not in cfg["experiment"]["model_keys"]:
        raise SystemExit(
            f"Unknown model key {model_key!r}; frozen keys are "
            f"{cfg['experiment']['model_keys']}"
        )

    output_root = Path(args.output_root)
    model_out = output_root / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    cohort_freeze = output_root / "calibration_cohort_freeze.json"
    if cohort_freeze.exists() and args.stage in {"calibration", "both"}:
        raise SystemExit(
            "Cross-model cohort calibration is already frozen; refusing further "
            f"calibration because {cohort_freeze} exists."
        )

    model_meta = dict(cfg["models"][model_key])
    backend = _backend(cfg, model_key)

    print(
        f"Loading {model_key}: {model_meta['model']} "
        f"(loader={backend.get('model_loader', 'auto')})",
        flush=True,
    )
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))

    provenance = {
        "git_sha": _git_sha(),
        "config": str(cfg_path),
        "config_sha256": sha256(cfg_path.read_bytes()).hexdigest(),
        "protocol": cfg["experiment"]["name"],
        "model_key": model_key,
        "family": model_meta["family"],
        "nominal_parameters_b": model_meta["nominal_parameters_b"],
        "heldout_constructed": False,
        "runtime": client.runtime_info(),
    }
    (model_out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    gate_path = model_out / "transfer_gate.json"
    gate = None
    if args.stage in {"gate", "both"}:
        if gate_path.exists() and args.resume:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            print("Reusing existing transfer gate result.", flush=True)
        elif gate_path.exists():
            raise SystemExit(
                f"{gate_path} already exists. Use --resume to reuse it."
            )
        else:
            print("Running frozen transfer gate...", flush=True)
            gate = _run_transfer_gate(client=client, cfg=cfg, output=gate_path)
            print(json.dumps(gate, indent=2, sort_keys=True), flush=True)

        if args.stage == "gate":
            if not bool(gate.get("passed", False)):
                raise SystemExit("Transfer gate failed; this is a reportable model result.")
            return

    if gate is None:
        if not gate_path.exists():
            raise SystemExit(
                "Calibration requires a completed transfer gate. Run --stage gate first."
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))

    if not bool(gate.get("passed", False)):
        raise SystemExit(
            "Transfer gate did not pass. Frozen protocol forbids repairing or "
            "model-specifically tuning the persona prompts."
        )

    calibration_out = model_out / "calibration"
    calibration_out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in cfg["experiment"]["calibration_seeds"]]
    checkpoints = [int(x) for x in cfg["budget"]["evaluation_checkpoints"]]

    results = []
    budget_traces = []
    for seed in seeds:
        seed_out = calibration_out / f"seed_{seed}"
        summary_path = seed_out / "summary.json"
        if summary_path.exists() and args.resume:
            print(f"Reusing completed calibration seed {seed}.", flush=True)
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        elif summary_path.exists():
            raise SystemExit(
                f"{summary_path} already exists. Use --resume to reuse completed seeds."
            )
        else:
            print(f"Running cross-model anchored calibration seed {seed}...", flush=True)
            result = run_seed(
                seed=seed,
                cfg=cfg,
                client=client,
                seed_out=seed_out,
            )
        results.append(result)
        budget_traces.append(_budget_trace(seed_out, checkpoints))

    improvements = [float(r["improvement_fraction"]) for r in results]
    aggregate = {
        "status": "cross_model_calibration_complete",
        "protocol": cfg["experiment"]["name"],
        "git_sha": _git_sha(),
        "model_key": model_key,
        "model": model_meta["model"],
        "family": model_meta["family"],
        "nominal_parameters_b": model_meta["nominal_parameters_b"],
        "runtime": client.runtime_info(),
        "heldout_constructed": False,
        "calibration_seeds": seeds,
        "n_seeds": len(results),
        "per_seed": results,
        "mean_improvement_fraction": statistics.fmean(improvements),
        "median_improvement_fraction": statistics.median(improvements),
        "min_improvement_fraction": min(improvements),
        "max_improvement_fraction": max(improvements),
        "seeds_with_positive_improvement": sum(x > 0 for x in improvements),
        "total_logical_simulator_calls": sum(
            int(r["logical_simulator_calls"]) for r in results
        ),
        "budget_trace_hashes": {
            str(trace["seed"]): _hash_jsonable(trace) for trace in budget_traces
        },
        "transfer_gate": gate,
    }
    (model_out / "calibration_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
