#!/usr/bin/env python3
"""Frozen cross-model held-out evaluation, including predeclared budget checkpoints."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eipg.simulators.huggingface_local import (
    HuggingFaceLocalChatClient,
    HuggingFaceLocalConfig,
)
from eipg.simulators.llm_choice import TextPersona, TextPersonaChoiceSimulator
from scripts.run_llm_prompt_refinement_calibration_v2 import (
    calibration_slates,
    initial_personas,
    target_moments,
    target_probability_table,
)
from scripts.run_local_hf_gemma12_anchored_mu_frozen_heldout import HeldoutEvaluator


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _backend(cfg: dict, model_key: str) -> dict:
    section = dict(cfg["backend_common"])
    section.update(dict(cfg["models"][model_key]))
    for key in ("family", "nominal_parameters_b"):
        section.pop(key, None)
    return section


def _personas(rows: list[dict]) -> tuple[TextPersona, ...]:
    result = tuple(
        TextPersona(
            persona_id=str(row["persona_id"]),
            segment_label=str(row["segment_label"]),
            prompt=str(row["prompt"]),
        )
        for row in rows
    )
    ids = [p.persona_id for p in result]
    if ids != ["budget", "quality", "sustain"]:
        raise ValueError(f"Unexpected persona order: {ids}")
    return result


def _fingerprint(personas: tuple[TextPersona, ...]) -> str:
    raw = json.dumps(
        [
            {
                "persona_id": p.persona_id,
                "segment_label": p.segment_label,
                "prompt": p.prompt,
            }
            for p in personas
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _verify_frozen_model(
    *,
    model_key: str,
    model_out: Path,
    freeze_entry: dict,
    calibration_seeds: list[int],
) -> None:
    summary_path = model_out / "calibration_summary.json"
    if _sha256_file(summary_path) != freeze_entry["calibration_summary_sha256"]:
        raise SystemExit(
            f"{model_key} calibration_summary.json changed after cohort freeze"
        )
    for seed in calibration_seeds:
        frozen = freeze_entry["calibration_seeds"][str(seed)]
        seed_dir = model_out / "calibration" / f"seed_{seed}"
        if _sha256_file(seed_dir / "summary.json") != frozen["seed_summary_sha256"]:
            raise SystemExit(
                f"{model_key} seed {seed} summary changed after cohort freeze"
            )
        if _sha256_file(seed_dir / "budget_trace.json") != frozen["budget_trace_sha256"]:
            raise SystemExit(
                f"{model_key} seed {seed} budget trace changed after cohort freeze"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_anchored_core.yaml",
    )
    ap.add_argument("--model-key", required=True)
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_anchored_core",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    freeze_path = output_root / "calibration_cohort_freeze.json"
    if not freeze_path.exists():
        raise SystemExit(
            "Cross-model heldout is locked until calibration_cohort_freeze.json exists."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256_file(cfg_path) != freeze["config_sha256"]:
        raise SystemExit("Cross-model config changed after cohort freeze.")

    model_key = str(args.model_key)
    if model_key not in freeze["model_entries"]:
        raise SystemExit(f"Model {model_key!r} is not part of the frozen cohort.")
    freeze_entry = freeze["model_entries"][model_key]
    if not bool(freeze_entry["gate_passed"]):
        raise SystemExit(
            f"{model_key} failed the frozen transfer gate; heldout is not applicable."
        )

    calibration_seeds = [int(x) for x in cfg["experiment"]["calibration_seeds"]]
    model_out = output_root / model_key
    _verify_frozen_model(
        model_key=model_key,
        model_out=model_out,
        freeze_entry=freeze_entry,
        calibration_seeds=calibration_seeds,
    )

    heldout_out = model_out / "cross_model_heldout_final"
    if heldout_out.exists() and any(heldout_out.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite frozen heldout directory: {heldout_out}"
        )
    heldout_out.mkdir(parents=True, exist_ok=True)

    backend = _backend(cfg, model_key)
    print(f"Loading frozen model {model_key}: {backend['model']}", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))
    simulator = TextPersonaChoiceSimulator(client)

    calibration_seed = int(cfg["cross_model_holdout"]["calibration_seed"])
    if calibration_seed not in calibration_seeds:
        raise SystemExit(
            "Cross-model heldout calibration_seed is not in the frozen calibration set."
        )
    heldout_seeds = [int(x) for x in cfg["cross_model_holdout"]["seeds"]]
    if not heldout_seeds:
        raise SystemExit("Cross-model heldout seed list is empty.")

    requested_eval_checkpoints = {
        int(x) for x in cfg["budget"]["heldout_evaluation_checkpoints"]
    }

    manifest = {
        "status": "cross_model_heldout_started",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "model_key": model_key,
        "model": cfg["models"][model_key]["model"],
        "runtime": client.runtime_info(),
        "cohort_freeze_sha256": _sha256_file(freeze_path),
        "optimization_performed": False,
        "selection_uses_heldout": False,
        "budget_checkpoints_predeclared": sorted(requested_eval_checkpoints),
        "calibration_seed": calibration_seed,
        "heldout_seeds": heldout_seeds,
    }
    (heldout_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    original_population = initial_personas()
    rows = []

    seed_dir = model_out / "calibration" / f"seed_{calibration_seed}"
    trace = json.loads(
        (seed_dir / "budget_trace.json").read_text(encoding="utf-8")
    )
    checkpoints = []
    for item in trace["checkpoints"]:
        label = str(item["label"])
        eval_budget = int(item["evaluation_budget"])
        if label == "full_search" or eval_budget in requested_eval_checkpoints:
            checkpoints.append(item)

    for heldout_seed in heldout_seeds:
        run_dir = heldout_out / (
            f"calibration_seed_{calibration_seed}__heldout_seed_{heldout_seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        slates = calibration_slates(
            heldout_seed,
            int(cfg["experiment"]["context_variants_per_intervention"]),
        )
        targets = target_probability_table(slates)
        moments = target_moments(targets)
        targets.to_csv(run_dir / "heldout_target_probabilities.csv", index=False)
        (run_dir / "heldout_target_moments.json").write_text(
            json.dumps(moments, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        original_result = HeldoutEvaluator(
            simulator=simulator,
            slates=slates,
            target=moments,
            output_dir=run_dir / "original",
            condition="original",
        ).evaluate(original_population)
        original_obj = float(original_result["objective"])
        original_fp = _fingerprint(original_population)

        evaluated_by_fp = {original_fp: original_result}
        for item in checkpoints:
            eval_budget = int(item["evaluation_budget"])
            call_budget = int(item["logical_simulator_call_budget"])
            label = str(item["label"])
            calibration_obj = float(item["selected_objective"])
            population = _personas(item["selected_personas"])
            fp = _fingerprint(population)

            if fp in evaluated_by_fp:
                heldout_result = evaluated_by_fp[fp]
            else:
                safe_label = label.replace("/", "_")
                heldout_result = HeldoutEvaluator(
                    simulator=simulator,
                    slates=slates,
                    target=moments,
                    output_dir=run_dir / safe_label,
                    condition=safe_label,
                ).evaluate(population)
                evaluated_by_fp[fp] = heldout_result

            heldout_obj = float(heldout_result["objective"])
            improvement = (original_obj - heldout_obj) / original_obj
            rows.append({
                "model_key": model_key,
                "model": cfg["models"][model_key]["model"],
                "family": cfg["models"][model_key]["family"],
                "calibration_seed": calibration_seed,
                "heldout_seed": heldout_seed,
                "budget_label": label,
                "evaluation_budget": eval_budget,
                "logical_simulator_call_budget": call_budget,
                "calibration_objective": calibration_obj,
                "original_heldout_objective": original_obj,
                "calibrated_heldout_objective": heldout_obj,
                "heldout_improvement_fraction": improvement,
                "heldout_improved": bool(heldout_obj < original_obj),
                "selected_prompt_fingerprint": fp,
            })
            print(
                f"{model_key} calibration {calibration_seed} -> heldout {heldout_seed} "
                f"{label}: {original_obj:.6f} -> {heldout_obj:.6f} "
                f"({improvement:+.2%}); calibration budget={call_budget}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(heldout_out / "budget_heldout_results.csv", index=False)

    summaries = []
    for label, group in frame.groupby("budget_label", sort=False):
        improvements = group["heldout_improvement_fraction"].astype(float).tolist()
        summaries.append({
            "model_key": model_key,
            "budget_label": label,
            "evaluation_budget_mean": float(group["evaluation_budget"].mean()),
            "logical_simulator_call_budget_mean": float(
                group["logical_simulator_call_budget"].mean()
            ),
            "n_pairs": int(len(group)),
            "pairs_improved": int(group["heldout_improved"].astype(bool).sum()),
            "fraction_pairs_improved": float(
                group["heldout_improved"].astype(bool).mean()
            ),
            "mean_heldout_improvement_fraction": statistics.fmean(improvements),
            "median_heldout_improvement_fraction": statistics.median(improvements),
            "mean_calibrated_heldout_objective": float(
                group["calibrated_heldout_objective"].mean()
            ),
        })

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(heldout_out / "budget_heldout_summary.csv", index=False)
    full_rows = frame.loc[frame["budget_label"] == "full_search"]
    full_improvements = full_rows["heldout_improvement_fraction"].astype(float).tolist()
    aggregate = {
        "status": "cross_model_heldout_complete",
        "protocol": cfg["experiment"]["name"],
        "git_sha": _git_sha(),
        "model_key": model_key,
        "model": cfg["models"][model_key]["model"],
        "family": cfg["models"][model_key]["family"],
        "n_full_search_pairs": int(len(full_rows)),
        "full_search_pairs_improved": int(
            full_rows["heldout_improved"].astype(bool).sum()
        ),
        "full_search_fraction_pairs_improved": float(
            full_rows["heldout_improved"].astype(bool).mean()
        ),
        "full_search_mean_improvement_fraction": statistics.fmean(
            full_improvements
        ),
        "full_search_median_improvement_fraction": statistics.median(
            full_improvements
        ),
        "budget_summary": summaries,
        "optimization_performed": False,
        "selection_uses_heldout": False,
        "final_metrics_reporting_only": True,
        "warning": (
            "Do not modify model-specific calibration or budget strategy using "
            "these cross-model heldout results."
        ),
    }
    (heldout_out / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
