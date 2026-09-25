#!/usr/bin/env python3
"""Evaluate frozen screen-confirm and fixed-prefix selections on fresh heldout seeds."""
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
from scripts.run_cross_model_anchored_heldout import _backend
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/cross_model_screen_confirm_budget.yaml",
    )
    ap.add_argument("--model-key", required=True)
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_screen_confirm_budget_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_cfg_path = Path("configs/cross_model_anchored_core.yaml")
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    out_root = Path(args.output_root)

    freeze_path = out_root / "strategy_cohort_freeze.json"
    if not freeze_path.exists():
        raise SystemExit(
            "Strategy selections are not frozen. Run derive_cross_model_screen_confirm.py first."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256_file(cfg_path) != freeze["config_sha256"]:
        raise SystemExit("Budget-strategy config changed after strategy freeze.")
    if _sha256_file(source_cfg_path) != freeze["source_protocol_config_sha256"]:
        raise SystemExit("Source cross-model config changed after strategy freeze.")

    model_key = str(args.model_key)
    if model_key not in cfg["experiment"]["model_keys"]:
        raise SystemExit(f"Unknown model key: {model_key}")
    entry = freeze["model_entries"][model_key]
    selection_path = out_root / model_key / "strategy_selection.json"
    if _sha256_file(selection_path) != entry["strategy_selection_sha256"]:
        raise SystemExit(f"{model_key} strategy selection changed after freeze.")

    heldout_out = out_root / model_key / "fresh_heldout"
    if heldout_out.exists() and any(heldout_out.iterdir()):
        raise SystemExit(f"Refusing to overwrite fresh heldout directory: {heldout_out}")
    heldout_out.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = [
        *selection["screen_confirm_selections"],
        *selection["fixed_prefix_selections"],
    ]

    # Deduplicate exact strategy/budget duplicates while retaining both strategy labels
    # in the final table. Model calls are reused by prompt fingerprint within each
    # heldout seed, but reported calibration budgets remain the predeclared values.
    backend = _backend(source_cfg, model_key)
    print(f"Loading {model_key}: {backend['model']}", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))
    simulator = TextPersonaChoiceSimulator(client)

    original = initial_personas()
    fresh_seeds = [int(x) for x in cfg["fresh_heldout"]["seeds"]]
    variants = int(source_cfg["experiment"]["context_variants_per_intervention"])
    rows = []

    manifest = {
        "status": "screen_confirm_fresh_heldout_started",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "model_key": model_key,
        "model": source_cfg["models"][model_key]["model"],
        "strategy_freeze_sha256": _sha256_file(freeze_path),
        "strategy_selection_sha256": _sha256_file(selection_path),
        "fresh_heldout_seeds": fresh_seeds,
        "optimization_performed": False,
        "strategy_selection_uses_this_heldout": False,
        "runtime": client.runtime_info(),
    }
    (heldout_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for heldout_seed in fresh_seeds:
        run_dir = heldout_out / f"heldout_seed_{heldout_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        slates = calibration_slates(heldout_seed, variants)
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
        ).evaluate(original)
        original_obj = float(original_result["objective"])

        evaluated_by_fp = {_fingerprint(original): original_result}
        for record in records:
            population = _personas(record["selected_personas"])
            fp = _fingerprint(population)
            if fp in evaluated_by_fp:
                result = evaluated_by_fp[fp]
            else:
                result = HeldoutEvaluator(
                    simulator=simulator,
                    slates=slates,
                    target=moments,
                    output_dir=run_dir / f"population_{fp[:12]}",
                    condition=f"population_{fp[:12]}",
                ).evaluate(population)
                evaluated_by_fp[fp] = result

            heldout_obj = float(result["objective"])
            gain = (original_obj - heldout_obj) / original_obj
            rows.append({
                "model_key": model_key,
                "model": source_cfg["models"][model_key]["model"],
                "family": source_cfg["models"][model_key]["family"],
                "heldout_seed": heldout_seed,
                "strategy": str(record["strategy"]),
                "budget_label": str(record["budget_label"]),
                "logical_simulator_call_budget": int(
                    record["logical_simulator_call_budget"]
                ),
                "selected_evaluation_index": record.get(
                    "selected_evaluation_index"
                ),
                "selected_full_calibration_objective": float(
                    record["selected_full_calibration_objective"]
                ),
                "original_heldout_objective": original_obj,
                "selected_heldout_objective": heldout_obj,
                "heldout_improvement_fraction": gain,
                "heldout_improved": bool(heldout_obj < original_obj),
                "selected_prompt_fingerprint": fp,
            })
            print(
                f"{model_key} seed={heldout_seed} strategy={record['strategy']} "
                f"budget={record['logical_simulator_call_budget']}: "
                f"{original_obj:.6f} -> {heldout_obj:.6f} ({gain:+.2%})",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(heldout_out / "strategy_results.csv", index=False)

    summary_rows = []
    for (strategy, budget), group in frame.groupby(
        ["strategy", "logical_simulator_call_budget"], sort=True
    ):
        gains = group["heldout_improvement_fraction"].astype(float).tolist()
        summary_rows.append({
            "model_key": model_key,
            "strategy": strategy,
            "logical_simulator_call_budget": int(budget),
            "n_pairs": int(len(group)),
            "pairs_improved": int(group["heldout_improved"].astype(bool).sum()),
            "fraction_pairs_improved": float(
                group["heldout_improved"].astype(bool).mean()
            ),
            "mean_heldout_improvement_fraction": statistics.fmean(gains),
            "median_heldout_improvement_fraction": statistics.median(gains),
            "mean_selected_heldout_objective": float(
                group["selected_heldout_objective"].mean()
            ),
            "mean_original_heldout_objective": float(
                group["original_heldout_objective"].mean()
            ),
        })

    pd.DataFrame(summary_rows).to_csv(
        heldout_out / "strategy_summary.csv",
        index=False,
    )
    aggregate = {
        "status": "screen_confirm_fresh_heldout_complete",
        "model_key": model_key,
        "model": source_cfg["models"][model_key]["model"],
        "fresh_heldout_seeds": fresh_seeds,
        "strategy_summary": summary_rows,
        "optimization_performed": False,
        "strategy_selection_uses_this_heldout": False,
        "final_metrics_reporting_only": True,
    }
    (heldout_out / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
