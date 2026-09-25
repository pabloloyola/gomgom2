#!/usr/bin/env python3
"""Evaluate frozen 1,320-call strategies on a third fresh heldout seed set."""
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

from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig
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
    if [p.persona_id for p in result] != ["budget", "quality", "sustain"]:
        raise ValueError("Unexpected persona order")
    return result


def _fingerprint(personas: tuple[TextPersona, ...]) -> str:
    raw = json.dumps(
        [
            {"persona_id": p.persona_id, "segment_label": p.segment_label, "prompt": p.prompt}
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
        default="configs/cross_model_random_subset_baseline.yaml",
    )
    ap.add_argument("--model-key", required=True)
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_random_subset_baseline_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_cfg_path = Path("configs/cross_model_anchored_core.yaml")
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))

    out_root = Path(args.output_root)
    freeze_path = out_root / "baseline_cohort_freeze.json"
    if not freeze_path.exists():
        raise SystemExit("Baseline cohort is not frozen. Run derivation first.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256_file(cfg_path) != freeze["config_sha256"]:
        raise SystemExit("Random-subset config changed after freeze.")

    model_key = str(args.model_key)
    if model_key not in cfg["experiment"]["model_keys"]:
        raise SystemExit(f"Unknown model key: {model_key}")
    entry = freeze["model_entries"][model_key]
    selection_path = out_root / model_key / "baseline_selection.json"
    if _sha256_file(selection_path) != entry["baseline_selection_sha256"]:
        raise SystemExit(f"{model_key} baseline selections changed after freeze.")

    heldout_out = out_root / model_key / "fresh_heldout"
    if heldout_out.exists() and any(heldout_out.iterdir()):
        raise SystemExit(f"Refusing to overwrite heldout directory: {heldout_out}")
    heldout_out.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = selection["comparison_records"]

    backend = _backend(source_cfg, model_key)
    print(f"Loading {model_key}: {backend['model']}", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))
    simulator = TextPersonaChoiceSimulator(client)

    original = initial_personas()
    fresh_seeds = [int(x) for x in cfg["fresh_heldout"]["seeds"]]
    variants = int(source_cfg["experiment"]["context_variants_per_intervention"])
    rows = []

    manifest = {
        "status": "random_subset_fresh_heldout_started",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "model_key": model_key,
        "model": source_cfg["models"][model_key]["model"],
        "baseline_freeze_sha256": _sha256_file(freeze_path),
        "baseline_selection_sha256": _sha256_file(selection_path),
        "fresh_heldout_seeds": fresh_seeds,
        "optimization_performed": False,
        "selection_uses_this_heldout": False,
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
                "subset_seed": record.get("subset_seed"),
                "logical_simulator_call_budget": int(record["logical_simulator_call_budget"]),
                "selected_evaluation_index": int(record["selected_evaluation_index"]),
                "selected_full_calibration_objective": float(
                    record["selected_full_calibration_objective"]
                ),
                "original_heldout_objective": original_obj,
                "selected_heldout_objective": heldout_obj,
                "heldout_improvement_fraction": gain,
                "heldout_improved": bool(heldout_obj < original_obj),
                "selected_prompt_fingerprint": fp,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(heldout_out / "strategy_results.csv", index=False)

    summary_rows = []
    grouped = frame.groupby(["strategy", "budget_label"], sort=True, dropna=False)
    for (strategy, label), group in grouped:
        gains = group["heldout_improvement_fraction"].astype(float).tolist()
        summary_rows.append({
            "model_key": model_key,
            "strategy": strategy,
            "budget_label": label,
            "subset_seed": (
                None if group["subset_seed"].isna().all()
                else int(group["subset_seed"].dropna().iloc[0])
            ),
            "logical_simulator_call_budget": int(
                group["logical_simulator_call_budget"].iloc[0]
            ),
            "n_pairs": int(len(group)),
            "pairs_improved": int(group["heldout_improved"].astype(bool).sum()),
            "fraction_pairs_improved": float(group["heldout_improved"].astype(bool).mean()),
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
    print(pd.DataFrame(summary_rows).to_csv(index=False))


if __name__ == "__main__":
    main()
