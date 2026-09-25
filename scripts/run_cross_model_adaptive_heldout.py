#!/usr/bin/env python3
"""Evaluate frozen adaptive and equal-budget baselines on fourth fresh heldout set."""
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


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _personas(rows):
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


def _fingerprint(personas) -> str:
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


def _records_for_model(model_key, cfg, out_root):
    random_root = Path(cfg["comparison"]["source_random_subset_root"])
    screen_root = Path(cfg["comparison"]["source_screen_confirm_root"])

    prior = json.loads(
        (random_root / model_key / "baseline_selection.json").read_text(encoding="utf-8")
    )
    prior_records = prior["comparison_records"]
    fixed = next(r for r in prior_records if r["strategy"] == "fixed_prefix")
    random_records = [r for r in prior_records if r["strategy"] == "random_subset"]

    screen = json.loads(
        (screen_root / model_key / "strategy_selection.json").read_text(encoding="utf-8")
    )
    screen_row = next(
        r for r in screen["screen_confirm_selections"]
        if r["budget_label"] == "screen_confirm_1320"
    )

    residual = json.loads(
        (out_root / model_key / "residual_linucb_selection.json").read_text(encoding="utf-8")
    )
    llm = json.loads(
        (out_root / model_key / "history_llm_selection.json").read_text(encoding="utf-8")
    )

    def normalize(row, strategy, label, subset_seed=None, proposer_calls=None):
        return {
            "strategy": strategy,
            "budget_label": label,
            "subset_seed": subset_seed,
            "logical_simulator_choice_query_budget": int(
                cfg["budget"]["logical_simulator_choice_queries"]
            ),
            "selected_evaluation_index": int(row["selected_evaluation_index"]),
            "selected_full_calibration_objective": float(
                row["selected_full_calibration_objective"]
            ),
            "selected_personas": row["selected_personas"],
            "proposer_calls": proposer_calls,
        }

    records = [
        normalize(fixed, "fixed_prefix", "fixed_prefix_1320"),
        normalize(screen_row, "screen_confirm", "screen_confirm_1320"),
        normalize(
            residual,
            "residual_linucb",
            "residual_linucb_1320",
        ),
        normalize(
            llm,
            "history_llm",
            "history_llm_1320",
            proposer_calls=int(llm["proposer_calls"]),
        ),
    ]
    records.extend(
        normalize(
            row,
            "random_subset",
            f"random_subset_seed_{int(row['subset_seed'])}",
            subset_seed=int(row["subset_seed"]),
        )
        for row in random_records
    )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_adaptive_search.yaml")
    ap.add_argument("--model-key", required=True)
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_adaptive_outer_search_v1",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_cfg = yaml.safe_load(
        Path("configs/cross_model_anchored_core.yaml").read_text(encoding="utf-8")
    )
    out_root = Path(args.output_root)
    freeze_path = out_root / "adaptive_cohort_freeze.json"
    if not freeze_path.exists():
        raise SystemExit("Adaptive cohort is not frozen.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256_file(cfg_path) != freeze["config_sha256"]:
        raise SystemExit("Adaptive config changed after freeze.")

    model_key = str(args.model_key)
    if model_key not in freeze["models"]:
        raise SystemExit(f"Model not frozen: {model_key}")

    for artifact in freeze["models"][model_key]["artifacts"].values():
        p = Path(artifact["path"])
        if _sha256_file(p) != artifact["sha256"]:
            raise SystemExit(f"Frozen strategy artifact changed: {p}")

    heldout_dir = out_root / model_key / "fresh_heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {heldout_dir}")
    heldout_dir.mkdir(parents=True, exist_ok=True)

    backend = _backend(source_cfg, model_key)
    print(f"Loading {model_key}: {backend['model']}", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))
    simulator = TextPersonaChoiceSimulator(client)
    records = _records_for_model(model_key, cfg, out_root)
    seeds = [int(x) for x in cfg["fresh_heldout"]["seeds"]]
    variants = int(source_cfg["experiment"]["context_variants_per_intervention"])

    (heldout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "adaptive_fourth_heldout_started",
                "git_sha": _git_sha(),
                "model_key": model_key,
                "model": backend["model"],
                "adaptive_freeze_sha256": _sha256_file(freeze_path),
                "fresh_heldout_seeds": seeds,
                "optimization_performed": False,
                "runtime": client.runtime_info(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows = []
    original = initial_personas()
    for seed in seeds:
        run_dir = heldout_dir / f"heldout_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        slates = calibration_slates(seed, variants)
        target_probs = target_probability_table(slates)
        moments = target_moments(target_probs)

        original_result = HeldoutEvaluator(
            simulator=simulator,
            slates=slates,
            target=moments,
            output_dir=run_dir / "original",
            condition="original",
        ).evaluate(original)
        original_obj = float(original_result["objective"])
        evaluated = {_fingerprint(original): original_result}

        for record in records:
            population = _personas(record["selected_personas"])
            fp = _fingerprint(population)
            if fp not in evaluated:
                evaluated[fp] = HeldoutEvaluator(
                    simulator=simulator,
                    slates=slates,
                    target=moments,
                    output_dir=run_dir / f"population_{fp[:12]}",
                    condition=f"population_{fp[:12]}",
                ).evaluate(population)
            result = evaluated[fp]
            obj = float(result["objective"])
            gain = (original_obj - obj) / original_obj
            rows.append({
                "model_key": model_key,
                "model": backend["model"],
                "family": source_cfg["models"][model_key]["family"],
                "heldout_seed": seed,
                "strategy": record["strategy"],
                "budget_label": record["budget_label"],
                "subset_seed": record["subset_seed"],
                "logical_simulator_choice_query_budget": record[
                    "logical_simulator_choice_query_budget"
                ],
                "proposer_calls": record["proposer_calls"],
                "selected_evaluation_index": record["selected_evaluation_index"],
                "selected_full_calibration_objective": record[
                    "selected_full_calibration_objective"
                ],
                "original_heldout_objective": original_obj,
                "selected_heldout_objective": obj,
                "heldout_improvement_fraction": gain,
                "heldout_improved": bool(obj < original_obj),
                "selected_prompt_fingerprint": fp,
            })
            print(
                f"{model_key} seed={seed} {record['budget_label']}: "
                f"{original_obj:.6f}->{obj:.6f} ({gain:+.2%})",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(heldout_dir / "strategy_results.csv", index=False)

    summary_rows = []
    for (strategy, label), group in frame.groupby(
        ["strategy", "budget_label"], sort=True, dropna=False
    ):
        gains = group["heldout_improvement_fraction"].astype(float).tolist()
        summary_rows.append({
            "model_key": model_key,
            "strategy": strategy,
            "budget_label": label,
            "subset_seed": (
                None if group["subset_seed"].isna().all()
                else int(group["subset_seed"].dropna().iloc[0])
            ),
            "logical_simulator_choice_query_budget": int(
                group["logical_simulator_choice_query_budget"].iloc[0]
            ),
            "proposer_calls": (
                None if group["proposer_calls"].isna().all()
                else int(group["proposer_calls"].dropna().iloc[0])
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
        heldout_dir / "strategy_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
