#!/usr/bin/env python3
"""Evaluate frozen GEPA and prior strategies on the fifth fresh heldout set."""
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


def _load_records(model_key: str, cfg: dict, gepa_root: Path):
    records = []

    adaptive_root = Path(cfg["comparison"]["adaptive_root"])
    screen_root = Path(cfg["comparison"]["screen_confirm_root"])
    random_root = Path(cfg["comparison"]["random_subset_root"])
    full_root = Path(cfg["comparison"]["full_coordinate_root"])

    residual = json.loads(
        (adaptive_root / model_key / "residual_linucb_selection.json").read_text(encoding="utf-8")
    )
    history_llm = json.loads(
        (adaptive_root / model_key / "history_llm_selection.json").read_text(encoding="utf-8")
    )
    screen = json.loads(
        (screen_root / model_key / "strategy_selection.json").read_text(encoding="utf-8")
    )
    screen_1320 = next(
        r for r in screen["screen_confirm_selections"]
        if r["budget_label"] == "screen_confirm_1320"
    )
    random_payload = json.loads(
        (random_root / model_key / "baseline_selection.json").read_text(encoding="utf-8")
    )
    random_rows = [
        r for r in random_payload["comparison_records"]
        if r["strategy"] == "random_subset"
    ]
    fixed = next(
        r for r in random_payload["comparison_records"]
        if r["strategy"] == "fixed_prefix"
    )

    budget_trace = json.loads(
        (
            full_root
            / model_key
            / "calibration"
            / f"seed_{int(cfg['experiment']['calibration_seed'])}"
            / "budget_trace.json"
        ).read_text(encoding="utf-8")
    )
    full = next(r for r in budget_trace["checkpoints"] if r["label"] == "full_search")

    def add(strategy, label, budget, selected_personas, calibration_objective, subset_seed=None):
        records.append({
            "strategy": strategy,
            "budget_label": label,
            "logical_simulator_choice_query_budget": int(budget),
            "subset_seed": subset_seed,
            "selected_personas": selected_personas,
            "selected_full_calibration_objective": float(calibration_objective),
        })

    add(
        "fixed_prefix",
        "fixed_prefix_1320",
        1320,
        fixed["selected_personas"],
        fixed["selected_full_calibration_objective"],
    )
    add(
        "screen_confirm",
        "screen_confirm_1320",
        1320,
        screen_1320["selected_personas"],
        screen_1320["selected_full_calibration_objective"],
    )
    add(
        "residual_linucb",
        "residual_linucb_1320",
        1320,
        residual["selected_personas"],
        residual["selected_full_calibration_objective"],
    )
    add(
        "history_llm",
        "history_llm_1320",
        1320,
        history_llm["selected_personas"],
        history_llm["selected_full_calibration_objective"],
    )
    for row in random_rows:
        add(
            "random_subset",
            f"random_subset_seed_{int(row['subset_seed'])}",
            1320,
            row["selected_personas"],
            row["selected_full_calibration_objective"],
            subset_seed=int(row["subset_seed"]),
        )

    for label, budget in (("gepa_1320", 1320), ("gepa_3720", 3720)):
        payload = json.loads(
            (gepa_root / model_key / f"{label}_selection.json").read_text(encoding="utf-8")
        )
        add(
            "gepa",
            label,
            budget,
            payload["selected_personas"],
            -float(payload["gepa_best_score_within_budget"]),
        )

    add(
        "full_coordinate",
        "full_coordinate_3720",
        3720,
        full["selected_personas"],
        full["selected_objective"],
    )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_gepa_anchored_text.yaml")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--output-root", default="outputs/cross_model_gepa_anchored_text_v1")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_cfg = yaml.safe_load(
        Path("configs/cross_model_anchored_core.yaml").read_text(encoding="utf-8")
    )
    root = Path(args.output_root)
    freeze_path = root / "gepa_cohort_freeze.json"
    if not freeze_path.exists():
        raise SystemExit("GEPA cohort is not frozen.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256_file(cfg_path) != freeze["config_sha256"]:
        raise SystemExit("GEPA config changed after freeze.")

    model_key = str(args.model_key)
    if model_key not in freeze["models"]:
        raise SystemExit(f"Model not frozen: {model_key}")
    for label, artifact in freeze["models"][model_key]["checkpoints"].items():
        p = Path(artifact["path"])
        if _sha256_file(p) != artifact["sha256"]:
            raise SystemExit(f"Frozen GEPA checkpoint changed: {label}")

    heldout_dir = root / model_key / "fresh_heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {heldout_dir}")
    heldout_dir.mkdir(parents=True, exist_ok=True)

    backend = _backend(source_cfg, model_key)
    print(f"Loading heldout simulator {model_key}: {backend['model']}", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(backend))
    simulator = TextPersonaChoiceSimulator(client)

    records = _load_records(model_key, cfg, root)
    seeds = [int(x) for x in cfg["fresh_heldout"]["seeds"]]
    variants = int(cfg["experiment"]["context_variants_per_intervention"])

    (heldout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "gepa_fifth_heldout_started",
                "git_sha": _git_sha(),
                "model_key": model_key,
                "model": backend["model"],
                "gepa_freeze_sha256": _sha256_file(freeze_path),
                "fresh_heldout_seeds": seeds,
                "optimization_performed": False,
                "runtime": client.runtime_info(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    original = initial_personas()
    rows = []
    for seed in seeds:
        run_dir = heldout_dir / f"heldout_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        slates = calibration_slates(seed, variants)
        moments = target_moments(target_probability_table(slates))
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
        heldout_dir / "strategy_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
