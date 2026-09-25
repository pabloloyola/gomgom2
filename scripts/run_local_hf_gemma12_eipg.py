#!/usr/bin/env python3
"""One-process local Hugging Face EIPG experiment for an A100-40GB machine.

Loads Gemma 3 12B once, runs the frozen transfer gate, then (only if it passes)
runs the pre-heldout calibration/prompt-refinement protocol. No held-out LLM
contexts are constructed or scored.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eipg.experiments.llm_transfer import TransferGateThresholds, evaluate_transfer_gate
from eipg.experiments.prompt_refinement import PromptRefinementLoop
from eipg.personas.prompt_refinement import ResidualPromptEditor, ResidualPromptEditorConfig
from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig
from eipg.simulators.llm_choice import TextPersonaChoiceSimulator

from scripts.run_llm_prompt_refinement_calibration_v2 import (
    CalibrationEvaluator,
    _append_progress,
    _pairs,
    _selected,
    calibration_slates,
    initial_personas,
    target_moments,
    target_probability_table,
)
from scripts.run_llm_transfer_gate import diagnostic_pairs


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _run_transfer_gate(
    *,
    client: HuggingFaceLocalChatClient,
    cfg: dict,
    output: Path,
) -> dict:
    # Deliberately disable cache: repeated gate calls must be actual generations.
    previous_cache = client.cache_dir
    client.cache_dir = None
    simulator = TextPersonaChoiceSimulator(client)

    repeats = int(cfg["transfer_gate"]["repeats_per_pair"])
    request_rows: list[dict] = []
    choice_rows: list[dict] = []
    expected_by_pair: dict[str, int] = {}

    for pair_id, persona, slate, expected in diagnostic_pairs():
        expected_by_pair[pair_id] = expected
        for repeat_id in range(repeats):
            try:
                result = simulator.choose(slate=slate, persona=persona)
                request_rows.append(
                    {"pair_id": pair_id, "repeat_id": repeat_id, "parse_ok": True}
                )
                frame = slate.copy()
                frame["pair_id"] = pair_id
                frame["repeat_id"] = repeat_id
                frame["persona_id"] = persona.persona_id
                frame["chosen"] = (
                    frame["alternative_id"].astype(int) == result.alternative_id
                ).astype(int)
                choice_rows.extend(frame.to_dict("records"))
            except Exception as exc:
                request_rows.append(
                    {
                        "pair_id": pair_id,
                        "repeat_id": repeat_id,
                        "parse_ok": False,
                        "error": repr(exc),
                    }
                )

    client.cache_dir = previous_cache
    requests = pd.DataFrame(request_rows)
    choices = pd.DataFrame(choice_rows)

    if choices.empty:
        payload = {
            "model": client.config.model,
            "backend": "huggingface_local",
            "repeats_per_pair": repeats,
            "passed": False,
            "reason": "no_successful_choice_records",
            "request_failures": request_rows,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    chosen = choices.loc[choices["chosen"].astype(int) == 1]
    direction_checks: list[bool] = []
    for pair_id, group in chosen.groupby("pair_id"):
        modal = int(group["alternative_id"].astype(int).value_counts().idxmax())
        direction_checks.append(modal == expected_by_pair[pair_id])

    gate_cfg = cfg["transfer_gate"]
    report = evaluate_transfer_gate(
        request_log=requests,
        choice_records=choices,
        direction_checks=direction_checks,
        thresholds=TransferGateThresholds(
            min_parse_success=float(gate_cfg["min_parse_success"]),
            min_repeat_consistency=float(gate_cfg["min_repeat_consistency"]),
            min_directional_accuracy=float(gate_cfg["min_directional_accuracy"]),
        ),
    )
    payload = report.to_dict()
    payload.update(
        {
            "model": client.config.model,
            "backend": "huggingface_local",
            "repeats_per_pair": repeats,
            "direction_checks": direction_checks,
            "request_failures": requests.loc[
                ~requests["parse_ok"].astype(bool)
            ].to_dict("records"),
        }
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _serialize_history(histories: dict) -> tuple[dict, dict]:
    history_payload: dict = {}
    selected_payload: dict = {}
    for condition, history in histories.items():
        history_payload[condition] = [
            {
                "iteration": item.iteration,
                "objective": item.objective,
                "diagnostics": item.diagnostics,
                "personas": [
                    {
                        "persona_id": p.persona_id,
                        "segment_label": p.segment_label,
                        "prompt": p.prompt,
                        "prompt_sha256": sha256(p.prompt.encode("utf-8")).hexdigest(),
                    }
                    for p in item.personas
                ],
            }
            for item in history
        ]
        best = _selected(history)
        selected_payload[condition] = {
            "iteration": best.iteration,
            "objective": best.objective,
            "personas": [
                {
                    "persona_id": p.persona_id,
                    "segment_label": p.segment_label,
                    "prompt": p.prompt,
                    "prompt_sha256": sha256(p.prompt.encode("utf-8")).hexdigest(),
                }
                for p in best.personas
            ],
        }
    return history_payload, selected_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/llm_prompt_refinement_local_hf_gemma12.yaml",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/llm_prompt_refinement_local_hf_gemma12",
    )
    ap.add_argument(
        "--transfer-only",
        action="store_true",
        help="Run only the 12-request transfer gate, then exit.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    provenance = {
        "git_sha": _git_sha(),
        "config": args.config,
        "model": cfg["backend"]["model"],
        "dtype": cfg["backend"]["dtype"],
        "backend": "huggingface_local",
        "heldout_constructed": False,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Loading {cfg['backend']['model']} once via Hugging Face Transformers...", flush=True)
    client = HuggingFaceLocalChatClient(
        HuggingFaceLocalConfig.from_config(cfg["backend"])
    )

    gate_path = out / "transfer_gate.json"
    print("Running frozen transfer gate...", flush=True)
    gate = _run_transfer_gate(client=client, cfg=cfg, output=gate_path)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    if not bool(gate.get("passed", False)):
        raise SystemExit(
            f"Transfer gate failed. Inspect {gate_path}; calibration was not run."
        )
    if args.transfer_only:
        print("Transfer gate passed; --transfer-only requested, exiting.", flush=True)
        return

    calibration_out = out / "calibration"
    calibration_out.mkdir(parents=True, exist_ok=True)
    (calibration_out / "progress.jsonl").unlink(missing_ok=True)
    _append_progress(
        calibration_out,
        {
            "event": "run_started",
            "model": cfg["backend"]["model"],
            "provider": "HuggingFaceLocal",
        },
    )

    simulator = TextPersonaChoiceSimulator(client)
    editor_cfg = cfg["editor"]
    editor = ResidualPromptEditor(
        client,
        ResidualPromptEditorConfig(
            max_prompt_chars=int(editor_cfg["max_prompt_chars"]),
            max_residuals=int(editor_cfg["max_residuals"]),
        ),
    )

    slates = calibration_slates(
        int(cfg["experiment"]["controlled_seed"]),
        int(cfg["experiment"]["context_variants_per_intervention"]),
    )
    targets = target_probability_table(slates)
    moments = target_moments(targets)
    targets.to_csv(calibration_out / "synthetic_human_calibration_targets.csv", index=False)
    (calibration_out / "target_moments.json").write_text(
        json.dumps(moments, indent=2, sort_keys=True), encoding="utf-8"
    )

    evaluator = CalibrationEvaluator(simulator, slates, moments, calibration_out)
    loop = PromptRefinementLoop(
        editor=editor,
        evaluate=evaluator,
        iterations=int(cfg["experiment"]["iterations"]),
    )
    histories = loop.run_all(initial_personas=initial_personas())
    history_payload, selected_payload = _serialize_history(histories)

    initial = histories["original"][0].objective
    summary = {
        "status": str(cfg["experiment"]["name"]) + "_complete",
        "git_sha": _git_sha(),
        "heldout_constructed": False,
        "simulator_model": cfg["backend"]["model"],
        "simulator_backend": "huggingface_local",
        "editor_model": cfg["backend"]["model"],
        "editor_backend": "huggingface_local",
        "dtype": cfg["backend"]["dtype"],
        "iterations": int(cfg["experiment"]["iterations"]),
        "context_variants_per_intervention": int(
            cfg["experiment"]["context_variants_per_intervention"]
        ),
        "n_calibration_choice_observations_per_evaluation": len(
            _pairs(initial_personas(), slates)
        ),
        "initial_objective": initial,
        "selected": selected_payload,
        "economic_selected_improvement_fraction": (
            initial - selected_payload["economic_residual_rewrite"]["objective"]
        )
        / initial,
        "generic_selected_improvement_fraction": (
            initial - selected_payload["generic_rewrite_control"]["objective"]
        )
        / initial,
        "evaluation_calls": evaluator.calls,
        "transfer_gate": gate,
    }

    (calibration_out / "history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (calibration_out / "selected_prompts.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (calibration_out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _append_progress(
        calibration_out,
        {
            "event": "run_completed",
            "initial_objective": initial,
            "economic_selected_improvement_fraction": summary[
                "economic_selected_improvement_fraction"
            ],
            "generic_selected_improvement_fraction": summary[
                "generic_selected_improvement_fraction"
            ],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
