#!/usr/bin/env python3
"""Seed robustness for the frozen anchored-mu coordinate-search mechanism.

Runs the exact same anchored-mu search over three additional calibration-context
seeds, while retaining seed 1729 as the frozen reference result. The local model
is loaded once. No held-out contexts are constructed or scored.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eipg.personas.anchored_mu import FEATURES, initial_anchored_personas, neighborhood, render_population
from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig
from eipg.simulators.llm_choice import TextPersonaChoiceSimulator

from scripts.run_llm_prompt_refinement_calibration_v2 import (
    CalibrationEvaluator,
    _append_progress,
    _pairs,
    calibration_slates,
    target_moments,
    target_probability_table,
)
from scripts.run_local_hf_gemma12_eipg import _run_transfer_gate


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _persona_payload(personas):
    rendered = render_population(personas)
    return [
        {
            "persona_id": structured.persona_id,
            "segment_label": structured.segment_label,
            "delta_mu": structured.as_dict(),
            "prompt": text_persona.prompt,
        }
        for structured, text_persona in zip(personas, rendered)
    ]


def run_seed(*, seed: int, cfg: dict, client, seed_out: Path) -> dict:
    seed_out.mkdir(parents=True, exist_ok=True)
    (seed_out / "progress.jsonl").unlink(missing_ok=True)

    simulator = TextPersonaChoiceSimulator(client)
    slates = calibration_slates(
        int(seed),
        int(cfg["experiment"]["context_variants_per_intervention"]),
    )
    targets = target_probability_table(slates)
    moments = target_moments(targets)
    targets.to_csv(seed_out / "synthetic_human_calibration_targets.csv", index=False)
    (seed_out / "target_moments.json").write_text(
        json.dumps(moments, indent=2, sort_keys=True), encoding="utf-8"
    )

    evaluator = CalibrationEvaluator(simulator, slates, moments, seed_out)
    current_personas = initial_anchored_personas()
    rendered_initial = render_population(current_personas)
    logical_calls_per_evaluation = len(_pairs(rendered_initial, slates))
    current_eval = evaluator(rendered_initial)
    initial_objective = float(current_eval.objective)

    history = [{
        "iteration": 0,
        "status": "initial_exact_original_prompts",
        "selected": True,
        "objective": initial_objective,
        "evaluation_index": 0,
        "logical_calls_per_evaluation": logical_calls_per_evaluation,
        "cumulative_logical_simulator_calls": logical_calls_per_evaluation,
        "personas": _persona_payload(current_personas),
        "residuals": [asdict(r) | {"error": r.error} for r in current_eval.residuals],
    }]

    step = float(cfg["experiment"]["coordinate_step"])
    lower = float(cfg["mu"]["lower_bound"])
    upper = float(cfg["mu"]["upper_bound"])
    max_iterations = int(cfg["experiment"]["max_accepted_iterations"])
    accepted = 0
    accepted_moves = []

    for iteration in range(1, max_iterations + 1):
        candidates = []
        for candidate_index, (persona_index, feature, signed_step, candidate_personas) in enumerate(
            neighborhood(current_personas, step=step, lower=lower, upper=upper),
            start=1,
        ):
            candidate_eval = evaluator(render_population(candidate_personas))
            record = {
                "iteration": iteration,
                "candidate_index": candidate_index,
                "status": "candidate",
                "selected": False,
                "persona_index": persona_index,
                "persona_id": candidate_personas[persona_index].persona_id,
                "feature": feature,
                "signed_step": signed_step,
                "objective": float(candidate_eval.objective),
                "evaluation_index": evaluator.calls - 1,
                "logical_calls_per_evaluation": logical_calls_per_evaluation,
                "cumulative_logical_simulator_calls": evaluator.calls * logical_calls_per_evaluation,
                "personas": _persona_payload(candidate_personas),
                "residuals": [asdict(r) | {"error": r.error} for r in candidate_eval.residuals],
            }
            history.append(record)
            candidates.append((
                float(candidate_eval.objective),
                persona_index,
                feature,
                signed_step,
                candidate_personas,
                candidate_eval,
                record,
            ))

        best = min(candidates, key=lambda x: (x[0], x[1], FEATURES.index(x[2]), x[3]))
        best_obj, pidx, feature, signed_step, best_personas, best_eval, best_record = best
        if best_obj + 1e-12 < float(current_eval.objective):
            best_record["selected"] = True
            best_record["status"] = "accepted"
            accepted += 1
            accepted_moves.append({
                "iteration": iteration,
                "persona_id": best_personas[pidx].persona_id,
                "feature": feature,
                "signed_step": signed_step,
                "objective": best_obj,
            })
            current_personas = best_personas
            current_eval = best_eval
            _append_progress(seed_out, {
                "event": "coordinate_update_accepted",
                "seed": seed,
                "iteration": iteration,
                "persona_id": current_personas[pidx].persona_id,
                "feature": feature,
                "signed_step": signed_step,
                "objective": best_obj,
            })
        else:
            best_record["status"] = "best_rejected_no_improvement"
            _append_progress(seed_out, {
                "event": "coordinate_search_stopped",
                "seed": seed,
                "iteration": iteration,
                "current_objective": float(current_eval.objective),
                "best_candidate_objective": best_obj,
            })
            break

    final_objective = float(current_eval.objective)
    improvement = (initial_objective - final_objective) / initial_objective
    result = {
        "seed": seed,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "improvement_fraction": improvement,
        "accepted_iterations": accepted,
        "accepted_moves": accepted_moves,
        "selected_personas": _persona_payload(current_personas),
        "evaluation_calls": evaluator.calls,
        "logical_calls_per_evaluation": logical_calls_per_evaluation,
        "logical_simulator_calls": evaluator.calls * logical_calls_per_evaluation,
    }
    (seed_out / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
    )
    (seed_out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/anchored_mu_seed_robustness_local_hf_gemma12.yaml",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/anchored_mu_seed_robustness_local_hf_gemma12",
    )
    ap.add_argument("--skip-transfer-gate", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    provenance = {
        "git_sha": _git_sha(),
        "config": args.config,
        "model": cfg["backend"]["model"],
        "backend": "huggingface_local",
        "robustness_axis": "calibration_context_seed",
        "heldout_constructed": False,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Loading {cfg['backend']['model']} once via Hugging Face Transformers...", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(cfg["backend"]))

    if args.skip_transfer_gate:
        gate = {
            "passed": True,
            "skipped": True,
            "reason": "same_local_model_already_passed_frozen_gate",
        }
    else:
        gate = _run_transfer_gate(client=client, cfg=cfg, output=out / "transfer_gate.json")
        if not bool(gate.get("passed", False)):
            raise SystemExit("Transfer gate failed; seed robustness not run.")

    reference_seed = int(cfg["experiment"]["reference_seed"])
    ref = dict(cfg["reference_result"])
    reference = {
        "seed": reference_seed,
        "initial_objective": float(ref["initial_objective"]),
        "final_objective": float(ref["final_objective"]),
        "improvement_fraction": float(ref["improvement_fraction"]),
        "accepted_iterations": int(ref["accepted_iterations"]),
        "accepted_moves": [{
            "iteration": 1,
            **dict(ref["selected_coordinate"]),
            "objective": float(ref["final_objective"]),
        }],
        "source": "frozen_reference_result",
    }

    results = [reference]
    for seed in [int(x) for x in cfg["experiment"]["additional_seeds"]]:
        print(f"Running anchored-mu robustness seed {seed}...", flush=True)
        results.append(
            run_seed(
                seed=seed,
                cfg=cfg,
                client=client,
                seed_out=out / f"seed_{seed}",
            )
        )

    improvements = [float(r["improvement_fraction"]) for r in results]
    finals = [float(r["final_objective"]) for r in results]
    initials = [float(r["initial_objective"]) for r in results]
    first_moves = [
        r["accepted_moves"][0] if r.get("accepted_moves") else None
        for r in results
    ]
    move_labels = [
        None if m is None else f"{m['persona_id']}.{m['feature']}:{float(m['signed_step']):+.2f}"
        for m in first_moves
    ]
    reference_label = move_labels[0]
    same_as_reference = sum(label == reference_label for label in move_labels)

    aggregate = {
        "status": str(cfg["experiment"]["name"]) + "_complete",
        "git_sha": _git_sha(),
        "heldout_constructed": False,
        "model": cfg["backend"]["model"],
        "backend": "huggingface_local",
        "seeds": [r["seed"] for r in results],
        "n_seeds": len(results),
        "per_seed": results,
        "mean_initial_objective": statistics.fmean(initials),
        "mean_final_objective": statistics.fmean(finals),
        "mean_improvement_fraction": statistics.fmean(improvements),
        "median_improvement_fraction": statistics.median(improvements),
        "min_improvement_fraction": min(improvements),
        "max_improvement_fraction": max(improvements),
        "seeds_with_positive_improvement": sum(x > 0 for x in improvements),
        "first_move_labels": move_labels,
        "reference_first_move_label": reference_label,
        "seeds_matching_reference_first_move": same_as_reference,
        "fraction_matching_reference_first_move": same_as_reference / len(results),
        "transfer_gate": gate,
    }

    (out / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
