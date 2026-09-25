#!/usr/bin/env python3
"""Run frozen GEPA anchored-text calibration for one simulator model."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gepa

from eipg.experiments.gepa_persona import EIPGGEPAAdapter, LocalHFReflectionLM
from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig
from eipg.simulators.llm_choice import TextPersonaChoiceSimulator
from scripts.run_cross_model_anchored_heldout import _backend


REFLECTION_TEMPLATE = """You are evolving one behavioral adjustment suffix for an immutable consumer persona.

Current suffix:
<CURRENT>
<curr_param>
</CURRENT>

Behavioral evaluation traces and directional economic feedback:
<FEEDBACK>
<side_info>
</FEEDBACK>

Write a revised suffix that makes the smallest targeted behavioral change supported by the feedback.

Hard constraints:
- Return only the revised suffix text.
- Do not rewrite or restate the persona identity.
- Do not mention calibration, residuals, objectives, scores, targets, experiments, context IDs, model names, held-out/test data, or specific alternatives.
- Do not copy numeric values from the feedback.
- Do not tell the persona to choose a particular alternative.
- Express only general preferences or tradeoffs that can transfer across product contexts.
- Preserve unrelated preferences.
- Prefer concise changes over long instructions.
"""


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class LogicalChoiceStopper:
    def __init__(self, adapter: EIPGGEPAAdapter, max_calls: int) -> None:
        self.adapter = adapter
        self.max_calls = int(max_calls)

    def __call__(self, _state) -> bool:
        return int(self.adapter.logical_calls) >= self.max_calls


class DiscoveryBudgetRecorder:
    """Record logical query count when each full-val candidate enters GEPA's pool."""

    def __init__(self, adapter: EIPGGEPAAdapter) -> None:
        self.adapter = adapter
        self.discovery_calls = adapter.candidate_discovery_calls

    def on_valset_evaluated(self, event) -> None:
        self.discovery_calls[int(event["candidate_idx"])] = int(self.adapter.logical_calls)


def _checkpoint_payload(
    *,
    result,
    adapter,
    reflection_lm,
    recorder,
    label: str,
    budget: int,
    model_key: str,
):
    eligible = [
        idx
        for idx, calls in recorder.discovery_calls.items()
        if int(calls) <= int(budget)
    ]
    if 0 not in eligible and result.candidates:
        eligible.append(0)
        recorder.discovery_calls.setdefault(0, 120)
    if not eligible:
        raise RuntimeError(f"No GEPA candidate discovered within budget {budget}")

    best_idx = max(
        eligible,
        key=lambda idx: (float(result.val_aggregate_scores[idx]), -int(idx)),
    )
    candidate = dict(result.candidates[best_idx])
    ok, error = adapter.renderer.validate(candidate)
    if not ok:
        raise RuntimeError(f"GEPA checkpoint candidate failed suffix validation: {error}")
    personas = adapter.renderer.render(candidate)

    return {
        "status": f"{label}_frozen",
        "git_sha": _git_sha(),
        "model_key": model_key,
        "budget_label": label,
        "logical_choice_query_budget": int(budget),
        "selected_candidate_discovery_calls": int(recorder.discovery_calls[best_idx]),
        "gepa_native_total_metric_calls_at_run_end": int(result.total_metric_calls or 0),
        "gepa_run_logical_calls_at_end": int(adapter.logical_calls),
        "gepa_num_candidates_at_run_end": int(result.num_candidates),
        "gepa_best_index_within_budget": int(best_idx),
        "gepa_best_score_within_budget": float(result.val_aggregate_scores[best_idx]),
        "reflection_lm_calls_at_run_end": int(reflection_lm.calls),
        "candidate_suffixes": candidate,
        "selected_personas": [
            {
                "persona_id": p.persona_id,
                "segment_label": p.segment_label,
                "prompt": p.prompt,
                "prompt_sha256": sha256(p.prompt.encode("utf-8")).hexdigest(),
            }
            for p in personas
        ],
        "candidate_discovery_logical_calls": {
            str(k): int(v) for k, v in sorted(recorder.discovery_calls.items())
        },
        "heldout_evaluated": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_gepa_anchored_text.yaml")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--output-root", default="outputs/cross_model_gepa_anchored_text_v1")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_key = str(args.model_key)
    if model_key not in cfg["experiment"]["model_keys"]:
        raise SystemExit(f"Unknown model key: {model_key}")

    source_cfg = yaml.safe_load(
        Path("configs/cross_model_anchored_core.yaml").read_text(encoding="utf-8")
    )
    out = Path(args.output_root) / model_key
    if out.exists() and (out / "calibration_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite completed GEPA calibration: {out}")

    protocol_marker = out / "run_protocol.json"
    legacy_partial = out / "gepa_run"
    if legacy_partial.exists() and not protocol_marker.exists():
        raise SystemExit(
            "Found a pre-fix/incompatible partial GEPA run at "
            f"{out}. Its failed reflection iterations consumed behavioral queries, "
            "so it must not be resumed for the frozen budget comparison. "
            f"Remove {out} and rerun this model."
        )

    out.mkdir(parents=True, exist_ok=True)
    if not protocol_marker.exists():
        protocol_marker.write_text(
            json.dumps(
                {
                    "protocol": cfg["experiment"]["name"],
                    "config_sha256": _sha256_file(cfg_path),
                    "adapter_state_schema": 1,
                    "git_sha_at_start": _git_sha(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    sim_backend = _backend(source_cfg, model_key)
    print(f"Loading simulator {model_key}: {sim_backend['model']}", flush=True)
    sim_client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(sim_backend))
    simulator = TextPersonaChoiceSimulator(sim_client)

    reflection_key = str(cfg["reflection"]["model_key"])
    reflection_backend = dict(_backend(source_cfg, reflection_key))
    reflection_backend["max_new_tokens"] = int(cfg["reflection"]["max_new_tokens"])
    reflection_backend["do_sample"] = bool(cfg["reflection"]["do_sample"])
    reflection_backend["temperature"] = cfg["reflection"]["temperature"]
    reflection_backend["cache_dir"] = str(Path(".cache/eipg/gepa_reflection") / reflection_key)
    if model_key == reflection_key:
        print("Reusing Gemma-12B simulator client as fixed GEPA reflection LM.", flush=True)
        reflection_client = sim_client
    else:
        print(f"Loading fixed GEPA reflection LM: {reflection_backend['model']}", flush=True)
        reflection_client = HuggingFaceLocalChatClient(
            HuggingFaceLocalConfig.from_config(reflection_backend)
        )
    reflection_lm = LocalHFReflectionLM(reflection_client)

    adapter = EIPGGEPAAdapter(
        simulator=simulator,
        calibration_seed=int(cfg["experiment"]["calibration_seed"]),
        variants=int(cfg["experiment"]["context_variants_per_intervention"]),
        output_dir=out / "adapter",
        reflection_cfg=cfg["reflection"],
    )

    trainset = [
        {"kind": "variant", "variant": i}
        for i in range(int(cfg["experiment"]["context_variants_per_intervention"]))
    ]
    valset = [{"kind": "full"}]
    seed_candidate = {
        "budget_suffix": "",
        "quality_suffix": "",
        "sustain_suffix": "",
    }

    max_budget = max(
        int(x["logical_choice_query_budget"])
        for x in cfg["gepa"]["budget_checkpoints"]
    )
    recorder = DiscoveryBudgetRecorder(adapter)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy=str(cfg["gepa"]["candidate_selection_strategy"]),
        frontier_type=str(cfg["gepa"]["frontier_type"]),
        skip_perfect_score=False,
        reflection_minibatch_size=int(cfg["gepa"]["reflection_minibatch_size"]),
        reflection_prompt_template=REFLECTION_TEMPLATE,
        module_selector=str(cfg["gepa"]["module_selector"]),
        use_merge=bool(cfg["gepa"]["use_merge"]),
        max_merge_invocations=int(cfg["gepa"]["max_merge_invocations"]),
        stop_callbacks=[LogicalChoiceStopper(adapter, max_budget)],
        run_dir=str(out / "gepa_run"),
        cache_evaluation=bool(cfg["gepa"]["cache_evaluation"]),
        seed=int(cfg["gepa"]["seed"]),
        acceptance_criterion=str(cfg["gepa"]["acceptance_criterion"]),
        track_best_outputs=True,
        display_progress_bar=True,
        callbacks=[recorder],
    )

    if not recorder.discovery_calls:
        raise RuntimeError("GEPA callback did not record candidate discovery budgets.")

    checkpoint_files = {}
    for checkpoint in cfg["gepa"]["budget_checkpoints"]:
        label = str(checkpoint["label"])
        budget = int(checkpoint["logical_choice_query_budget"])
        payload = _checkpoint_payload(
            result=result,
            adapter=adapter,
            reflection_lm=reflection_lm,
            recorder=recorder,
            label=label,
            budget=budget,
            model_key=model_key,
        )
        path = out / f"{label}_selection.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        checkpoint_files[label] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        print(
            f"{model_key} {label}: selected_idx={payload['gepa_best_index_within_budget']} "
            f"discovery_calls={payload['selected_candidate_discovery_calls']} "
            f"score={payload['gepa_best_score_within_budget']:.6f}",
            flush=True,
        )

    manifest = {
        "status": "gepa_calibration_complete_heldout_unseen",
        "git_sha": _git_sha(),
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "gepa_version": str(cfg["gepa"]["package_version"]),
        "model_key": model_key,
        "simulator_model": sim_backend["model"],
        "reflection_model": reflection_backend["model"],
        "run_logical_choice_queries": int(adapter.logical_calls),
        "reflection_lm_calls": int(reflection_lm.calls),
        "fresh_heldout_evaluated": False,
        "checkpoint_files": checkpoint_files,
    }
    (out / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
