#!/usr/bin/env python3
"""History-aware LLM policy over the fixed 30-action calibration replay."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig
from scripts.derive_cross_model_residual_linucb import _history


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256_file(path):
    return sha256(path.read_bytes()).hexdigest()


def _ranked_residuals(row, limit=10):
    ranked = sorted(row["residuals"], key=lambda r: abs(float(r["error"])) * abs(float(r.get("weight", 1.0))), reverse=True)[:limit]
    return [{
        "name": str(r["name"]),
        "sim_minus_target": round(float(r["error"]), 6),
        "weight": round(float(r.get("weight", 1.0)), 6),
    } for r in ranked]


def _parse_index(text):
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.I | re.S)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[i:])
        except json.JSONDecodeError:
            continue
        value = obj.get("evaluation_index") if isinstance(obj, dict) else None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        value = obj.get("evaluation_index") if isinstance(obj, dict) else None
        if isinstance(value, int):
            return value
    return None


def _messages(model_key, baseline, remaining, revealed, previous_invalid=None):
    actions = [{
        "evaluation_index": idx,
        "persona_id": str(row["persona_id"]),
        "feature": str(row["feature"]),
        "signed_step": float(row["signed_step"]),
    } for idx, row in sorted(remaining.items())]
    history = [{
        "step": int(r["step"]),
        "evaluation_index": int(r["evaluation_index"]),
        "edit": f"{r['persona_id']}.{r['feature']}:{float(r['signed_step']):+.1f}",
        "calibration_objective": round(float(r["objective"]), 8),
        "improvement_vs_baseline": round(float(r["reward"]), 8),
        "largest_post_edit_residuals": r["largest_post_edit_residuals"],
    } for r in revealed]
    payload = {
        "simulator_model_key": model_key,
        "baseline_calibration_objective": round(float(baseline["objective"]), 8),
        "baseline_largest_residuals": _ranked_residuals(baseline, 12),
        "evaluated_history": history,
        "remaining_actions": actions,
    }
    if previous_invalid:
        payload["previous_response_was_invalid"] = previous_invalid[:1000]
        payload["repair_instruction"] = "Choose exactly one evaluation_index from remaining_actions."
    system = (
        "You are a sequential outer-loop optimizer for persona calibration. "
        "Choose exactly one unevaluated structured edit that is most useful to evaluate next. "
        "Use only the supplied calibration residuals and revealed evaluation history. "
        "Do not invent or combine edits, rewrite persona text, or use held-out information. "
        "Return only JSON with one integer key: evaluation_index."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}]


def _fallback_index(remaining, residual_path):
    frozen = json.loads(residual_path.read_text(encoding="utf-8"))
    for row in frozen["revealed_candidate_evaluations"]:
        idx = int(row["evaluation_index"])
        if idx in remaining:
            return idx
    return min(remaining)


def run_model(model_key, cfg, proposer, output_root):
    source_root = Path(cfg["experiment"]["source_output_root"])
    seed = int(cfg["experiment"]["calibration_seed"])
    seed_dir = source_root / model_key / "calibration" / f"seed_{seed}"
    rows = _history(seed_dir)
    baseline = rows[0]
    baseline_obj = float(baseline["objective"])
    remaining = {idx: rows[idx] for idx in range(1, 31)}
    revealed = []
    max_attempts = int(cfg["history_llm"]["max_parse_attempts_per_step"])
    n_steps = int(cfg["budget"]["candidate_evaluations_after_baseline"])
    proposer_calls = 0
    fallback_count = 0
    residual_path = output_root / model_key / "residual_linucb_selection.json"
    if not residual_path.exists():
        raise FileNotFoundError(f"Run residual LinUCB derivation first: {residual_path}")

    for step in range(1, n_steps + 1):
        chosen_idx = None
        attempts = []
        previous_invalid = None
        for attempt in range(1, max_attempts + 1):
            result = proposer.chat(_messages(model_key, baseline, remaining, revealed, previous_invalid), response_format={"type": "json_object"})
            proposer_calls += 1
            attempts.append({"attempt": attempt, "text": result.text, "prompt_hash": result.prompt_hash})
            parsed = _parse_index(result.text)
            if parsed in remaining:
                chosen_idx = int(parsed)
                break
            previous_invalid = result.text
        used_fallback = False
        if chosen_idx is None:
            chosen_idx = _fallback_index(remaining, residual_path)
            used_fallback = True
            fallback_count += 1
        chosen = remaining.pop(chosen_idx)
        reward = baseline_obj - float(chosen["objective"])
        revealed.append({
            "step": step,
            "evaluation_index": chosen_idx,
            "persona_id": str(chosen["persona_id"]),
            "feature": str(chosen["feature"]),
            "signed_step": float(chosen["signed_step"]),
            "objective": float(chosen["objective"]),
            "reward": float(reward),
            "largest_post_edit_residuals": _ranked_residuals(chosen, 6),
            "proposer_attempts": attempts,
            "used_fallback": used_fallback,
        })
        print(f"{model_key} step={step} eval={chosen_idx} obj={float(chosen['objective']):.6f} reward={reward:+.6f} fallback={used_fallback}", flush=True)

    eligible = [baseline] + [rows[int(r["evaluation_index"])] for r in revealed]
    selected = min(eligible, key=lambda r: (float(r["objective"]), int(r["evaluation_index"])))
    payload = {
        "model_key": model_key,
        "strategy": "history_llm",
        "proposer_model": cfg["history_llm"]["proposer_model"],
        "calibration_seed": seed,
        "logical_simulator_choice_query_budget": int(cfg["budget"]["logical_simulator_choice_queries"]),
        "proposer_calls": proposer_calls,
        "fallback_count": fallback_count,
        "baseline_objective": baseline_obj,
        "revealed_candidate_evaluations": revealed,
        "selected_evaluation_index": int(selected["evaluation_index"]),
        "selected_full_calibration_objective": float(selected["objective"]),
        "selected_personas": selected["personas"],
        "source_history_sha256": _sha256_file(seed_dir / "history.json"),
        "unrevealed_candidate_objectives_used_for_selection": False,
        "heldout_information_used": False,
    }
    model_out = output_root / model_key
    model_out.mkdir(parents=True, exist_ok=True)
    (model_out / "history_llm_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_adaptive_search.yaml")
    ap.add_argument("--output-root", default="outputs/cross_model_adaptive_outer_search_v1")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hc = cfg["history_llm"]
    proposer_cfg = {
        "model": hc["proposer_model"],
        "model_loader": hc["proposer_model_loader"],
        "dtype": hc["dtype"],
        "device_map": hc["device_map"],
        "max_new_tokens": hc["max_new_tokens"],
        "do_sample": hc["do_sample"],
        "temperature": hc["temperature"],
        "trust_remote_code": hc["trust_remote_code"],
        "cache_dir": hc["cache_dir"],
    }
    print(f"Loading fixed proposer: {proposer_cfg['model']}", flush=True)
    proposer = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(proposer_cfg))
    manifest = {
        "status": "history_llm_derived",
        "git_sha": _git_sha(),
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "proposer_runtime": proposer.runtime_info(),
        "fresh_heldout_evaluated": False,
        "models": {},
    }
    for model_key in cfg["experiment"]["model_keys"]:
        result = run_model(str(model_key), cfg, proposer, output_root)
        manifest["models"][str(model_key)] = {
            "selected_evaluation_index": result["selected_evaluation_index"],
            "selected_full_calibration_objective": result["selected_full_calibration_objective"],
            "proposer_calls": result["proposer_calls"],
            "fallback_count": result["fallback_count"],
        }
    (output_root / "history_llm_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
