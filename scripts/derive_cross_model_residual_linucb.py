#!/usr/bin/env python3
"""Derive the frozen residual-prior LinUCB policy by replaying calibration artifacts."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERSONAS = ("budget", "quality", "sustain")
PERSONA_TO_SEGMENT = {
    "budget": "budget_sensitive",
    "quality": "quality_oriented",
    "sustain": "sustainability_oriented",
}
FEATURES = ("price", "quality", "sustain", "novelty", "brand")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _history(seed_dir: Path) -> dict[int, dict]:
    rows = json.loads((seed_dir / "history.json").read_text(encoding="utf-8"))
    out = {
        int(row["evaluation_index"]): row
        for row in rows
        if "evaluation_index" in row
    }
    if set(out) != set(range(31)):
        raise RuntimeError(f"Expected eval indices 0..30, got {sorted(out)}")
    return out


def _residual_map(row: dict) -> dict[str, dict]:
    return {str(r["name"]): r for r in row["residuals"]}


def _desired_sign(feature: str, error: float) -> int:
    if abs(error) < 1e-15:
        return 0
    raw = 1 if error > 0 else -1
    return raw if feature == "price" else -raw


def _prior_score(action: dict, baseline: dict) -> float:
    residuals = _residual_map(baseline)
    persona = str(action["persona_id"])
    feature = str(action["feature"])
    direction = 1 if float(action["signed_step"]) > 0 else -1
    segment = PERSONA_TO_SEGMENT[persona]

    total = 0.0
    name = f"{segment}.mean_chosen_{feature}"
    if name in residuals:
        r = residuals[name]
        err = float(r["error"])
        mag = abs(err) * abs(float(r.get("weight", 1.0)))
        desired = _desired_sign(feature, err)
        total += mag if desired == direction else -mag

    response_name = f"response.{feature}.alt1_share_change"
    if response_name in residuals:
        r = residuals[response_name]
        err = float(r["error"])
        mag = abs(err) * abs(float(r.get("weight", 1.0)))
        desired = _desired_sign(feature, err)
        total += mag if desired == direction else -mag

    return float(total)


def _action_features(action: dict) -> np.ndarray:
    persona = str(action["persona_id"])
    feature = str(action["feature"])
    direction = 1.0 if float(action["signed_step"]) > 0 else -1.0
    x = np.zeros(len(PERSONAS) + len(FEATURES) + 1, dtype=float)
    x[PERSONAS.index(persona)] = 1.0
    x[len(PERSONAS) + FEATURES.index(feature)] = 1.0
    x[-1] = direction
    return x


def derive_for_model(*, model_key: str, cfg: dict, output_root: Path) -> dict:
    source_root = Path(cfg["experiment"]["source_output_root"])
    seed = int(cfg["experiment"]["calibration_seed"])
    seed_dir = source_root / model_key / "calibration" / f"seed_{seed}"
    rows = _history(seed_dir)
    baseline = rows[0]
    baseline_obj = float(baseline["objective"])

    actions = [rows[i] for i in range(1, 31)]
    raw_priors = {int(a["evaluation_index"]): _prior_score(a, baseline) for a in actions}
    scale = max(1e-12, max(abs(v) for v in raw_priors.values()))
    priors = {idx: val / scale for idx, val in raw_priors.items()}

    policy = cfg["residual_linucb"]
    lam = float(policy["ridge_lambda"])
    beta = float(policy["exploration_beta"])
    prior_weight = float(policy["residual_prior_weight"])
    n_steps = int(cfg["budget"]["candidate_evaluations_after_baseline"])

    dim = len(_action_features(actions[0]))
    A = lam * np.eye(dim, dtype=float)
    b = np.zeros(dim, dtype=float)
    remaining = {int(a["evaluation_index"]): a for a in actions}
    revealed: list[dict[str, Any]] = []

    for step in range(1, n_steps + 1):
        invA = np.linalg.inv(A)
        theta = invA @ b
        scored = []
        for idx, action in remaining.items():
            x = _action_features(action)
            predicted = float(x @ theta)
            uncertainty = float(np.sqrt(max(0.0, x @ invA @ x)))
            score = (
                prior_weight * float(priors[idx])
                + predicted
                + beta * uncertainty
            )
            scored.append((score, -idx, idx, predicted, uncertainty))
        scored.sort(reverse=True)
        _, _, chosen_idx, predicted, uncertainty = scored[0]
        chosen = remaining.pop(chosen_idx)
        reward = baseline_obj - float(chosen["objective"])
        x = _action_features(chosen)
        A = A + np.outer(x, x)
        b = b + x * reward
        revealed.append({
            "step": step,
            "evaluation_index": chosen_idx,
            "persona_id": chosen["persona_id"],
            "feature": chosen["feature"],
            "signed_step": float(chosen["signed_step"]),
            "objective": float(chosen["objective"]),
            "reward": float(reward),
            "normalized_residual_prior": float(priors[chosen_idx]),
            "predicted_reward_before_reveal": predicted,
            "uncertainty_before_reveal": uncertainty,
            "residuals": chosen["residuals"],
        })

    eligible = [baseline] + [rows[int(x["evaluation_index"])] for x in revealed]
    selected = min(
        eligible,
        key=lambda row: (float(row["objective"]), int(row["evaluation_index"])),
    )
    result = {
        "model_key": model_key,
        "strategy": "residual_linucb",
        "calibration_seed": seed,
        "logical_simulator_choice_query_budget": int(
            cfg["budget"]["logical_simulator_choice_queries"]
        ),
        "baseline_objective": baseline_obj,
        "revealed_candidate_evaluations": revealed,
        "selected_evaluation_index": int(selected["evaluation_index"]),
        "selected_full_calibration_objective": float(selected["objective"]),
        "selected_personas": selected["personas"],
        "source_history_sha256": _sha256_file(seed_dir / "history.json"),
        "policy_parameters": {
            "ridge_lambda": lam,
            "exploration_beta": beta,
            "residual_prior_weight": prior_weight,
        },
        "unrevealed_candidate_objectives_used_for_selection": False,
    }

    model_out = output_root / model_key
    model_out.mkdir(parents=True, exist_ok=True)
    path = model_out / "residual_linucb_selection.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_adaptive_search.yaml")
    ap.add_argument(
        "--output-root",
        default="outputs/cross_model_adaptive_outer_search_v1",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "residual_linucb_derived",
        "git_sha": _git_sha(),
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "fresh_heldout_evaluated": False,
        "models": {},
    }
    for model_key in cfg["experiment"]["model_keys"]:
        result = derive_for_model(
            model_key=str(model_key),
            cfg=cfg,
            output_root=output_root,
        )
        manifest["models"][str(model_key)] = {
            "selected_evaluation_index": result["selected_evaluation_index"],
            "selected_full_calibration_objective": result[
                "selected_full_calibration_objective"
            ],
        }
        print(
            model_key,
            "selected_eval=",
            result["selected_evaluation_index"],
            "objective=",
            result["selected_full_calibration_objective"],
            flush=True,
        )

    (output_root / "residual_linucb_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
