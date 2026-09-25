#!/usr/bin/env python3
"""Freeze GEPA checkpoints across all models before fifth heldout evaluation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import yaml


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross_model_gepa_anchored_text.yaml")
    ap.add_argument("--output-root", default="outputs/cross_model_gepa_anchored_text_v1")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(args.output_root)
    freeze_path = root / "gepa_cohort_freeze.json"
    if freeze_path.exists():
        raise SystemExit(f"Refusing to overwrite freeze: {freeze_path}")

    entries = {}
    for model_key in cfg["experiment"]["model_keys"]:
        model_key = str(model_key)
        model_dir = root / model_key
        manifest_path = model_dir / "calibration_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_entries = {}
        for checkpoint in cfg["gepa"]["budget_checkpoints"]:
            label = str(checkpoint["label"])
            path = model_dir / f"{label}_selection.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bool(payload.get("heldout_evaluated")):
                raise RuntimeError(f"Heldout already marked evaluated in {path}")
            checkpoint_entries[label] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "selected_candidate_discovery_calls": int(payload["selected_candidate_discovery_calls"]),
                "logical_choice_query_budget": int(payload["logical_choice_query_budget"]),
                "best_score": float(payload["gepa_best_score_within_budget"]),
                "candidate_suffixes": payload["candidate_suffixes"],
                "selected_prompt_hashes": {
                    row["persona_id"]: row["prompt_sha256"]
                    for row in payload["selected_personas"]
                },
            }
        entries[model_key] = {
            "calibration_manifest_sha256": _sha256_file(manifest_path),
            "checkpoints": checkpoint_entries,
        }

    root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "status": "gepa_cohort_frozen_before_fifth_heldout",
        "git_sha": _git_sha(),
        "protocol": cfg["experiment"]["name"],
        "config": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "gepa_version": str(cfg["gepa"]["package_version"]),
        "fresh_heldout_seeds_reserved_but_not_evaluated": [
            int(x) for x in cfg["fresh_heldout"]["seeds"]
        ],
        "models": entries,
        "heldout_evaluated": False,
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
