#!/usr/bin/env python3
"""One-time frozen held-out evaluation for anchored-mu text personas.

For each calibration seed, load the already-selected calibrated persona prompts,
construct a fresh independently seeded held-out context ensemble, and evaluate
the original and calibrated populations on exactly the same held-out contexts.

No search, rewrite, coordinate update, or prompt selection occurs in this file.
Held-out metrics are reporting-only and must not be used for further tuning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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

from scripts.run_llm_prompt_refinement_calibration_v2 import (
    calibration_slates,
    initial_personas,
    residual_packet,
    simulated_moments,
    target_moments,
    target_probability_table,
    weighted_rmse,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _prompt_hash(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()


def _persona_payload(personas: tuple[TextPersona, ...]) -> list[dict]:
    return [
        {
            "persona_id": p.persona_id,
            "segment_label": p.segment_label,
            "prompt": p.prompt,
            "prompt_sha256": _prompt_hash(p.prompt),
        }
        for p in personas
    ]


def _reference_1729_personas() -> tuple[TextPersona, ...]:
    base = {p.persona_id: p for p in initial_personas()}
    quality = base["quality"]
    revised = TextPersona(
        persona_id=quality.persona_id,
        segment_label=quality.segment_label,
        prompt=(
            quality.prompt
            + " For this simulation, relative to that baseline description, "
            + "place slightly less weight on familiar brands. "
            + "Preserve all other preferences and tradeoffs from the baseline."
        ),
    )
    return (base["budget"], revised, base["sustain"])


def _load_selected_personas(
    *,
    calibration_seed: int,
    robustness_dir: Path,
    reference_seed: int,
) -> tuple[TextPersona, ...]:
    if calibration_seed == reference_seed:
        return _reference_1729_personas()

    path = robustness_dir / f"seed_{calibration_seed}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing frozen calibration result for seed {calibration_seed}: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("selected_personas")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"invalid selected_personas in {path}")

    personas = tuple(
        TextPersona(
            persona_id=str(row["persona_id"]),
            segment_label=str(row["segment_label"]),
            prompt=str(row["prompt"]),
        )
        for row in rows
    )
    ids = [p.persona_id for p in personas]
    if ids != ["budget", "quality", "sustain"]:
        raise ValueError(f"unexpected persona order in {path}: {ids}")
    return personas


class HeldoutEvaluator:
    def __init__(
        self,
        *,
        simulator: TextPersonaChoiceSimulator,
        slates: dict[str, pd.DataFrame],
        target: dict[str, float],
        output_dir: Path,
        condition: str,
    ) -> None:
        self.simulator = simulator
        self.slates = slates
        self.target = target
        self.output_dir = output_dir
        self.condition = condition

    def evaluate(self, personas: tuple[TextPersona, ...]) -> dict:
        frames = []
        for persona in personas:
            for context_id in sorted(self.slates):
                slate = self.slates[context_id]
                result = self.simulator.choose(slate=slate, persona=persona)
                frame = slate.copy().reset_index(drop=True)
                frame.insert(0, "observation_id", f"{persona.persona_id}__{context_id}")
                frame.insert(1, "dataset", "heldout_final")
                frame.insert(2, "condition", self.condition)
                frame.insert(3, "persona_id", persona.persona_id)
                frame.insert(4, "persona_segment_label", persona.segment_label)
                frame["chosen"] = frame["alternative_id"].astype(int).eq(result.alternative_id).astype(int)
                frame["llm_raw_text"] = result.raw_text
                frame["llm_prompt_hash"] = result.prompt_hash
                frame["llm_cached"] = bool(result.cached)
                frame["llm_latency_seconds"] = float(result.latency_seconds)
                frame["persona_prompt_hash"] = _prompt_hash(persona.prompt)
                frames.append(frame)

        choices = pd.concat(frames, ignore_index=True)
        simulated = simulated_moments(choices)
        residuals = residual_packet(self.target, simulated)
        objective = weighted_rmse(residuals)
        diagnostics = {
            "weighted_moment_rmse": objective,
            "n_choice_observations": int(choices["observation_id"].nunique()),
            "cache_hit_rate": float(choices["llm_cached"].astype(bool).mean()),
            "mean_latency_seconds": float(choices["llm_latency_seconds"].mean()),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        choices.to_csv(self.output_dir / "choices.csv", index=False)
        result = {
            "condition": self.condition,
            "objective": objective,
            "target": self.target,
            "simulated": simulated,
            "residuals": [asdict(r) | {"error": r.error} for r in residuals],
            "diagnostics": diagnostics,
            "personas": _persona_payload(personas),
        }
        (self.output_dir / "moments.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/anchored_mu_frozen_heldout_local_hf_gemma12.yaml",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/anchored_mu_frozen_heldout_local_hf_gemma12",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite existing final-test directory: {out}. "
            "Use a new output directory if a rerun is scientifically justified."
        )
    out.mkdir(parents=True, exist_ok=True)

    robustness_dir = Path(cfg["inputs"]["robustness_output_dir"])
    reference_seed = int(cfg["inputs"]["reference_seed"])
    calibration_seeds = [int(x) for x in cfg["experiment"]["calibration_seeds"]]
    heldout_map = {
        int(k): int(v)
        for k, v in cfg["experiment"]["heldout_seed_map"].items()
    }
    if set(calibration_seeds) != set(heldout_map):
        raise SystemExit("heldout seed map must exactly match frozen calibration seeds")

    # Load and fingerprint every frozen selected prompt before constructing heldout contexts.
    frozen_populations = {
        seed: _load_selected_personas(
            calibration_seed=seed,
            robustness_dir=robustness_dir,
            reference_seed=reference_seed,
        )
        for seed in calibration_seeds
    }
    frozen_prompt_manifest = {
        str(seed): _persona_payload(personas)
        for seed, personas in frozen_populations.items()
    }
    (out / "frozen_prompt_manifest.json").write_text(
        json.dumps(frozen_prompt_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    provenance = {
        "protocol": "frozen_text_persona_heldout_final",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": args.config,
        "model": cfg["backend"]["model"],
        "backend": "huggingface_local",
        "calibration_seeds": calibration_seeds,
        "heldout_seed_map": {str(k): heldout_map[k] for k in calibration_seeds},
        "optimization_performed": False,
        "prompt_selection_uses_heldout": False,
        "heldout_generated_after_prompt_freeze": True,
        "final_metrics_reporting_only": True,
    }
    (out / "final_test_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Loading {cfg['backend']['model']} once via Hugging Face Transformers...", flush=True)
    client = HuggingFaceLocalChatClient(HuggingFaceLocalConfig.from_config(cfg["backend"]))
    simulator = TextPersonaChoiceSimulator(client)

    original = initial_personas()
    paired_rows = []

    for calibration_seed in calibration_seeds:
        heldout_seed = heldout_map[calibration_seed]
        run_dir = out / f"calibration_seed_{calibration_seed}__heldout_seed_{heldout_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Heldout contexts and truth moments are generated only here, after prompts are frozen.
        slates = calibration_slates(
            heldout_seed,
            int(cfg["experiment"]["context_variants_per_intervention"]),
        )
        targets = target_probability_table(slates)
        moments = target_moments(targets)
        targets.to_csv(run_dir / "heldout_target_probabilities.csv", index=False)
        (run_dir / "heldout_target_moments.json").write_text(
            json.dumps(moments, indent=2, sort_keys=True), encoding="utf-8"
        )

        original_result = HeldoutEvaluator(
            simulator=simulator,
            slates=slates,
            target=moments,
            output_dir=run_dir / "original",
            condition="original",
        ).evaluate(original)

        calibrated = frozen_populations[calibration_seed]
        calibrated_result = HeldoutEvaluator(
            simulator=simulator,
            slates=slates,
            target=moments,
            output_dir=run_dir / "calibrated",
            condition="calibrated",
        ).evaluate(calibrated)

        original_obj = float(original_result["objective"])
        calibrated_obj = float(calibrated_result["objective"])
        improvement_fraction = (original_obj - calibrated_obj) / original_obj

        paired = {
            "calibration_seed": calibration_seed,
            "heldout_seed": heldout_seed,
            "original_objective": original_obj,
            "calibrated_objective": calibrated_obj,
            "improvement_fraction": improvement_fraction,
            "improved": bool(calibrated_obj < original_obj),
            "original_cache_hit_rate": original_result["diagnostics"]["cache_hit_rate"],
            "calibrated_cache_hit_rate": calibrated_result["diagnostics"]["cache_hit_rate"],
        }
        paired_rows.append(paired)
        (run_dir / "paired_summary.json").write_text(
            json.dumps(paired, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"seed {calibration_seed} -> heldout {heldout_seed}: "
            f"{original_obj:.6f} -> {calibrated_obj:.6f} "
            f"({improvement_fraction:+.2%})",
            flush=True,
        )

    improvements = [float(r["improvement_fraction"]) for r in paired_rows]
    original_objs = [float(r["original_objective"]) for r in paired_rows]
    calibrated_objs = [float(r["calibrated_objective"]) for r in paired_rows]

    aggregate = {
        "status": str(cfg["experiment"]["name"]) + "_complete",
        "protocol": "frozen_text_persona_heldout_final",
        "git_sha": _git_sha(),
        "model": cfg["backend"]["model"],
        "backend": "huggingface_local",
        "n_pairs": len(paired_rows),
        "paired_results": paired_rows,
        "mean_original_objective": statistics.fmean(original_objs),
        "mean_calibrated_objective": statistics.fmean(calibrated_objs),
        "mean_improvement_fraction": statistics.fmean(improvements),
        "median_improvement_fraction": statistics.median(improvements),
        "min_improvement_fraction": min(improvements),
        "max_improvement_fraction": max(improvements),
        "pairs_improved": sum(bool(r["improved"]) for r in paired_rows),
        "fraction_pairs_improved": sum(bool(r["improved"]) for r in paired_rows) / len(paired_rows),
        "optimization_performed": False,
        "selection_uses_heldout": False,
        "final_metrics_reporting_only": True,
        "warning": "Do not tune prompts, search settings, or seed choices using these held-out results.",
    }
    pd.DataFrame(paired_rows).to_csv(out / "paired_results.csv", index=False)
    (out / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
