#!/usr/bin/env python3
"""Pre-heldout LLM prompt-refinement calibration v2.

No held-out contexts are constructed or evaluated here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from eipg.econ.mnl import MNLConfig, MultinomialLogitModel
from eipg.experiments.prompt_refinement import PersonaEvaluation, PromptRefinementLoop
from eipg.personas.prompt_refinement import (
    EconomicResidual,
    ResidualPromptEditor,
    ResidualPromptEditorConfig,
)
from eipg.simulators.llm_choice import TextPersona, TextPersonaChoiceSimulator
from eipg.simulators.openai_compatible import OpenAICompatibleChatClient, OpenAICompatibleConfig

FEATURES = ("price", "quality", "sustain", "novelty", "brand")
SEGMENTS = ("budget_sensitive", "quality_oriented", "sustainability_oriented")
INTERVENTIONS = ("price", "quality", "sustain", "novelty")

TRUTH_BETA = {
    "budget_sensitive": np.array([-2.00, 0.70, 0.30, 0.20, 0.20], dtype=float),
    "quality_oriented": np.array([-0.80, 1.80, 0.40, 0.20, 0.80], dtype=float),
    "sustainability_oriented": np.array([-0.90, 0.60, 1.80, 0.90, 0.20], dtype=float),
}
MOMENT_WEIGHTS = {
    "price": 0.50,
    "quality": 1.00,
    "sustain": 1.00,
    "novelty": 1.00,
    "brand": 1.00,
    "response": 1.00,
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _provider_body(provider: str | None) -> dict | None:
    if not provider or provider.lower() in {"openai", "direct", "native", "local", "vllm", "lmstudio"}:
        return None
    return {"provider": {"order": [provider], "allow_fallbacks": False}}


def initial_personas() -> tuple[TextPersona, ...]:
    """Frozen deliberately under-grounded starting templates."""
    return (
        TextPersona(
            "budget",
            "budget_sensitive",
            (
                "A practical shopper who looks for good overall value. They compare price, quality, "
                "and familiar brands, and they are open to sustainability or novelty when those features "
                "make the offer more appealing."
            ),
        ),
        TextPersona(
            "quality",
            "quality_oriented",
            (
                "A balanced shopper who considers price, product quality, and brand familiarity together. "
                "They also notice sustainability and new features, but usually weigh several aspects before deciding."
            ),
        ),
        TextPersona(
            "sustain",
            "sustainability_oriented",
            (
                "An environmentally aware shopper who likes sustainable and innovative products, but usually "
                "balances those benefits against price, quality, and brand familiarity rather than paying a strong premium."
            ),
        ),
    )


def _base_templates() -> dict[str, np.ndarray]:
    return {
        "price": np.array([
            [1.20, 0.58, 0.42, 0.32, 0.20],
            [1.75, 0.72, 0.55, 0.46, 0.70],
            [2.20, 0.80, 0.62, 0.58, 0.45],
        ], dtype=float),
        "quality": np.array([
            [1.35, 0.48, 0.55, 0.40, 0.25],
            [1.70, 0.67, 0.52, 0.45, 0.65],
            [2.10, 0.82, 0.60, 0.52, 0.40],
        ], dtype=float),
        "sustain": np.array([
            [1.30, 0.68, 0.30, 0.42, 0.55],
            [1.75, 0.70, 0.57, 0.48, 0.45],
            [2.15, 0.74, 0.78, 0.55, 0.35],
        ], dtype=float),
        "novelty": np.array([
            [1.25, 0.66, 0.48, 0.22, 0.55],
            [1.70, 0.68, 0.53, 0.52, 0.45],
            [2.05, 0.72, 0.58, 0.76, 0.35],
        ], dtype=float),
    }


def calibration_slates(seed: int, variants: int) -> dict[str, pd.DataFrame]:
    """Create and freeze multiple nearby contexts per intervention direction."""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    templates = _base_templates()
    for intervention in INTERVENTIONS:
        template = templates[intervention]
        for v in range(variants):
            x = template.copy()
            # Frozen mild context heterogeneity. Price and non-price attributes use
            # different perturbation scales; alternative ordering remains fixed.
            x[:, 0] += rng.normal(0.0, 0.08, size=3)
            x[:, 1:] += rng.normal(0.0, 0.035, size=(3, 4))
            x[:, 0] = np.clip(x[:, 0], 0.60, 3.00)
            x[:, 1:] = np.clip(x[:, 1:], 0.0, 1.0)
            base_id = f"{intervention}_v{v:02d}_base"
            int_id = f"{intervention}_v{v:02d}_intervention"
            base = pd.DataFrame(x, columns=FEATURES)
            base.insert(0, "alternative_id", [0, 1, 2])
            base.insert(0, "context_id", base_id)
            changed = base.copy()
            changed["context_id"] = int_id
            mask = changed["alternative_id"].astype(int).eq(1)
            if intervention == "price":
                changed.loc[mask, "price"] *= 1.25
            else:
                changed.loc[mask, intervention] = np.minimum(
                    1.0, changed.loc[mask, intervention].astype(float) + 0.25
                )
            out[base_id] = base
            out[int_id] = changed
    return out


def _softmax(values: np.ndarray) -> np.ndarray:
    z = values - np.max(values)
    e = np.exp(z)
    return e / e.sum()


def target_probability_table(slates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict] = []
    for segment in SEGMENTS:
        beta = TRUTH_BETA[segment]
        for context_id, slate in slates.items():
            x = slate[list(FEATURES)].to_numpy(dtype=float)
            probs = _softmax(x @ beta)
            for (_, row), prob in zip(slate.iterrows(), probs):
                rec = {
                    "segment_label": segment,
                    "context_id": context_id,
                    "alternative_id": int(row["alternative_id"]),
                    "target_probability": float(prob),
                }
                for feature in FEATURES:
                    rec[feature] = float(row[feature])
                records.append(rec)
    return pd.DataFrame(records)


def _is_context(series: pd.Series, intervention: str, suffix: str) -> pd.Series:
    return series.astype(str).str.startswith(f"{intervention}_v") & series.astype(str).str.endswith(suffix)


def target_moments(target_probs: pd.DataFrame) -> dict[str, float]:
    moments: dict[str, float] = {}
    for segment in SEGMENTS:
        g = target_probs[target_probs["segment_label"] == segment]
        n_contexts = g["context_id"].nunique()
        for feature in FEATURES:
            moments[f"{segment}.mean_chosen_{feature}"] = float(
                (g["target_probability"] * g[feature]).sum() / n_contexts
            )
    for intervention in INTERVENTIONS:
        alt1 = target_probs[target_probs["alternative_id"].astype(int).eq(1)]
        base = alt1[_is_context(alt1["context_id"], intervention, "_base")]["target_probability"].mean()
        changed = alt1[_is_context(alt1["context_id"], intervention, "_intervention")]["target_probability"].mean()
        moments[f"response.{intervention}.alt1_share_change"] = float(changed - base)
    return moments


def simulated_moments(choices: pd.DataFrame) -> dict[str, float]:
    chosen = choices[choices["chosen"].astype(int).eq(1)].copy()
    moments: dict[str, float] = {}
    for segment in SEGMENTS:
        g = chosen[chosen["persona_segment_label"] == segment]
        if g.empty:
            raise ValueError(f"no chosen rows for segment {segment}")
        for feature in FEATURES:
            moments[f"{segment}.mean_chosen_{feature}"] = float(g[feature].mean())
    for intervention in INTERVENTIONS:
        base = chosen[_is_context(chosen["context_id"], intervention, "_base")]["alternative_id"].astype(int).eq(1).mean()
        changed = chosen[_is_context(chosen["context_id"], intervention, "_intervention")]["alternative_id"].astype(int).eq(1).mean()
        moments[f"response.{intervention}.alt1_share_change"] = float(changed - base)
    return moments


def residual_packet(target: dict[str, float], simulated: dict[str, float]) -> list[EconomicResidual]:
    out = []
    for name, target_value in target.items():
        if name.startswith("response."):
            weight = MOMENT_WEIGHTS["response"]
        else:
            weight = MOMENT_WEIGHTS[name.rsplit("_", 1)[-1]]
        out.append(EconomicResidual(name, float(target_value), float(simulated[name]), float(weight)))
    return out


def weighted_rmse(residuals: list[EconomicResidual]) -> float:
    x = np.asarray([r.error * r.weight for r in residuals], dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def _pairs(personas: tuple[TextPersona, ...], slates: dict[str, pd.DataFrame]):
    return [
        (f"{p.persona_id}__{cid}", p, slates[cid])
        for p in personas
        for cid in sorted(slates)
    ]


def _append_progress(output_dir: Path, event: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)


class CalibrationEvaluator:
    def __init__(self, simulator, slates, target, output_dir):
        self.simulator = simulator
        self.slates = slates
        self.target = target
        self.output_dir = output_dir
        self.calls = 0

    def __call__(self, personas: tuple[TextPersona, ...]) -> PersonaEvaluation:
        eval_id = self.calls
        self.calls += 1
        pairs = _pairs(personas, self.slates)
        frames = []
        _append_progress(self.output_dir, {"event": "evaluation_started", "evaluation_id": eval_id, "pairs_total": len(pairs)})
        for pair_index, (observation_id, persona, slate) in enumerate(pairs, start=1):
            result = self.simulator.choose(slate=slate, persona=persona)
            frame = slate.copy().reset_index(drop=True)
            frame.insert(0, "observation_id", observation_id)
            frame.insert(1, "dataset", "calibration_v2")
            frame.insert(2, "persona_id", persona.persona_id)
            frame.insert(3, "persona_segment_label", persona.segment_label)
            frame["chosen"] = frame["alternative_id"].astype(int).eq(result.alternative_id).astype(int)
            frame["llm_raw_text"] = result.raw_text
            frame["llm_prompt_hash"] = result.prompt_hash
            frame["llm_cached"] = result.cached
            frame["llm_latency_seconds"] = result.latency_seconds
            frame["persona_prompt_hash"] = sha256(persona.prompt.encode("utf-8")).hexdigest()
            frames.append(frame)
            if pair_index % 10 == 0 or pair_index == len(pairs):
                _append_progress(self.output_dir, {
                    "event": "choice_progress", "evaluation_id": eval_id,
                    "pair_index": pair_index, "pairs_total": len(pairs),
                    "cached": bool(result.cached), "latency_seconds": float(result.latency_seconds),
                })
        choices = pd.concat(frames, ignore_index=True)
        simulated = simulated_moments(choices)
        residuals = residual_packet(self.target, simulated)
        objective = weighted_rmse(residuals)
        fit = MultinomialLogitModel(MNLConfig(features=FEATURES, l2=1e-3, max_iter=300)).fit(choices)
        diagnostics = {
            "weighted_moment_rmse": objective,
            "n_choice_observations": float(choices["observation_id"].nunique()),
            "mean_latency_seconds": float(choices["llm_latency_seconds"].mean()),
            "cache_hit_rate": float(choices["llm_cached"].astype(bool).mean()),
            **{f"mnl_beta_{k}": float(v) for k, v in fit.beta_by_feature.items()},
        }
        d = self.output_dir / "evaluations" / f"eval_{eval_id:02d}"
        d.mkdir(parents=True, exist_ok=True)
        choices.to_csv(d / "choices.csv", index=False)
        (d / "moments.json").write_text(json.dumps({
            "target": self.target,
            "simulated": simulated,
            "residuals": [asdict(r) | {"error": r.error} for r in residuals],
            "objective": objective,
            "diagnostics": diagnostics,
            "persona_prompts": {p.persona_id: p.prompt for p in personas},
        }, indent=2, sort_keys=True), encoding="utf-8")
        _append_progress(self.output_dir, {"event": "evaluation_completed", "evaluation_id": eval_id, "objective": objective})
        return PersonaEvaluation(residuals=residuals, objective=objective, diagnostics=diagnostics)


def _client_config(base: dict, *, model: str, provider: str, cache_dir: Path) -> OpenAICompatibleConfig:
    section = dict(base)
    for k in ("models", "providers", "provider_routing"):
        section.pop(k, None)
    section["model"] = model
    section["cache_dir"] = str(cache_dir)
    extra_body = dict(section.get("extra_body", {}) or {})
    provider_body = _provider_body(provider)
    if provider_body:
        extra_body.update(provider_body)
    section["extra_body"] = extra_body or None
    return OpenAICompatibleConfig.from_config(section)


def _selected(history):
    return min(history, key=lambda x: (x.objective, x.iteration))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_prompt_refinement_v2.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model not in cfg["backend"]["models"]:
        raise SystemExit("model is not in frozen v2 model set")
    if cfg["backend"]["providers"].get(args.model) != args.provider:
        raise SystemExit("provider does not match frozen v2 provider")

    out = Path(args.output_dir or f"outputs/llm_prompt_refinement_v2/{_slug(args.model)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").unlink(missing_ok=True)
    _append_progress(out, {"event": "run_started", "model": args.model, "provider": args.provider})

    cache = out / "cache"
    sim_client = OpenAICompatibleChatClient(_client_config(cfg["backend"], model=args.model, provider=args.provider, cache_dir=cache/"simulator"))
    simulator = TextPersonaChoiceSimulator(sim_client)
    ecfg = cfg["editor"]
    editor_client = OpenAICompatibleChatClient(_client_config(cfg["backend"], model=ecfg["model"], provider=ecfg["provider"], cache_dir=cache/"editor"))
    editor = ResidualPromptEditor(editor_client, ResidualPromptEditorConfig(max_prompt_chars=int(ecfg["max_prompt_chars"]), max_residuals=int(ecfg["max_residuals"])))

    slates = calibration_slates(int(cfg["experiment"]["controlled_seed"]), int(cfg["experiment"]["context_variants_per_intervention"]))
    targets = target_probability_table(slates)
    moments = target_moments(targets)
    targets.to_csv(out/"synthetic_human_calibration_targets.csv", index=False)
    (out/"target_moments.json").write_text(json.dumps(moments, indent=2, sort_keys=True), encoding="utf-8")

    evaluator = CalibrationEvaluator(simulator, slates, moments, out)
    loop = PromptRefinementLoop(editor=editor, evaluate=evaluator, iterations=int(cfg["experiment"]["iterations"]))
    histories = loop.run_all(initial_personas=initial_personas())

    history_payload = {}
    selected_payload = {}
    for condition, history in histories.items():
        history_payload[condition] = [{
            "iteration": item.iteration,
            "objective": item.objective,
            "diagnostics": item.diagnostics,
            "personas": [{"persona_id": p.persona_id, "segment_label": p.segment_label, "prompt": p.prompt,
                           "prompt_sha256": sha256(p.prompt.encode("utf-8")).hexdigest()} for p in item.personas],
        } for item in history]
        best = _selected(history)
        selected_payload[condition] = {
            "iteration": best.iteration,
            "objective": best.objective,
            "personas": [{"persona_id": p.persona_id, "segment_label": p.segment_label, "prompt": p.prompt,
                           "prompt_sha256": sha256(p.prompt.encode("utf-8")).hexdigest()} for p in best.personas],
        }

    initial = histories["original"][0].objective
    summary = {
        "status": str(cfg["experiment"]["name"]) + "_complete",
        "heldout_constructed": False,
        "simulator_model": args.model,
        "simulator_provider": args.provider,
        "editor_model": ecfg["model"],
        "editor_provider": ecfg["provider"],
        "iterations": int(cfg["experiment"]["iterations"]),
        "context_variants_per_intervention": int(cfg["experiment"]["context_variants_per_intervention"]),
        "n_calibration_choice_observations_per_evaluation": len(_pairs(initial_personas(), slates)),
        "initial_objective": initial,
        "selected": selected_payload,
        "economic_selected_improvement_fraction": (initial-selected_payload["economic_residual_rewrite"]["objective"])/initial,
        "generic_selected_improvement_fraction": (initial-selected_payload["generic_rewrite_control"]["objective"])/initial,
        "evaluation_calls": evaluator.calls,
    }
    (out/"history.json").write_text(json.dumps(history_payload, indent=2, sort_keys=True), encoding="utf-8")
    (out/"selected_prompts.json").write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out/"config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _append_progress(out, {"event": "run_completed", "summary": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
