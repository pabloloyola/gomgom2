"""GEPA adapter for anchored natural-language persona calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from eipg.personas.prompt_refinement import EconomicResidual, ResidualPromptEditor
from eipg.simulators.llm_choice import TextPersona
from scripts.run_llm_prompt_refinement_calibration_v2 import (
    INTERVENTIONS,
    SEGMENTS,
    _pairs,
    calibration_slates,
    initial_personas,
    residual_packet,
    simulated_moments,
    target_moments,
    target_probability_table,
    weighted_rmse,
)


COMPONENT_TO_PERSONA = {
    "budget_suffix": ("budget", "budget_sensitive"),
    "quality_suffix": ("quality", "quality_oriented"),
    "sustain_suffix": ("sustain", "sustainability_oriented"),
}
SEGMENT_TO_COMPONENT = {
    "budget_sensitive": "budget_suffix",
    "quality_oriented": "quality_suffix",
    "sustainability_oriented": "sustain_suffix",
}


@dataclass(frozen=True)
class GEPATrace:
    example_label: str
    objective: float
    residuals: list[dict[str, Any]]
    block_scores: dict[str, float]
    candidate_suffixes: dict[str, str]


def _variant_slates(all_slates: dict[str, pd.DataFrame], variant: int) -> dict[str, pd.DataFrame]:
    marker = f"_v{variant:02d}_"
    return {k: v for k, v in all_slates.items() if marker in str(k)}


def _block_scores(residuals: list[EconomicResidual]) -> dict[str, float]:
    groups: dict[str, list[EconomicResidual]] = {
        "budget_segment": [],
        "quality_segment": [],
        "sustain_segment": [],
        "intervention_response": [],
    }
    for r in residuals:
        if r.name.startswith("budget_sensitive."):
            groups["budget_segment"].append(r)
        elif r.name.startswith("quality_oriented."):
            groups["quality_segment"].append(r)
        elif r.name.startswith("sustainability_oriented."):
            groups["sustain_segment"].append(r)
        elif r.name.startswith("response."):
            groups["intervention_response"].append(r)
    out = {}
    for name, rows in groups.items():
        if not rows:
            out[name] = 0.0
            continue
        x = np.asarray([r.error * r.weight for r in rows], dtype=float)
        out[name] = -float(np.sqrt(np.mean(x ** 2)))
    return out


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text))


class AnchoredSuffixRenderer:
    def __init__(
        self,
        *,
        max_suffix_chars: int,
        forbidden_terms: list[str],
        no_numeric_target_copying: bool,
        target_values: list[float],
    ) -> None:
        self.max_suffix_chars = int(max_suffix_chars)
        self.forbidden_terms = tuple(x.lower() for x in forbidden_terms)
        self.no_numeric_target_copying = bool(no_numeric_target_copying)
        self.target_numeric_tokens = set()
        for value in target_values:
            for rendered in (f"{value}", f"{value:.3f}", f"{value:.4f}", f"{value:.6f}"):
                self.target_numeric_tokens |= _numeric_tokens(rendered)

    def validate(self, candidate: dict[str, str]) -> tuple[bool, str | None]:
        expected = set(COMPONENT_TO_PERSONA)
        if set(candidate) != expected:
            return False, f"candidate components must be exactly {sorted(expected)}"
        for component, suffix in candidate.items():
            if not isinstance(suffix, str):
                return False, f"{component} must be text"
            clean = suffix.strip()
            if len(clean) > self.max_suffix_chars:
                return False, f"{component} exceeds max suffix length"
            lowered = clean.lower()
            bad = [term for term in self.forbidden_terms if term in lowered]
            if bad:
                return False, f"{component} contains forbidden terms: {bad}"
            if self.no_numeric_target_copying and clean:
                overlap = _numeric_tokens(clean) & self.target_numeric_tokens
                if overlap:
                    return False, f"{component} copies numeric calibration values: {sorted(overlap)}"
        return True, None

    def render(self, candidate: dict[str, str]) -> tuple[TextPersona, ...]:
        ok, error = self.validate(candidate)
        if not ok:
            raise ValueError(error)
        base = {p.persona_id: p for p in initial_personas()}
        out = []
        for component in ("budget_suffix", "quality_suffix", "sustain_suffix"):
            persona_id, segment = COMPONENT_TO_PERSONA[component]
            suffix = candidate[component].strip()
            base_prompt = base[persona_id].prompt
            if not suffix:
                prompt = base_prompt
            else:
                prompt = (
                    base_prompt
                    + " For this simulation, relative to that baseline description, "
                    + suffix
                    + " Preserve all other preferences and tradeoffs from the baseline."
                )
            out.append(TextPersona(persona_id, segment, prompt))
        return tuple(out)


class LocalHFReflectionLM:
    """Minimal GEPA LanguageModel wrapper around the existing local HF client."""

    def __init__(self, client) -> None:
        self.client = client
        self.calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self.calls += 1
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        result = self.client.chat(messages)
        return result.text


class EIPGGEPAAdapter:
    """Custom GEPA adapter using the established EIPG behavioral objective."""

    # GEPA 0.1.4 accesses this optional protocol attribute directly rather than
    # through getattr(). Explicit None selects GEPA's built-in reflection LM.
    propose_new_texts = None

    def __init__(
        self,
        *,
        simulator,
        calibration_seed: int,
        variants: int,
        output_dir: Path,
        reflection_cfg: dict[str, Any],
    ) -> None:
        self.simulator = simulator
        self.calibration_seed = int(calibration_seed)
        self.variants = int(variants)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_slates = calibration_slates(self.calibration_seed, self.variants)
        self.full_target = target_moments(target_probability_table(self.all_slates))
        self.variant_slates = {
            i: _variant_slates(self.all_slates, i) for i in range(self.variants)
        }
        self.variant_targets = {
            i: target_moments(target_probability_table(self.variant_slates[i]))
            for i in range(self.variants)
        }
        self.renderer = AnchoredSuffixRenderer(
            max_suffix_chars=int(reflection_cfg["max_suffix_chars"]),
            forbidden_terms=list(reflection_cfg["forbidden_terms"]),
            no_numeric_target_copying=bool(reflection_cfg["no_numeric_target_copying"]),
            target_values=list(self.full_target.values()),
        )
        self.evaluation_counter = 0
        self.logical_calls = 0
        self.candidate_discovery_calls: dict[int, int] = {}

    def get_adapter_state(self) -> dict[str, Any]:
        """Persist budget/discovery accounting across GEPA checkpoint resumes."""
        return {
            "evaluation_counter": int(self.evaluation_counter),
            "logical_calls": int(self.logical_calls),
            "candidate_discovery_calls": {
                str(k): int(v) for k, v in self.candidate_discovery_calls.items()
            },
        }

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        """Restore budget/discovery accounting from a GEPA checkpoint."""
        self.evaluation_counter = int(state.get("evaluation_counter", 0))
        self.logical_calls = int(state.get("logical_calls", 0))
        restored = {
            int(k): int(v)
            for k, v in dict(state.get("candidate_discovery_calls", {})).items()
        }
        self.candidate_discovery_calls.clear()
        self.candidate_discovery_calls.update(restored)

    def _evaluate_one(self, item: dict[str, Any], candidate: dict[str, str]):
        kind = str(item["kind"])
        if kind == "full":
            slates = self.all_slates
            target = self.full_target
            logical_calls = 120
            label = "full"
        elif kind == "variant":
            variant = int(item["variant"])
            slates = self.variant_slates[variant]
            target = self.variant_targets[variant]
            logical_calls = 24
            label = f"variant_{variant}"
        else:
            raise ValueError(f"Unknown GEPA data item: {item}")

        personas = self.renderer.render(candidate)
        frames = []
        for observation_id, persona, slate in _pairs(personas, slates):
            result = self.simulator.choose(slate=slate, persona=persona)
            frame = slate.copy().reset_index(drop=True)
            frame.insert(0, "observation_id", observation_id)
            frame.insert(1, "dataset", f"gepa_{label}")
            frame.insert(2, "persona_id", persona.persona_id)
            frame.insert(3, "persona_segment_label", persona.segment_label)
            frame["chosen"] = frame["alternative_id"].astype(int).eq(result.alternative_id).astype(int)
            frame["llm_raw_text"] = result.raw_text
            frame["llm_prompt_hash"] = result.prompt_hash
            frame["llm_cached"] = result.cached
            frame["llm_latency_seconds"] = result.latency_seconds
            frame["persona_prompt_hash"] = sha256(persona.prompt.encode("utf-8")).hexdigest()
            frames.append(frame)
        choices = pd.concat(frames, ignore_index=True)
        simulated = simulated_moments(choices)
        residuals = residual_packet(target, simulated)
        objective = weighted_rmse(residuals)
        blocks = _block_scores(residuals)
        trace = GEPATrace(
            example_label=label,
            objective=float(objective),
            residuals=[asdict(r) | {"error": r.error} for r in residuals],
            block_scores=blocks,
            candidate_suffixes=dict(candidate),
        )
        output = {
            "example_label": label,
            "objective": float(objective),
            "block_scores": blocks,
        }
        return output, -float(objective), trace, blocks, logical_calls

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        ok, validation_error = self.renderer.validate(candidate)
        if not ok:
            outputs = [
                {"example_label": str(item), "invalid_candidate": validation_error}
                for item in batch
            ]
            scores = [-10.0 for _ in batch]
            traces = None
            if capture_traces:
                traces = [
                    GEPATrace(
                        example_label=str(item),
                        objective=10.0,
                        residuals=[],
                        block_scores={
                            "budget_segment": -10.0,
                            "quality_segment": -10.0,
                            "sustain_segment": -10.0,
                            "intervention_response": -10.0,
                        },
                        candidate_suffixes=dict(candidate),
                    )
                    for item in batch
                ]
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=traces,
                objective_scores=[
                    {
                        "budget_segment": -10.0,
                        "quality_segment": -10.0,
                        "sustain_segment": -10.0,
                        "intervention_response": -10.0,
                    }
                    for _ in batch
                ],
                num_metric_calls=0,
            )

        outputs, scores, traces, objective_scores = [], [], [], []
        logical_calls = 0
        for item in batch:
            output, score, trace, blocks, calls = self._evaluate_one(item, candidate)
            outputs.append(output)
            scores.append(score)
            objective_scores.append(blocks)
            logical_calls += calls
            if capture_traces:
                traces.append(trace)

        self.logical_calls += logical_calls
        self.evaluation_counter += 1
        record = {
            "evaluation_counter": self.evaluation_counter,
            "candidate": dict(candidate),
            "batch": batch,
            "scores": scores,
            "outputs": outputs,
            "logical_simulator_choice_queries": logical_calls,
            "cumulative_logical_simulator_choice_queries": self.logical_calls,
        }
        with (self.output_dir / "evaluation_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
            objective_scores=objective_scores,
            num_metric_calls=logical_calls,
        )

    @staticmethod
    def _feedback_for_component(trace: GEPATrace, component: str) -> str:
        _persona_id, segment = COMPONENT_TO_PERSONA[component]
        residuals = []
        for row in trace.residuals:
            name = str(row["name"])
            if name.startswith(segment + ".") or name.startswith("response."):
                residuals.append(row)
        residuals.sort(
            key=lambda r: abs(float(r["error"])) * abs(float(r.get("weight", 1.0))),
            reverse=True,
        )
        lines = [
            f"Behavioral evaluation for {trace.example_label}.",
            "Use these diagnostics only to revise the behavioral suffix; do not mention the diagnostics in the persona text.",
        ]
        for row in residuals[:8]:
            r = EconomicResidual(
                name=str(row["name"]),
                target=float(row["target"]),
                simulated=float(row["simulated"]),
                weight=float(row.get("weight", 1.0)),
            )
            magnitude = abs(r.error * r.weight)
            if magnitude >= 0.12:
                level = "large"
            elif magnitude >= 0.05:
                level = "moderate"
            else:
                level = "small"
            guidance = ResidualPromptEditor._guidance_for_residual(r)
            lines.append(f"- {level} discrepancy in {r.name}: {guidance}.")
        lines.append(
            "Constraint: generalize the correction across product contexts; preserve unrelated tradeoffs and never copy numeric targets."
        )
        return "\n".join(lines)

    def make_reflective_dataset(
        self,
        candidate,
        eval_batch,
        components_to_update,
    ):
        traces = eval_batch.trajectories or []
        result = {}
        for component in components_to_update:
            rows = []
            for trace in traces:
                rows.append(
                    {
                        "Inputs": {
                            "persona_component": component,
                            "evaluation_context": trace.example_label,
                            "current_suffix": candidate[component],
                        },
                        "Generated Outputs": {
                            "current_suffix": candidate[component],
                        },
                        "Feedback": self._feedback_for_component(trace, component),
                    }
                )
            result[component] = rows
        return result
