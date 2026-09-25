"""Model-agnostic orchestration for residual-driven persona prompt refinement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from eipg.personas.prompt_refinement import EconomicResidual, ResidualPromptEditor
from eipg.simulators.llm_choice import TextPersona


@dataclass(frozen=True)
class PersonaEvaluation:
    residuals: list[EconomicResidual]
    objective: float
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class RefinementIteration:
    condition: str
    iteration: int
    personas: tuple[TextPersona, ...]
    objective: float
    diagnostics: dict[str, float]


EvaluatePersonas = Callable[[tuple[TextPersona, ...]], PersonaEvaluation]


class PromptRefinementLoop:
    """Run fixed-budget refinement without exposing held-out metrics to the editor.

    ``evaluate`` must use calibration data only. Held-out evaluation is intentionally
    absent from this class so it cannot accidentally enter selection or rewriting.
    """

    def __init__(
        self,
        *,
        editor: ResidualPromptEditor,
        evaluate: EvaluatePersonas,
        iterations: int = 3,
    ) -> None:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        self.editor = editor
        self.evaluate = evaluate
        self.iterations = int(iterations)

    def _rewrite(
        self,
        personas: tuple[TextPersona, ...],
        residuals: list[EconomicResidual],
        *,
        use_economic_signal: bool,
    ) -> tuple[TextPersona, ...]:
        revised: list[TextPersona] = []
        for persona in personas:
            result = self.editor.edit(
                current_prompt=persona.prompt,
                residuals=residuals,
                segment_label=persona.segment_label,
                use_economic_signal=use_economic_signal,
            )
            revised.append(
                TextPersona(
                    persona_id=persona.persona_id,
                    segment_label=persona.segment_label,
                    prompt=result.prompt,
                )
            )
        return tuple(revised)

    def run_condition(
        self,
        *,
        condition: str,
        initial_personas: tuple[TextPersona, ...],
    ) -> list[RefinementIteration]:
        if not initial_personas:
            raise ValueError("initial_personas must be non-empty")
        if condition not in {"original", "generic_rewrite_control", "economic_residual_rewrite"}:
            raise ValueError(f"unknown condition: {condition}")

        current = tuple(initial_personas)
        history: list[RefinementIteration] = []
        evaluation = self.evaluate(current)
        history.append(
            RefinementIteration(condition, 0, current, evaluation.objective, dict(evaluation.diagnostics))
        )

        if condition == "original":
            return history

        use_signal = condition == "economic_residual_rewrite"
        for iteration in range(1, self.iterations + 1):
            current = self._rewrite(current, evaluation.residuals, use_economic_signal=use_signal)
            evaluation = self.evaluate(current)
            history.append(
                RefinementIteration(
                    condition,
                    iteration,
                    current,
                    evaluation.objective,
                    dict(evaluation.diagnostics),
                )
            )
        return history

    def run_all(
        self,
        *,
        initial_personas: tuple[TextPersona, ...],
    ) -> dict[str, list[RefinementIteration]]:
        return {
            condition: self.run_condition(condition=condition, initial_personas=initial_personas)
            for condition in (
                "original",
                "generic_rewrite_control",
                "economic_residual_rewrite",
            )
        }
