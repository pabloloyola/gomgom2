"""Structured behavioral coordinates and deterministic persona rendering.

The optimizer operates on a small latent preference vector mu.  Natural-language
persona prompts are a deterministic rendering of mu, so the LLM is used only as
the behavioral simulator in this experiment, not as the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from eipg.personas.prompt_refinement import EconomicResidual
from eipg.simulators.llm_choice import TextPersona

FEATURES = ("price", "quality", "sustain", "novelty", "brand")
SEGMENTS = ("budget_sensitive", "quality_oriented", "sustainability_oriented")


@dataclass(frozen=True)
class StructuredPersona:
    persona_id: str
    segment_label: str
    mu: tuple[float, float, float, float, float]

    def as_dict(self) -> dict[str, float]:
        return {feature: float(value) for feature, value in zip(FEATURES, self.mu)}


def _strength_phrase(feature: str, value: float) -> str:
    """Render one scalar preference coordinate as controlled natural language.

    For price, larger values mean greater price sensitivity.  For the other
    dimensions, larger values mean stronger positive preference.
    """
    v = float(value)
    if feature == "price":
        if v <= -1.25:
            return "is very willing to pay higher prices when other attributes justify it"
        if v <= -0.50:
            return "is relatively insensitive to price and accepts moderate price premiums"
        if v < 0.50:
            return "notices price but balances it with the other product attributes"
        if v < 1.25:
            return "is price-conscious and tends to prefer lower-priced options"
        return "is highly price-sensitive and strongly favors lower-priced options"

    labels = {
        "quality": "product quality",
        "sustain": "sustainability",
        "novelty": "novel and innovative features",
        "brand": "familiar brands",
    }
    label = labels[feature]
    if v <= -1.25:
        return f"actively avoids paying for {label}"
    if v <= -0.50:
        return f"places relatively little value on {label}"
    if v < 0.50:
        return f"considers {label}, but does not treat it as a dominant factor"
    if v < 1.25:
        return f"places clear value on {label}"
    return f"places very high value on {label}"


def render_structured_persona(persona: StructuredPersona) -> TextPersona:
    values = persona.as_dict()
    clauses = [_strength_phrase(feature, values[feature]) for feature in FEATURES]
    prompt = (
        "A consumer making tradeoffs across product attributes. This shopper "
        + "; ".join(clauses)
        + ". They choose the option that best matches these preferences rather than following an absolute rule."
    )
    return TextPersona(
        persona_id=persona.persona_id,
        segment_label=persona.segment_label,
        prompt=prompt,
    )


def initial_structured_personas() -> tuple[StructuredPersona, ...]:
    """Frozen starting point, chosen to be intentionally moderate/under-grounded."""
    return (
        StructuredPersona("budget", "budget_sensitive", (0.75, 0.25, 0.00, 0.00, 0.25)),
        StructuredPersona("quality", "quality_oriented", (0.00, 0.75, 0.00, 0.00, 0.50)),
        StructuredPersona("sustain", "sustainability_oriented", (0.00, 0.25, 0.75, 0.50, 0.00)),
    )


def _direction_for_residual(r: EconomicResidual, feature: str) -> float:
    """Return the sign of the desired mu update for one residual/feature."""
    err = float(r.error)
    if err == 0.0:
        return 0.0

    if ".mean_chosen_" in r.name:
        residual_feature = r.name.rsplit("_", 1)[-1]
        if residual_feature != feature:
            return 0.0
        if feature == "price":
            # Chosen prices too low -> lower price sensitivity.
            return -1.0 if err < 0 else 1.0
        # Chosen non-price attribute too low -> increase preference.
        return 1.0 if err < 0 else -1.0

    if r.name.startswith("response."):
        parts = r.name.split(".")
        intervention = parts[1] if len(parts) > 1 else ""
        if intervention != feature:
            return 0.0
        if feature == "price":
            # More-negative-than-target response -> reduce price sensitivity.
            return -1.0 if err < 0 else 1.0
        # Too-positive response -> reduce responsiveness/preference.
        return -1.0 if err > 0 else 1.0

    return 0.0


def residual_update_direction(
    residuals: list[EconomicResidual],
    *,
    segment_label: str,
) -> dict[str, float]:
    """Aggregate own-segment levels plus global intervention responses.

    Magnitudes are used only to weight agreement among residuals. The returned
    vector is normalized to max absolute coordinate 1, keeping step size
    interpretable and preventing raw moment units from becoming preference units.
    """
    scores = {feature: 0.0 for feature in FEATURES}
    for residual in residuals:
        own = residual.name.startswith(f"{segment_label}.")
        global_response = residual.name.startswith("response.")
        if not (own or global_response):
            continue
        magnitude = abs(float(residual.error) * float(residual.weight))
        for feature in FEATURES:
            direction = _direction_for_residual(residual, feature)
            if direction:
                scores[feature] += direction * magnitude

    max_abs = max((abs(v) for v in scores.values()), default=0.0)
    if max_abs > 0:
        scores = {k: v / max_abs for k, v in scores.items()}
    return scores


def propose_update(
    personas: tuple[StructuredPersona, ...],
    residuals: list[EconomicResidual],
    *,
    step: float,
    lower: float = -1.75,
    upper: float = 1.75,
) -> tuple[StructuredPersona, ...]:
    """Apply one residual-directed step in structured preference space."""
    revised = []
    for persona in personas:
        direction = residual_update_direction(
            residuals, segment_label=persona.segment_label
        )
        current = persona.as_dict()
        new_values = tuple(
            min(upper, max(lower, current[feature] + float(step) * direction[feature]))
            for feature in FEATURES
        )
        revised.append(
            StructuredPersona(
                persona_id=persona.persona_id,
                segment_label=persona.segment_label,
                mu=new_values,
            )
        )
    return tuple(revised)


def render_population(
    personas: tuple[StructuredPersona, ...],
) -> tuple[TextPersona, ...]:
    return tuple(render_structured_persona(persona) for persona in personas)
