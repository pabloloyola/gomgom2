"""Anchored structured behavioral coordinates for LLM persona calibration.

Each persona retains its original natural-language description as an immutable
semantic anchor.  A low-dimensional vector delta_mu adds controlled behavioral
adjustments relative to that anchor.  The all-zero vector renders to the exact
original prompt, making the structured search directly comparable with the
free-form prompt-refinement baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from eipg.simulators.llm_choice import TextPersona

FEATURES = ("price", "quality", "sustain", "novelty", "brand")


@dataclass(frozen=True)
class AnchoredPersona:
    persona_id: str
    segment_label: str
    base_prompt: str
    delta_mu: tuple[float, float, float, float, float]

    def as_dict(self) -> dict[str, float]:
        return {
            feature: float(value)
            for feature, value in zip(FEATURES, self.delta_mu)
        }


_BASE_PERSONAS = (
    AnchoredPersona(
        "budget",
        "budget_sensitive",
        (
            "A practical shopper who looks for good overall value. They compare price, quality, "
            "and familiar brands, and they are open to sustainability or novelty when those features "
            "make the offer more appealing."
        ),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    AnchoredPersona(
        "quality",
        "quality_oriented",
        (
            "A balanced shopper who considers price, product quality, and brand familiarity together. "
            "They also notice sustainability and new features, but usually weigh several aspects before deciding."
        ),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    AnchoredPersona(
        "sustain",
        "sustainability_oriented",
        (
            "An environmentally aware shopper who likes sustainable and innovative products, but usually "
            "balances those benefits against price, quality, and brand familiarity rather than paying a strong premium."
        ),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)


def initial_anchored_personas() -> tuple[AnchoredPersona, ...]:
    return _BASE_PERSONAS


def _intensity(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude <= 0.5 + 1e-9:
        return "slightly"
    if magnitude <= 1.0 + 1e-9:
        return "moderately"
    return "substantially"


def _adjustment_phrase(feature: str, value: float) -> str:
    if abs(float(value)) < 1e-12:
        raise ValueError("zero coordinates should not be rendered as adjustments")
    direction = 1 if value > 0 else -1
    intensity = _intensity(value)

    if feature == "price":
        return (
            f"be {intensity} more price-sensitive and more inclined toward lower-priced options"
            if direction > 0
            else f"be {intensity} less price-sensitive and more willing to pay a premium when justified"
        )

    labels = {
        "quality": "product quality",
        "sustain": "sustainability",
        "novelty": "novel and innovative features",
        "brand": "familiar brands",
    }
    label = labels[feature]
    return (
        f"place {intensity} more weight on {label}"
        if direction > 0
        else f"place {intensity} less weight on {label}"
    )


def render_anchored_persona(persona: AnchoredPersona) -> TextPersona:
    """Render delta_mu relative to the immutable base prompt.

    The zero vector returns the base prompt byte-for-byte.  Non-zero coordinates
    append one compact calibration sentence whose clauses are deterministic.
    """
    values = persona.as_dict()
    active = [
        _adjustment_phrase(feature, values[feature])
        for feature in FEATURES
        if abs(values[feature]) >= 1e-12
    ]
    if not active:
        prompt = persona.base_prompt
    else:
        prompt = (
            persona.base_prompt
            + " For this simulation, relative to that baseline description, "
            + "; ".join(active)
            + ". Preserve all other preferences and tradeoffs from the baseline."
        )
    return TextPersona(
        persona_id=persona.persona_id,
        segment_label=persona.segment_label,
        prompt=prompt,
    )


def render_population(
    personas: tuple[AnchoredPersona, ...],
) -> tuple[TextPersona, ...]:
    return tuple(render_anchored_persona(persona) for persona in personas)


def coordinate_neighbor(
    personas: tuple[AnchoredPersona, ...],
    *,
    persona_index: int,
    feature_index: int,
    delta: float,
    lower: float,
    upper: float,
) -> tuple[AnchoredPersona, ...]:
    """Return one population with exactly one delta_mu coordinate changed."""
    if not (0 <= persona_index < len(personas)):
        raise IndexError("persona_index out of range")
    if not (0 <= feature_index < len(FEATURES)):
        raise IndexError("feature_index out of range")

    persona = personas[persona_index]
    values = list(persona.delta_mu)
    proposed = min(upper, max(lower, float(values[feature_index]) + float(delta)))
    if abs(proposed - float(values[feature_index])) < 1e-12:
        return personas
    values[feature_index] = proposed

    updated = list(personas)
    updated[persona_index] = replace(persona, delta_mu=tuple(values))
    return tuple(updated)


def neighborhood(
    personas: tuple[AnchoredPersona, ...],
    *,
    step: float,
    lower: float,
    upper: float,
) -> list[tuple[int, str, float, tuple[AnchoredPersona, ...]]]:
    """Enumerate +/- one-coordinate neighbors in deterministic order."""
    out = []
    for persona_index, _persona in enumerate(personas):
        for feature_index, feature in enumerate(FEATURES):
            for signed_step in (-float(step), float(step)):
                candidate = coordinate_neighbor(
                    personas,
                    persona_index=persona_index,
                    feature_index=feature_index,
                    delta=signed_step,
                    lower=lower,
                    upper=upper,
                )
                if candidate == personas:
                    continue
                out.append((persona_index, feature, signed_step, candidate))
    return out
