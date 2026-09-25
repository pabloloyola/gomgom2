"""Render structured persona latents into human-readable text.

This module is deliberately independent of any LLM API.  It only maps the
numeric latent ``z`` into a stable text description.  Later milestones will add
choice-context rendering and LM Studio calls.
"""

from __future__ import annotations

from eipg.personas.schema import PersonaLatent


def _feature_gloss(name: str, value: float) -> str:
    mag = abs(float(value))

    if name == "price":
        if value < -1.5:
            return "strongly dislikes high prices"
        if value < -0.5:
            return "prefers lower prices"
        if value < 0:
            return "is mildly price-conscious"
        if value > 0.5:
            return "is relatively insensitive to higher prices"
        return "has no strong price preference"

    label = {
        "quality": "quality",
        "sustain": "sustainability claims",
        "sustainability": "sustainability claims",
        "novelty": "novelty",
        "brand": "brand names",
    }.get(name, name.replace("_", " "))

    if mag < 0.25:
        return f"is not especially influenced by {label}"
    if value > 0:
        if mag > 1.2:
            return f"strongly values {label}"
        return f"cares moderately about {label}"
    if mag > 1.2:
        return f"strongly avoids high {label}"
    return f"does not prioritize {label}"


class PersonaRenderer:
    """Canonical renderer for explaining or prompting with a sampled ``z``."""

    def render_profile(self, persona: PersonaLatent) -> str:
        """Render a compact, prompt-friendly persona description."""

        traits = persona.traits
        clauses: list[str] = []
        for name in persona.features:
            clauses.append(_feature_gloss(name, traits[name]))

        # Join as one short paragraph.  The resulting string is suitable for a
        # LaTeX quote block or for an LLM prompt in later milestones.
        body = ", ".join(clauses[:-1])
        if len(clauses) > 1:
            body = body + ", and " + clauses[-1]
        elif clauses:
            body = clauses[0]
        else:
            body = "has unspecified preferences"

        return f"You are a {persona.segment_label.replace('_', '-')} shopper who {body}."

    def render_structured(self, persona: PersonaLatent) -> str:
        """Render numeric coordinates plus verbal glosses for debugging."""

        lines = [f"Persona {persona.persona_id} ({persona.segment_label})"]
        for name, value in persona.traits.items():
            lines.append(f"- {name}: {value:+.2f} -> {_feature_gloss(name, value)}")
        return "\n".join(lines)
