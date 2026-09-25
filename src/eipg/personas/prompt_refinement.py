"""Residual-driven textual persona refinement for EIPG.

This module keeps the editor agnostic to the underlying chat provider.  The
editor sees only calibration residuals and the current prompt; held-out labels
must never appear in the payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
import re


class ChatLike(Protocol):
    def chat(self, messages: list[dict[str, str]], *, response_format: dict | None = None): ...


@dataclass(frozen=True)
class EconomicResidual:
    name: str
    target: float
    simulated: float
    weight: float = 1.0

    @property
    def error(self) -> float:
        return float(self.simulated - self.target)


@dataclass(frozen=True)
class PromptEditResult:
    prompt: str
    raw_text: str
    prompt_hash: str


@dataclass(frozen=True)
class ResidualPromptEditorConfig:
    max_prompt_chars: int = 2400
    max_residuals: int = 24
    forbidden_terms: tuple[str, ...] = (
        "held-out",
        "heldout",
        "test choice",
        "test label",
        "task 4 outcome",
        "counterfactual truth",
    )


class ResidualPromptEditor:
    """Rewrite a persona prompt using only economic calibration residuals."""

    def __init__(self, client: ChatLike, config: ResidualPromptEditorConfig | None = None) -> None:
        self.client = client
        self.config = config or ResidualPromptEditorConfig()

    def _validate_no_leakage(self, text: str) -> None:
        lowered = text.lower()
        bad = [term for term in self.config.forbidden_terms if term.lower() in lowered]
        if bad:
            raise ValueError(f"editor payload contains forbidden held-out information: {bad}")

    def render_residuals(
        self,
        residuals: list[EconomicResidual],
        *,
        segment_label: str | None = None,
    ) -> str:
        relevant = residuals
        if segment_label is not None:
            relevant = [
                r
                for r in residuals
                if r.name.startswith(f"{segment_label}.") or r.name.startswith("response.")
            ]
        ranked = sorted(
            relevant,
            key=lambda r: abs(r.error) * abs(r.weight),
            reverse=True,
        )[: self.config.max_residuals]
        rows = [
            {
                "moment": r.name,
                "target": round(float(r.target), 6),
                "simulated": round(float(r.simulated), 6),
                "sim_minus_target": round(float(r.error), 6),
                "weight": round(float(r.weight), 6),
            }
            for r in ranked
        ]
        rendered = json.dumps(rows, ensure_ascii=False, indent=2)
        self._validate_no_leakage(rendered)
        return rendered


    @staticmethod
    def _guidance_for_residual(r: EconomicResidual) -> str:
        """Translate a signed calibration residual into a behavioral edit direction.

        The mapping is deterministic and uses only calibration moments.  It avoids
        asking the LLM editor to infer economic sign conventions from raw numbers.
        """
        name = r.name
        err = r.error
        if ".mean_chosen_" in name:
            feature = name.rsplit("_", 1)[-1]
            if feature == "price":
                return (
                    "reduce price sensitivity slightly; be more willing to choose higher-priced options"
                    if err < 0
                    else "increase price sensitivity slightly; favor lower-priced options more"
                )
            labels = {
                "quality": "quality",
                "sustain": "sustainability",
                "novelty": "novelty",
                "brand": "brand familiarity",
            }
            label = labels.get(feature, feature)
            return (
                f"increase preference for {label} slightly"
                if err < 0
                else f"decrease preference for {label} slightly"
            )
        if name.startswith("response."):
            parts = name.split(".")
            intervention = parts[1] if len(parts) > 1 else ""
            if intervention == "price":
                return (
                    "reduce responsiveness to price increases slightly"
                    if err < 0
                    else "increase responsiveness to price increases slightly"
                )
            labels = {
                "quality": "quality improvements",
                "sustain": "sustainability improvements",
                "novelty": "novelty improvements",
            }
            label = labels.get(intervention, intervention)
            return (
                f"reduce responsiveness to {label} slightly"
                if err > 0
                else f"increase responsiveness to {label} slightly"
            )
        return "make only a small adjustment toward reducing this residual"

    def render_guidance(
        self,
        residuals: list[EconomicResidual],
        *,
        segment_label: str,
    ) -> str:
        relevant = [
            r
            for r in residuals
            if r.name.startswith(f"{segment_label}.") or r.name.startswith("response.")
        ]
        ranked = sorted(
            relevant,
            key=lambda r: abs(r.error) * abs(r.weight),
            reverse=True,
        )[: self.config.max_residuals]
        rows = [
            {
                "rank": i + 1,
                "moment": r.name,
                "direction": self._guidance_for_residual(r),
            }
            for i, r in enumerate(ranked)
        ]
        rendered = json.dumps(rows, ensure_ascii=False, indent=2)
        self._validate_no_leakage(rendered)
        return rendered

    def render_messages(
        self,
        *,
        current_prompt: str,
        residuals: list[EconomicResidual],
        segment_label: str,
        use_economic_signal: bool = True,
    ) -> list[dict[str, str]]:
        if len(current_prompt) > self.config.max_prompt_chars:
            raise ValueError("current persona prompt exceeds configured length limit")
        self._validate_no_leakage(current_prompt)
        if use_economic_signal:
            guidance_text = self.render_guidance(
                residuals,
                segment_label=segment_label,
            )
            system = (
                "You are editing a consumer persona used in a behavioral choice simulator. "
                "The supplied behavioral directions were deterministically derived from signed "
                "calibration residuals, so follow their stated direction rather than re-interpreting "
                "the underlying sign. Preserve identity and demographic framing. Make a minimal, "
                "conservative behavioral edit: adjust only the two or three highest-ranked directions "
                "that are relevant and mutually consistent. Do not turn a preference into an absolute "
                "rule and avoid words such as 'always', 'never', 'above all else', or 'strongly' unless "
                "they already appear in the current persona. Do not mention calibration, models, "
                "residuals, experiments, ranks, or numeric targets. Return only JSON with key revised_prompt."
            )
            user = (
                f"Segment label: {segment_label}\n\n"
                f"Current persona prompt:\n{current_prompt}\n\n"
                f"Residual-derived behavioral directions (highest priority first):\n{guidance_text}\n\n"
                "Revise the persona with the smallest natural wording changes needed to move choices "
                "in these directions. Preserve unrelated preferences and tradeoffs."
            )
        else:
            system = (
                "You are performing a style-only paraphrase of a consumer persona. "
                "Preserve every behavioral preference, priority ordering, tradeoff, and preference "
                "strength exactly. Do not add, remove, strengthen, or weaken any preference. "
                "Only improve clarity and naturalness. Return only JSON with key revised_prompt."
            )
            user = (
                f"Current persona prompt:\n{current_prompt}\n\n"
                "Paraphrase for clarity while preserving the behavioral content exactly."
            )
        self._validate_no_leakage(user)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def parse_revised_prompt(text: str) -> str:
        """Parse editor output while tolerating harmless formatting wrappers."""
        stripped = text.strip()
        candidates = [stripped]

        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(stripped):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(stripped[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("revised_prompt"), str):
                candidates.append(json.dumps(obj))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("revised_prompt"), str):
                prompt = payload["revised_prompt"].strip()
                if prompt:
                    return prompt

        raise ValueError("editor response must contain JSON with string key revised_prompt")

    def edit(
        self,
        *,
        current_prompt: str,
        residuals: list[EconomicResidual],
        segment_label: str,
        use_economic_signal: bool = True,
    ) -> PromptEditResult:
        messages = self.render_messages(
            current_prompt=current_prompt,
            residuals=residuals,
            segment_label=segment_label,
            use_economic_signal=use_economic_signal,
        )
        result = self.client.chat(messages, response_format={"type": "json_object"})
        try:
            revised = self.parse_revised_prompt(result.text)
            raw_text = result.text
            prompt_hash = result.prompt_hash
        except ValueError:
            # Direct HF generation cannot enforce response_format. Make one
            # formatting-only repair call that preserves the proposed wording.
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Convert the supplied text into valid JSON only. "
                        "Return exactly one JSON object with string key revised_prompt. "
                        "Preserve the proposed persona wording exactly; do not rewrite, "
                        "add, remove, strengthen, or weaken any preference."
                    ),
                },
                {"role": "user", "content": result.text},
            ]
            repaired = self.client.chat(
                repair_messages, response_format={"type": "json_object"}
            )
            revised = self.parse_revised_prompt(repaired.text)
            raw_text = repaired.text
            prompt_hash = repaired.prompt_hash

        self._validate_no_leakage(revised)
        if len(revised) > self.config.max_prompt_chars:
            raise ValueError("revised persona prompt exceeds configured length limit")
        return PromptEditResult(revised, raw_text, prompt_hash)
