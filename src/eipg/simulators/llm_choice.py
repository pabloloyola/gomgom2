"""Text-persona choice simulator built on the generic chat client."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
import json
import re

import pandas as pd


class ChatLike(Protocol):
    def chat(self, messages: list[dict[str, str]], *, response_format: dict | None = None): ...


@dataclass(frozen=True)
class TextPersona:
    persona_id: str
    segment_label: str
    prompt: str


@dataclass(frozen=True)
class TextChoiceResult:
    alternative_id: int
    raw_text: str
    prompt_hash: str
    cached: bool
    latency_seconds: float


class TextPersonaChoiceSimulator:
    def __init__(self, client: ChatLike) -> None:
        self.client = client

    @staticmethod
    def render_slate(slate: pd.DataFrame) -> str:
        if "alternative_id" not in slate.columns:
            raise ValueError("slate must contain alternative_id")
        excluded = {"context_id", "alternative_id", "observation_id", "dataset", "chosen"}
        feature_cols = [
            c for c in slate.columns
            if c not in excluded and not c.startswith("z_") and pd.api.types.is_numeric_dtype(slate[c])
        ]
        lines = []
        for _, row in slate.sort_values("alternative_id").iterrows():
            attrs = ", ".join(f"{c}={row[c]:.4g}" for c in feature_cols)
            lines.append(f"Alternative {int(row['alternative_id'])}: {attrs}")
        return "\n".join(lines)

    def render_messages(self, *, slate: pd.DataFrame, persona: TextPersona) -> list[dict[str, str]]:
        ids = sorted(int(v) for v in slate["alternative_id"].unique().tolist())
        system = (
            "Simulate exactly one consumer choice while faithfully adopting the supplied persona. "
            "Choose one available alternative. Do not explain your reasoning. Return only JSON "
            "with key alternative_id."
        )
        user = (
            f"Persona segment: {persona.segment_label}\n"
            f"Persona:\n{persona.prompt}\n\n"
            f"Available alternatives:\n{self.render_slate(slate)}\n\n"
            f"Valid alternative_id values: {ids}."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def parse_choice(text: str, valid_ids: list[int]) -> int:
        """Parse one choice while tolerating common provider formatting wrappers.

        The scientific object is the selected alternative, not markdown formatting. We
        therefore accept strict JSON first, then fenced JSON, then an explicit labelled
        ``alternative_id`` field, and finally a bare integer. We intentionally do not
        scrape arbitrary integers from free-form reasoning.
        """
        valid = set(map(int, valid_ids))
        stripped = text.strip()

        candidates = [stripped]
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict) and "alternative_id" in payload:
                    value = int(payload["alternative_id"])
                    if value in valid:
                        return value
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        labelled = re.search(
            r"(?:\"?alternative_id\"?|alternative)\s*[:=]\s*\"?(-?\d+)\"?",
            stripped,
            flags=re.IGNORECASE,
        )
        if labelled:
            value = int(labelled.group(1))
            if value in valid:
                return value

        exact = re.fullmatch(r"\s*(-?\d+)\s*", stripped)
        if exact and int(exact.group(1)) in valid:
            return int(exact.group(1))
        raise ValueError(f"invalid choice response: {text!r}")

    def choose(self, *, slate: pd.DataFrame, persona: TextPersona) -> TextChoiceResult:
        valid_ids = sorted(int(v) for v in slate["alternative_id"].unique().tolist())
        messages = self.render_messages(slate=slate, persona=persona)
        result = self.client.chat(messages, response_format={"type": "json_object"})
        choice = self.parse_choice(result.text, valid_ids)
        return TextChoiceResult(
            alternative_id=choice,
            raw_text=result.text,
            prompt_hash=result.prompt_hash,
            cached=bool(result.cached),
            latency_seconds=float(result.latency_seconds),
        )

    def simulate_fixed_pairs(
        self,
        *,
        pairs: list[tuple[str, TextPersona, pd.DataFrame]],
        dataset_label: str,
    ) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        for observation_id, persona, slate in pairs:
            result = self.choose(slate=slate, persona=persona)
            frame = slate.copy().reset_index(drop=True)
            frame.insert(0, "observation_id", str(observation_id))
            frame.insert(1, "dataset", dataset_label)
            frame.insert(2, "persona_id", persona.persona_id)
            frame.insert(3, "persona_segment_label", persona.segment_label)
            frame["chosen"] = (frame["alternative_id"].astype(int) == result.alternative_id).astype(int)
            frame["llm_raw_text"] = result.raw_text
            frame["llm_prompt_hash"] = result.prompt_hash
            frame["llm_cached"] = result.cached
            frame["llm_latency_seconds"] = result.latency_seconds
            frame["persona_prompt_hash"] = sha256(persona.prompt.encode("utf-8")).hexdigest()
            records.append(frame)
        if not records:
            raise ValueError("pairs must be non-empty")
        return pd.concat(records, ignore_index=True)
