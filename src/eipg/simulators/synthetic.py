"""Controlled synthetic choice simulator ``pi_theta``.

Paper alignment
---------------
This module implements the first controlled-benchmark simulator used by the
clean rebuild.  Given a choice context ``x`` and a persona latent ``z``, the
simulator draws a choice ``y`` from a random-utility model:

    U(j | x, z) = z^T f(x, j) + epsilon_j

where ``f(x, j)`` is the attribute vector of alternative ``j`` in context ``x``.
Equivalently, after integrating out i.i.d. Gumbel noise, choice probabilities
follow a softmax over systematic utilities.  This is deliberately simple: the
purpose of the controlled benchmark is to make the data-generating process
transparent and auditable before adding LLM simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eipg.personas.schema import PersonaLatent


@dataclass(frozen=True)
class SyntheticSimulatorConfig:
    """Configuration for the random-utility synthetic simulator."""

    choice_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.choice_temperature <= 0:
            raise ValueError("choice_temperature must be positive")

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "SyntheticSimulatorConfig":
        return cls(choice_temperature=float(section.get("choice_temperature", 1.0)))


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(exp) / len(exp)
    return exp / total


def _feature_columns(contexts: pd.DataFrame, persona: PersonaLatent) -> list[str]:
    cols = [name for name in persona.features if name in contexts.columns]
    if not cols:
        raise ValueError(
            "No persona feature names are present in context columns. "
            f"persona.features={persona.features}, context columns={tuple(contexts.columns)}"
        )
    return cols


class RandomUtilityChoiceSimulator:
    """Simulate choices from persona latents and long-format choice contexts."""

    def __init__(self, config: SyntheticSimulatorConfig | None = None, seed: int = 0):
        self.config = config or SyntheticSimulatorConfig()
        self.rng = np.random.default_rng(seed)

    def choice_probabilities(self, slate: pd.DataFrame, persona: PersonaLatent) -> pd.DataFrame:
        """Return utility and choice probability for each alternative in one context."""

        if slate["context_id"].nunique() != 1:
            raise ValueError("choice_probabilities expects exactly one context/slate")
        feature_cols = _feature_columns(slate, persona)
        beta = np.array([persona.traits[name] for name in feature_cols], dtype=float)
        x = slate[feature_cols].to_numpy(dtype=float)
        utilities = x @ beta
        probs = _softmax(utilities, temperature=self.config.choice_temperature)
        out = slate.copy().reset_index(drop=True)
        out["utility"] = utilities
        out["choice_prob"] = probs
        return out

    def choose(self, slate: pd.DataFrame, persona: PersonaLatent) -> pd.DataFrame:
        """Draw one chosen alternative for a single ``(x, z)`` pair."""

        scored = self.choice_probabilities(slate, persona)
        alt_idx = self.rng.choice(len(scored), p=scored["choice_prob"].to_numpy(dtype=float))
        scored["chosen"] = 0
        scored.loc[int(alt_idx), "chosen"] = 1
        return scored

    def simulate_long_dataset(
        self,
        *,
        contexts: pd.DataFrame,
        personas: Iterable[PersonaLatent],
        n_observations: int,
        dataset_label: str,
    ) -> pd.DataFrame:
        """Simulate a long-format choice dataset.

        The returned DataFrame has one row per alternative per observation and a
        binary ``chosen`` column.  This is the format needed by later MNL fitting
        and moment calculations.
        """

        personas = list(personas)
        if not personas:
            raise ValueError("personas must be non-empty")
        if n_observations <= 0:
            raise ValueError("n_observations must be positive")
        if contexts.empty:
            raise ValueError("contexts must be non-empty")
        if "context_id" not in contexts.columns:
            raise ValueError("contexts must contain a context_id column")

        grouped = {cid: g.reset_index(drop=True) for cid, g in contexts.groupby("context_id", sort=True)}
        context_ids = np.array(sorted(grouped.keys()), dtype=object)

        records: list[pd.DataFrame] = []
        for obs_id in range(int(n_observations)):
            cid = str(self.rng.choice(context_ids))
            persona = personas[int(self.rng.integers(0, len(personas)))]
            chosen_slate = self.choose(grouped[cid], persona)
            chosen_slate.insert(0, "observation_id", f"{dataset_label}_{obs_id:06d}")
            chosen_slate.insert(1, "dataset", dataset_label)
            chosen_slate.insert(2, "persona_id", int(persona.persona_id))
            chosen_slate.insert(3, "persona_segment_id", int(persona.segment_id))
            chosen_slate.insert(4, "persona_segment_label", persona.segment_label)
            for feature, value in persona.traits.items():
                chosen_slate[f"z_{feature}"] = float(value)
            records.append(chosen_slate)

        return pd.concat(records, ignore_index=True)


def summarize_choice_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Small JSON-serializable summary for smoke runs and notebooks."""

    if df.empty:
        return {"n_rows": 0, "n_observations": 0}
    chosen = df[df["chosen"].astype(int) == 1].copy()
    attribute_cols = [
        c
        for c in ["price", "quality", "sustain", "novelty", "brand"]
        if c in chosen.columns
    ]
    return {
        "n_rows": int(len(df)),
        "n_observations": int(df["observation_id"].nunique()),
        "n_contexts_used": int(df["context_id"].nunique()),
        "n_chosen_rows": int(len(chosen)),
        "choice_share_by_alternative": {
            str(k): float(v)
            for k, v in chosen["alternative_id"].value_counts(normalize=True).sort_index().items()
        },
        "choice_share_by_segment": {
            str(k): float(v)
            for k, v in chosen["persona_segment_label"].value_counts(normalize=True).sort_index().items()
        },
        "mean_chosen_attributes": {
            c: float(chosen[c].mean()) for c in attribute_cols
        },
    }
