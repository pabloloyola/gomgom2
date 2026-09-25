"""Structured persona objects for EIPG.

Paper alignment
---------------
A sampled persona latent is the paper object ``z``.  In the clean
implementation, ``z`` is a numeric vector over named economic preference
features, for example price sensitivity, quality preference, sustainability
preference, novelty preference, and brand tendency.

The same ``z`` can later be realized in two ways:

- as utility coefficients in the controlled synthetic simulator;
- as a natural-language persona through a renderer for an LLM simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_FEATURES: tuple[str, ...] = (
    "price",
    "quality",
    "sustain",
    "novelty",
    "brand",
)

DEFAULT_SEGMENT_LABELS: tuple[str, ...] = (
    "budget_sensitive",
    "quality_brand_oriented",
    "sustainability_novelty_oriented",
)


def normalize_features(features: Iterable[str] | None, latent_dim: int | None = None) -> tuple[str, ...]:
    """Return a validated tuple of feature names.

    If ``features`` is omitted, use the first ``latent_dim`` entries from the
    paper-smoke default feature list.  For dimensions beyond the default list,
    generate generic names so experiments remain runnable.
    """

    if features is not None:
        out = tuple(str(f) for f in features)
        if not out:
            raise ValueError("features must be non-empty")
        if latent_dim is not None and len(out) != int(latent_dim):
            raise ValueError(
                f"latent_dim={latent_dim} but got {len(out)} feature names: {out}"
            )
        return out

    dim = int(latent_dim or len(DEFAULT_FEATURES))
    if dim <= len(DEFAULT_FEATURES):
        return DEFAULT_FEATURES[:dim]
    extra = tuple(f"feature_{i}" for i in range(len(DEFAULT_FEATURES), dim))
    return DEFAULT_FEATURES + extra


@dataclass(frozen=True)
class PersonaLatent:
    """A single sampled persona latent ``z``.

    Parameters
    ----------
    persona_id:
        Index within the current sampled batch.
    segment_id:
        Mixture component that generated the persona.
    segment_label:
        Human-readable segment label.
    z:
        Numeric latent preference vector.
    features:
        Names corresponding to the coordinates of ``z``.
    """

    persona_id: int
    segment_id: int
    segment_label: str
    z: np.ndarray
    features: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "features", tuple(self.features))
        if z.ndim != 1:
            raise ValueError(f"z must be a 1-D vector, got shape {z.shape}")
        if len(self.features) != len(z):
            raise ValueError(
                f"features length {len(self.features)} does not match z length {len(z)}"
            )

    @property
    def traits(self) -> dict[str, float]:
        """Coordinate dictionary, useful for logs and renderers."""

        return {name: float(value) for name, value in zip(self.features, self.z)}

    @property
    def id(self) -> int:
        """Compatibility alias for older code."""

        return self.persona_id

    @property
    def segment(self) -> int:
        """Compatibility alias for older code."""

        return self.segment_id

    def to_record(self) -> dict[str, object]:
        """Flat record suitable for a pandas DataFrame or parquet file."""

        rec: dict[str, object] = {
            "persona_id": int(self.persona_id),
            "segment_id": int(self.segment_id),
            "segment_label": self.segment_label,
        }
        for name, value in self.traits.items():
            rec[f"z_{name}"] = value
        return rec

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""

        return {
            "persona_id": int(self.persona_id),
            "segment_id": int(self.segment_id),
            "segment_label": self.segment_label,
            "features": list(self.features),
            "z": [float(v) for v in self.z],
            "traits": self.traits,
        }
