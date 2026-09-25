"""Mixture persona generator ``G_phi``.

The clean implementation represents ``G_phi`` as a mixture over numeric
preference vectors.  The optimized object is structured and simulator-agnostic:
LLM prompt text is produced later by a renderer, not optimized directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from eipg.personas.schema import (
    DEFAULT_SEGMENT_LABELS,
    PersonaLatent,
    normalize_features,
)


@dataclass(frozen=True)
class MixtureGeneratorParams:
    """Parameters of the mixture persona generator.

    This is the clean-code counterpart of the paper parameter ``phi``:
    mixture weights plus component means.  We keep covariance fixed for the
    early controlled synthetic benchmark.
    """

    weights: np.ndarray
    means: np.ndarray
    features: tuple[str, ...]
    within_component_std: float = 0.30
    segment_labels: tuple[str, ...] = DEFAULT_SEGMENT_LABELS

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        means = np.asarray(self.means, dtype=float)
        features = tuple(self.features)
        labels = tuple(self.segment_labels)

        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got {weights.shape}")
        if means.ndim != 2:
            raise ValueError(f"means must be 2-D, got {means.shape}")
        if means.shape[0] != len(weights):
            raise ValueError("number of weights must match number of component means")
        if means.shape[1] != len(features):
            raise ValueError("mean-vector dimension must match number of features")
        if np.any(weights < 0):
            raise ValueError("mixture weights must be non-negative")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("mixture weights must have positive sum")
        weights = weights / total
        if len(labels) < len(weights):
            labels = labels + tuple(f"segment_{i}" for i in range(len(labels), len(weights)))

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "within_component_std", float(self.within_component_std))
        object.__setattr__(self, "segment_labels", labels[: len(weights)])

    @property
    def k_components(self) -> int:
        return int(self.means.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def log_weights(self) -> np.ndarray:
        return np.log(np.clip(self.weights, 1e-12, None))

    def to_flat_phi(self) -> np.ndarray:
        """Encode parameters as a flat vector for later black-box optimizers."""

        return np.concatenate([self.log_weights, self.means.ravel()])

    @classmethod
    def from_flat_phi(
        cls,
        phi: np.ndarray,
        *,
        k_components: int,
        features: tuple[str, ...],
        within_component_std: float = 0.30,
        segment_labels: tuple[str, ...] = DEFAULT_SEGMENT_LABELS,
    ) -> "MixtureGeneratorParams":
        """Decode a flat vector into mixture weights and means."""

        phi = np.asarray(phi, dtype=float)
        dim = len(features)
        expected = k_components + k_components * dim
        if len(phi) != expected:
            raise ValueError(f"expected phi length {expected}, got {len(phi)}")
        logits = phi[:k_components]
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        means = phi[k_components:].reshape(k_components, dim)
        return cls(
            weights=weights,
            means=means,
            features=features,
            within_component_std=within_component_std,
            segment_labels=segment_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_components": self.k_components,
            "latent_dim": self.latent_dim,
            "features": list(self.features),
            "weights": [float(v) for v in self.weights],
            "means": self.means.tolist(),
            "within_component_std": float(self.within_component_std),
            "segment_labels": list(self.segment_labels),
            "flat_phi": [float(v) for v in self.to_flat_phi()],
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


class MixturePersonaGenerator:
    """Sample personas from ``p_phi(z)``."""

    def __init__(self, params: MixtureGeneratorParams, seed: int = 0):
        self.params = params
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> list[PersonaLatent]:
        """Sample ``n`` individual persona latents."""

        if n <= 0:
            return []
        segs = self.rng.choice(self.params.k_components, size=int(n), p=self.params.weights)
        out: list[PersonaLatent] = []
        for i, seg in enumerate(segs):
            z = self.rng.normal(
                loc=self.params.means[int(seg)],
                scale=self.params.within_component_std,
                size=self.params.latent_dim,
            )
            out.append(
                PersonaLatent(
                    persona_id=i,
                    segment_id=int(seg),
                    segment_label=self.params.segment_labels[int(seg)],
                    z=z,
                    features=self.params.features,
                )
            )
        return out

    def prototypes(self) -> list[PersonaLatent]:
        """Return deterministic prototype personas at component means."""

        return [
            PersonaLatent(
                persona_id=k,
                segment_id=k,
                segment_label=self.params.segment_labels[k],
                z=self.params.means[k].copy(),
                features=self.params.features,
            )
            for k in range(self.params.k_components)
        ]

    @property
    def effective_weights(self) -> np.ndarray:
        return self.params.weights.copy()

    @property
    def component_means(self) -> np.ndarray:
        return self.params.means.copy()


def _paper_smoke_means(k_components: int, latent_dim: int) -> np.ndarray:
    """Small, interpretable initialization used by the local smoke config.

    The first five coordinates are interpreted as price, quality,
    sustainability, novelty, and brand.  Extra dimensions, if any, are zero.
    """

    base = np.array(
        [
            [-2.0, 1.0, 0.2, 0.1, 0.0],   # budget-sensitive, moderate quality
            [-0.6, 1.8, 0.3, 0.1, 0.9],   # quality/brand oriented
            [-0.8, 0.8, 1.6, 1.2, 0.2],   # sustainability/novelty oriented
        ],
        dtype=float,
    )
    means = np.zeros((k_components, latent_dim), dtype=float)
    for k in range(k_components):
        row = base[k % len(base)]
        means[k, : min(latent_dim, len(row))] = row[: min(latent_dim, len(row))]
    return means


def params_from_config(section: dict[str, Any]) -> MixtureGeneratorParams:
    """Create generator parameters from a config section."""

    k = int(section.get("k_components", 3))
    latent_dim = int(section.get("latent_dim", len(section.get("features", [])) or 5))
    features = normalize_features(section.get("features"), latent_dim=latent_dim)
    within_std = float(section.get("within_component_std", 0.30))
    labels = tuple(section.get("segment_labels", DEFAULT_SEGMENT_LABELS))

    if "weights" in section:
        weights = np.asarray(section["weights"], dtype=float)
    else:
        weights = np.ones(k, dtype=float) / k

    if "means" in section:
        means = np.asarray(section["means"], dtype=float)
    else:
        init = str(section.get("init", "paper_smoke"))
        if init == "paper_smoke":
            means = _paper_smoke_means(k, latent_dim)
        elif init in {"zero", "zeros"}:
            means = np.zeros((k, latent_dim), dtype=float)
        else:
            raise ValueError(f"Unknown persona_generator.init: {init}")

    return MixtureGeneratorParams(
        weights=weights,
        means=means,
        features=features,
        within_component_std=within_std,
        segment_labels=labels,
    )
