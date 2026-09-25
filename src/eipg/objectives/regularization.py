"""Generator regularization for EIPG.

Paper alignment
---------------
The paper writes the generator penalty as

    R(phi) = alpha_H R_entropy(phi)
           + alpha_D R_disp(phi)
           + alpha_P R_prior(phi).

This module keeps the first implementation explicit and inspectable.  We report
both human-readable diagnostics, such as mixture entropy and average pairwise
component distance, and the signed objective terms used for minimization:

- ``R_entropy = -H(w)``, so minimizing encourages high mixture entropy;
- ``R_dispersion`` can either reward separation (legacy behavior) or impose a
  band penalty that is zero inside a plausible dispersion range and positive
  outside it;
- ``R_prior`` is an optional squared distance to a prior over mixture weights.

The regularizer is used by the v0.8 outer objective and is also computed for
all baseline generators so notebooks can reveal whether a candidate is collapsed,
well separated, or close to an optional prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from eipg.personas.generator import MixtureGeneratorParams


@dataclass(frozen=True)
class RegularizationConfig:
    """Weights and optional prior for the generator regularizer."""

    entropy_weight: float = 0.0
    dispersion_weight: float = 0.0
    prior_weight: float = 0.0
    prior_weights: tuple[float, ...] | None = None
    dispersion_mode: str = "reward_separation"
    dispersion_min: float = 0.0
    dispersion_max: float | None = None

    def __post_init__(self) -> None:
        if self.entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative")
        if self.dispersion_weight < 0:
            raise ValueError("dispersion_weight must be non-negative")
        if self.prior_weight < 0:
            raise ValueError("prior_weight must be non-negative")
        allowed_modes = {"reward_separation", "band"}
        if self.dispersion_mode not in allowed_modes:
            raise ValueError(f"dispersion_mode must be one of {sorted(allowed_modes)}")
        if self.dispersion_min < 0:
            raise ValueError("dispersion_min must be non-negative")
        if self.dispersion_max is not None and self.dispersion_max < self.dispersion_min:
            raise ValueError("dispersion_max must be >= dispersion_min")
        if self.dispersion_mode == "band" and self.dispersion_max is None:
            raise ValueError("dispersion_max is required when dispersion_mode='band'")
        if self.prior_weights is not None:
            prior = np.asarray(self.prior_weights, dtype=float)
            if prior.ndim != 1 or len(prior) == 0:
                raise ValueError("prior_weights must be a non-empty 1-D vector")
            if np.any(prior < 0) or prior.sum() <= 0:
                raise ValueError("prior_weights must be non-negative and have positive sum")
            prior = prior / prior.sum()
            object.__setattr__(self, "prior_weights", tuple(float(v) for v in prior))

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "RegularizationConfig":
        prior_weights = section.get("prior_weights")
        return cls(
            entropy_weight=float(section.get("entropy_weight", 0.0)),
            dispersion_weight=float(section.get("dispersion_weight", 0.0)),
            prior_weight=float(section.get("prior_weight", 0.0)),
            prior_weights=tuple(float(v) for v in prior_weights) if prior_weights else None,
            dispersion_mode=str(section.get("dispersion_mode", "reward_separation")),
            dispersion_min=float(section.get("dispersion_min", 0.0)),
            dispersion_max=(
                float(section["dispersion_max"])
                if section.get("dispersion_max") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy_weight": float(self.entropy_weight),
            "dispersion_weight": float(self.dispersion_weight),
            "prior_weight": float(self.prior_weight),
            "prior_weights": list(self.prior_weights) if self.prior_weights is not None else None,
            "dispersion_mode": self.dispersion_mode,
            "dispersion_min": float(self.dispersion_min),
            "dispersion_max": float(self.dispersion_max) if self.dispersion_max is not None else None,
        }


@dataclass(frozen=True)
class RegularizationReport:
    """Regularization terms for one generator."""

    terms: dict[str, float]
    config: RegularizationConfig

    def objective(self) -> float:
        return float(self.terms["regularization_objective"])

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "terms": self.terms}

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def mixture_entropy(weights: Iterable[float]) -> float:
    """Return Shannon entropy of mixture weights."""

    w = np.asarray(tuple(weights), dtype=float)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("weights must be a non-empty 1-D vector")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and have positive sum")
    w = w / w.sum()
    return float(-np.sum(w * np.log(np.clip(w, 1.0e-12, None))))


def pairwise_component_distances(means: np.ndarray) -> np.ndarray:
    """Return all pairwise Euclidean distances between component means."""

    mu = np.asarray(means, dtype=float)
    if mu.ndim != 2:
        raise ValueError("means must be a 2-D array")
    distances: list[float] = []
    for i in range(mu.shape[0]):
        for j in range(i + 1, mu.shape[0]):
            distances.append(float(np.linalg.norm(mu[i] - mu[j])))
    return np.asarray(distances, dtype=float)


def prior_weight_penalty(weights: Iterable[float], prior_weights: Iterable[float] | None) -> float:
    """Squared distance from mixture weights to an optional prior."""

    if prior_weights is None:
        return 0.0
    w = np.asarray(tuple(weights), dtype=float)
    p = np.asarray(tuple(prior_weights), dtype=float)
    if len(w) != len(p):
        raise ValueError("prior_weights length must match mixture weights length")
    w = w / w.sum()
    p = p / p.sum()
    return float(np.sum((w - p) ** 2))


def regularization_report(
    params: MixtureGeneratorParams,
    config: RegularizationConfig | None = None,
) -> RegularizationReport:
    """Compute regularization diagnostics and signed objective terms."""

    config = config or RegularizationConfig()
    weights = params.weights
    means = params.means

    entropy = mixture_entropy(weights)
    max_entropy = float(np.log(len(weights))) if len(weights) > 1 else 0.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

    distances = pairwise_component_distances(means)
    if len(distances) == 0:
        avg_dist = 0.0
        min_dist = 0.0
        max_dist = 0.0
    else:
        avg_dist = float(distances.mean())
        min_dist = float(distances.min())
        max_dist = float(distances.max())

    prior_pen = prior_weight_penalty(weights, config.prior_weights)

    r_entropy = -entropy
    if config.dispersion_mode == "band":
        below = max(0.0, float(config.dispersion_min) - avg_dist)
        assert config.dispersion_max is not None
        above = max(0.0, avg_dist - float(config.dispersion_max))
        r_dispersion = below * below + above * above
    else:
        below = 0.0
        above = 0.0
        r_dispersion = -avg_dist
    r_prior = prior_pen
    objective = (
        float(config.entropy_weight) * r_entropy
        + float(config.dispersion_weight) * r_dispersion
        + float(config.prior_weight) * r_prior
    )

    terms = {
        "mixture_entropy": float(entropy),
        "normalized_mixture_entropy": float(norm_entropy),
        "avg_pairwise_mean_distance": float(avg_dist),
        "min_pairwise_mean_distance": float(min_dist),
        "max_pairwise_mean_distance": float(max_dist),
        "prior_weight_penalty": float(prior_pen),
        "R_entropy": float(r_entropy),
        "R_dispersion": float(r_dispersion),
        "dispersion_below_min": float(below),
        "dispersion_above_max": float(above),
        "R_prior": float(r_prior),
        "regularization_objective": float(objective),
    }
    return RegularizationReport(terms=terms, config=config)


def regularization_table(reports: dict[str, RegularizationReport]) -> pd.DataFrame:
    """Turn named regularization reports into a notebook-friendly table."""

    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        row: dict[str, Any] = {"candidate": name}
        row.update(report.terms)
        rows.append(row)
    return pd.DataFrame(rows)
