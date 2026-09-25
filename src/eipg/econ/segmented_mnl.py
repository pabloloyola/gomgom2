"""Oracle segmented multinomial logit diagnostics.

This module is benchmark-only. In the controlled synthetic experiment, each
simulated observation carries the ground-truth mixture component that generated
its persona. We exploit those labels in two different ways:

1. ``predict_oracle_segmented_mnl_long`` routes each target choice occasion
   through the MNL associated with its *true target segment*. This is a strongly
   privileged upper-bound diagnostic.
2. ``predict_oracle_mixture_mnl_long`` uses the same oracle-fitted segment MNLs
   but withholds target segment labels and averages them using training-population
   segment weights. This is the fair comparison to an estimated latent-class
   MNL whose target classes are also unobserved.

Neither path is intended as a deployable estimator because the training segment
labels are known only in the controlled benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json

import numpy as np
import pandas as pd

from eipg.econ.mnl import MNLConfig, MNLFitResult, MultinomialLogitModel, predict_probabilities_long


@dataclass(frozen=True)
class OracleSegmentedMNLFit:
    """Collection of segment-specific MNL fits using known synthetic labels."""

    segment_column: str
    fits: dict[int, MNLFitResult]
    observation_weights: dict[int, float]
    persona_weights: dict[int, float] | None = None
    persona_id_column: str | None = None

    @property
    def segment_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.fits))

    def mixture_weights(self, basis: Literal["persona", "observation"] = "persona") -> dict[int, float]:
        """Return normalized segment weights for unlabeled-target mixture prediction.

        Persona-level weights are preferable because the latent segment belongs
        to a persona rather than to an individual choice occasion. For datasets
        without a persona identifier, the function falls back to observation
        weights.
        """

        if basis == "persona" and self.persona_weights:
            raw = self.persona_weights
        elif basis in {"persona", "observation"}:
            raw = self.observation_weights
        else:
            raise ValueError("basis must be 'persona' or 'observation'")

        weights = {int(k): float(raw[k]) for k in self.segment_ids}
        total = float(sum(weights.values()))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("oracle mixture weights must have positive finite mass")
        return {k: v / total for k, v in weights.items()}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "oracle_segmented_mnl",
            "segment_column": self.segment_column,
            "segment_ids": list(self.segment_ids),
            "observation_weights": {
                str(k): float(self.observation_weights[k]) for k in self.segment_ids
            },
            "mixture_weights_persona_or_fallback": {
                str(k): float(v) for k, v in self.mixture_weights("persona").items()
            },
            "segment_fits": {
                str(k): self.fits[k].to_dict() for k in self.segment_ids
            },
        }
        if self.persona_weights is not None:
            payload["persona_weights"] = {
                str(k): float(self.persona_weights[k]) for k in self.segment_ids
            }
            payload["persona_id_column"] = self.persona_id_column
        return payload

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def _observation_segment_table(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    required = {"observation_id", segment_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"segmented MNL data missing required columns: {missing}")
    obs = df[["observation_id", segment_column]].drop_duplicates()
    counts = obs.groupby("observation_id")[segment_column].nunique()
    if (counts != 1).any():
        bad = counts[counts != 1].head().to_dict()
        raise ValueError(f"each observation must belong to exactly one segment; examples={bad}")
    return obs


def _persona_segment_weights(
    df: pd.DataFrame,
    *,
    segment_column: str,
    persona_id_column: str,
) -> dict[int, float] | None:
    if persona_id_column not in df.columns:
        return None

    persona = df[[persona_id_column, segment_column]].drop_duplicates()
    counts = persona.groupby(persona_id_column)[segment_column].nunique()
    if (counts != 1).any():
        bad = counts[counts != 1].head().to_dict()
        raise ValueError(
            "each persona must belong to exactly one segment; "
            f"examples={bad}"
        )

    n_personas = int(persona[persona_id_column].nunique())
    if n_personas <= 0:
        return None
    proportions = persona[segment_column].value_counts(normalize=True)
    return {int(k): float(v) for k, v in proportions.items()}


def fit_oracle_segmented_mnl(
    df: pd.DataFrame,
    config: MNLConfig,
    *,
    segment_column: str = "persona_segment_id",
    persona_id_column: str = "persona_id",
) -> OracleSegmentedMNLFit:
    """Fit one MNL per known ground-truth persona segment."""

    obs = _observation_segment_table(df, segment_column)
    total_obs = max(int(obs["observation_id"].nunique()), 1)

    fits: dict[int, MNLFitResult] = {}
    weights: dict[int, float] = {}
    model = MultinomialLogitModel(config)

    for raw_segment in sorted(obs[segment_column].dropna().unique().tolist()):
        segment_id = int(raw_segment)
        observation_ids = obs.loc[
            obs[segment_column] == raw_segment, "observation_id"
        ]
        segment_df = df[df["observation_id"].isin(observation_ids)].copy()
        if segment_df.empty:
            continue
        fits[segment_id] = model.fit(segment_df)
        weights[segment_id] = float(len(observation_ids) / total_obs)

    if not fits:
        raise ValueError("no non-empty segments were available for segmented MNL fitting")

    persona_weights = _persona_segment_weights(
        df,
        segment_column=segment_column,
        persona_id_column=persona_id_column,
    )
    if persona_weights is not None:
        # Keep only fitted segments and renormalize in case a degenerate segment
        # was absent from the fitting sample.
        persona_weights = {
            k: float(persona_weights.get(k, 0.0)) for k in sorted(fits)
        }
        total = float(sum(persona_weights.values()))
        if total > 0.0:
            persona_weights = {k: v / total for k, v in persona_weights.items()}
        else:
            persona_weights = None

    return OracleSegmentedMNLFit(
        segment_column=segment_column,
        fits=fits,
        observation_weights=weights,
        persona_weights=persona_weights,
        persona_id_column=persona_id_column if persona_weights is not None else None,
    )


def predict_oracle_segmented_mnl_long(
    df: pd.DataFrame,
    fit: OracleSegmentedMNLFit,
) -> pd.DataFrame:
    """Predict each target observation using its *known* ground-truth segment.

    This path intentionally uses target segment labels and therefore represents
    a privileged routed upper bound.
    """

    obs = _observation_segment_table(df, fit.segment_column)
    outputs: list[pd.DataFrame] = []

    for raw_segment in sorted(obs[fit.segment_column].dropna().unique().tolist()):
        segment_id = int(raw_segment)
        if segment_id not in fit.fits:
            raise ValueError(
                f"target contains segment {segment_id} but no segment-specific MNL was fitted"
            )
        observation_ids = obs.loc[
            obs[fit.segment_column] == raw_segment, "observation_id"
        ]
        segment_df = df[df["observation_id"].isin(observation_ids)].copy()
        segment_fit = fit.fits[segment_id]
        pred = predict_probabilities_long(
            segment_df,
            beta=segment_fit.beta,
            features=segment_fit.features,
        )
        pred["oracle_segment_id"] = segment_id
        outputs.append(pred)

    if not outputs:
        raise ValueError("target data produced no segmented predictions")

    out = pd.concat(outputs, ignore_index=True)
    order = {obs_id: idx for idx, obs_id in enumerate(df["observation_id"].drop_duplicates())}
    out["_obs_order"] = out["observation_id"].map(order)
    out = out.sort_values(["_obs_order", "alternative_id"], kind="stable").drop(columns="_obs_order")
    return out.reset_index(drop=True)


def predict_oracle_mixture_mnl_long(
    df: pd.DataFrame,
    fit: OracleSegmentedMNLFit,
    *,
    weight_basis: Literal["persona", "observation"] = "persona",
    probability_column: str = "mnl_prob",
) -> pd.DataFrame:
    """Predict targets without using target segment labels.

    Each segment-specific oracle MNL predicts every target choice occasion and
    the resulting probabilities are averaged using segment weights estimated
    from the *training* synthetic population. This keeps oracle training labels
    but removes the privileged target-routing information.
    """

    weights = fit.mixture_weights(weight_basis)
    mixture = np.zeros(len(df), dtype=float)

    for segment_id in fit.segment_ids:
        segment_fit = fit.fits[segment_id]
        pred = predict_probabilities_long(
            df,
            beta=segment_fit.beta,
            features=segment_fit.features,
        )
        if probability_column not in pred.columns:
            raise ValueError(
                f"segment prediction missing probability column {probability_column!r}"
            )
        if len(pred) != len(df):
            raise ValueError("segment prediction changed row count during oracle-mixture prediction")
        probs = pred[probability_column].to_numpy(float)
        if not np.isfinite(probs).all():
            raise ValueError("oracle-mixture component produced non-finite probabilities")
        mixture += float(weights[segment_id]) * probs

    out = df.copy()
    out[probability_column] = mixture
    out["oracle_mixture_weight_basis"] = weight_basis

    sums = out.groupby("observation_id", sort=False)[probability_column].sum().to_numpy(float)
    if not np.allclose(sums, 1.0, atol=1e-7):
        raise ValueError("oracle-mixture probabilities do not sum to one within observations")
    return out


def segmented_coefficient_table(fit: OracleSegmentedMNLFit) -> pd.DataFrame:
    """Return one tidy row per segment-feature coefficient."""

    persona_weights = fit.mixture_weights("persona")
    rows: list[dict[str, Any]] = []
    for segment_id in fit.segment_ids:
        segment_fit = fit.fits[segment_id]
        for feature, beta in segment_fit.beta_by_feature.items():
            rows.append(
                {
                    "segment_id": int(segment_id),
                    "feature": feature,
                    "beta": float(beta),
                    "segment_persona_weight": float(persona_weights[segment_id]),
                    "segment_observation_weight": float(fit.observation_weights[segment_id]),
                    "n_observations": int(segment_fit.n_observations),
                    "train_nll_per_observation": float(segment_fit.train_nll_per_observation),
                }
            )
    return pd.DataFrame(rows)
