"""Multinomial logit inner model ``m_beta``.

Paper alignment
---------------
The EIPG inner loop fits an economic model to the simulated dataset induced by
one candidate generator ``G_phi``:

    beta*(phi) = argmin_beta L_econ(beta; D_phi)

This module implements the first clean version of that inner model: a standard
multinomial logit (MNL) over long-format choice data.  Each observation is a
choice slate with one chosen alternative and one row per available alternative.
The model assigns systematic utility

    v_ij = beta^T f(x_i, j)

and choice probabilities

    P(y_i = j | x_i) = exp(v_ij) / sum_k exp(v_ik).

The simulator generates ``D_phi``; this MNL model is fitted after that.  The MNL
therefore summarizes the behavior induced by the generator and simulator rather
than directly generating the synthetic choices used for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


DEFAULT_MNL_FEATURES: tuple[str, ...] = (
    "price",
    "quality",
    "sustain",
    "novelty",
    "brand",
)


@dataclass(frozen=True)
class MNLConfig:
    """Configuration for fitting the inner MNL model."""

    features: tuple[str, ...] = DEFAULT_MNL_FEATURES
    l2: float = 1.0e-4
    max_iter: int = 300

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("features must be non-empty")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")

    @classmethod
    def from_config(cls, section: dict[str, Any], *, default_features: Iterable[str] | None = None) -> "MNLConfig":
        features = tuple(section.get("features") or tuple(default_features or DEFAULT_MNL_FEATURES))
        return cls(
            features=features,
            l2=float(section.get("l2", 1.0e-4)),
            max_iter=int(section.get("max_iter", 300)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "l2": float(self.l2),
            "max_iter": int(self.max_iter),
        }


@dataclass(frozen=True)
class MNLFitResult:
    """Fitted MNL parameters and fit diagnostics."""

    beta: np.ndarray
    features: tuple[str, ...]
    train_nll: float
    train_nll_per_observation: float
    l2: float
    n_observations: int
    n_rows: int
    success: bool
    message: str
    n_iter: int

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=float)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "features", tuple(self.features))
        if beta.ndim != 1:
            raise ValueError("beta must be 1-D")
        if len(beta) != len(self.features):
            raise ValueError("beta length must match number of features")

    @property
    def beta_by_feature(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.features, self.beta)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "mnl",
            "features": list(self.features),
            "beta": [float(v) for v in self.beta],
            "beta_by_feature": self.beta_by_feature,
            "train_nll": float(self.train_nll),
            "train_nll_per_observation": float(self.train_nll_per_observation),
            "l2": float(self.l2),
            "n_observations": int(self.n_observations),
            "n_rows": int(self.n_rows),
            "success": bool(self.success),
            "message": str(self.message),
            "n_iter": int(self.n_iter),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


class MultinomialLogitModel:
    """Fit and evaluate a simple MNL model on long-format choice data."""

    def __init__(self, config: MNLConfig):
        self.config = config

    def fit(self, df: pd.DataFrame) -> MNLFitResult:
        """Fit beta by minimizing regularized negative log-likelihood."""

        _validate_long_choice_data(df, self.config.features)
        n_obs = int(df["observation_id"].nunique())
        x_blocks, y_blocks = _blocks_from_long_df(df, self.config.features)
        beta0 = np.zeros(len(self.config.features), dtype=float)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            return _nll_and_grad(beta, x_blocks, y_blocks, l2=self.config.l2)

        result = minimize(
            fun=lambda b: objective(b)[0],
            x0=beta0,
            jac=lambda b: objective(b)[1],
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iter},
        )
        train_nll = negative_log_likelihood_long(
            df,
            beta=result.x,
            features=self.config.features,
            include_l2=False,
        )
        return MNLFitResult(
            beta=np.asarray(result.x, dtype=float),
            features=self.config.features,
            train_nll=float(train_nll),
            train_nll_per_observation=float(train_nll / max(n_obs, 1)),
            l2=float(self.config.l2),
            n_observations=n_obs,
            n_rows=int(len(df)),
            success=bool(result.success),
            message=str(result.message),
            n_iter=int(result.nit),
        )


class FittedMNL:
    """Convenience wrapper for a fitted MNL result."""

    def __init__(self, fit: MNLFitResult):
        self.fit = fit

    def predict_long(self, df: pd.DataFrame) -> pd.DataFrame:
        return predict_probabilities_long(df, beta=self.fit.beta, features=self.fit.features)

    def nll(self, df: pd.DataFrame) -> float:
        return negative_log_likelihood_long(df, beta=self.fit.beta, features=self.fit.features)

    def nll_per_observation(self, df: pd.DataFrame) -> float:
        n_obs = max(int(df["observation_id"].nunique()), 1)
        return float(self.nll(df) / n_obs)

    def accuracy(self, df: pd.DataFrame) -> float:
        pred = self.predict_long(df)
        rows: list[bool] = []
        for _, g in pred.groupby("observation_id", sort=False):
            pred_alt = g.loc[g["mnl_prob"].idxmax(), "alternative_id"]
            chosen = g.loc[g["chosen"].astype(int) == 1, "alternative_id"]
            if len(chosen) != 1:
                continue
            rows.append(pred_alt == chosen.iloc[0])
        if not rows:
            return float("nan")
        return float(np.mean(rows))


def _validate_long_choice_data(df: pd.DataFrame, features: Iterable[str]) -> None:
    required = {"observation_id", "alternative_id", "chosen"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"choice data missing required columns: {missing}")
    features = tuple(features)
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"choice data missing model feature columns: {missing_features}")
    if df.empty:
        raise ValueError("choice data must be non-empty")
    chosen_counts = df.groupby("observation_id")["chosen"].sum()
    bad = chosen_counts[chosen_counts.astype(float) != 1.0]
    if len(bad) > 0:
        examples = bad.head().to_dict()
        raise ValueError(f"each observation must have exactly one chosen row; examples={examples}")


def _blocks_from_long_df(df: pd.DataFrame, features: Iterable[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for _, g in df.groupby("observation_id", sort=False):
        x_blocks.append(g[list(features)].to_numpy(dtype=float))
        y_blocks.append(g["chosen"].to_numpy(dtype=float))
    return x_blocks, y_blocks


def _nll_and_grad(
    beta: np.ndarray,
    x_blocks: list[np.ndarray],
    y_blocks: list[np.ndarray],
    *,
    l2: float = 0.0,
) -> tuple[float, np.ndarray]:
    beta = np.asarray(beta, dtype=float)
    nll = 0.0
    grad = np.zeros_like(beta)
    for x, y in zip(x_blocks, y_blocks):
        utilities = x @ beta
        log_denom = logsumexp(utilities)
        probs = np.exp(utilities - log_denom)
        nll -= float(y @ utilities - log_denom)
        grad += x.T @ (probs - y)
    if l2 > 0:
        nll += 0.5 * float(l2) * float(beta @ beta)
        grad += float(l2) * beta
    return float(nll), grad


def negative_log_likelihood_long(
    df: pd.DataFrame,
    *,
    beta: np.ndarray,
    features: Iterable[str],
    include_l2: bool = False,
    l2: float = 0.0,
) -> float:
    """Compute MNL negative log-likelihood on long-format choice data."""

    _validate_long_choice_data(df, features)
    x_blocks, y_blocks = _blocks_from_long_df(df, features)
    nll, _ = _nll_and_grad(np.asarray(beta, dtype=float), x_blocks, y_blocks, l2=l2 if include_l2 else 0.0)
    return float(nll)


def predict_probabilities_long(
    df: pd.DataFrame,
    *,
    beta: np.ndarray,
    features: Iterable[str],
) -> pd.DataFrame:
    """Add MNL utilities and probabilities to long-format choice data."""

    _validate_long_choice_data(df, features)
    beta = np.asarray(beta, dtype=float)
    features = tuple(features)
    records: list[pd.DataFrame] = []
    for _, g in df.groupby("observation_id", sort=False):
        out = g.copy()
        utilities = out[list(features)].to_numpy(dtype=float) @ beta
        log_denom = logsumexp(utilities)
        out["mnl_utility"] = utilities
        out["mnl_prob"] = np.exp(utilities - log_denom)
        records.append(out)
    return pd.concat(records, ignore_index=True)


def evaluate_fitted_mnl(fit: MNLFitResult, datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Evaluate one fitted MNL on named datasets."""

    model = FittedMNL(fit)
    out: dict[str, Any] = {
        "model": "mnl",
        "features": list(fit.features),
        "beta_by_feature": fit.beta_by_feature,
        "datasets": {},
    }
    for name, df in datasets.items():
        n_obs = int(df["observation_id"].nunique())
        out["datasets"][name] = {
            "n_observations": n_obs,
            "n_rows": int(len(df)),
            "nll": float(model.nll(df)),
            "nll_per_observation": float(model.nll_per_observation(df)),
            "top_choice_accuracy": float(model.accuracy(df)),
        }
    return out
