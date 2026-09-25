"""Estimated latent-class multinomial logit model.

This module implements a small finite-mixture MNL using expectation-
maximization (EM). Unlike :mod:`eipg.econ.segmented_mnl`, it never reads or
routes on ground-truth persona-segment labels.

v1.9.1 uses a *panel* latent-class likelihood. Class membership is persistent
within a persona, so the E-step aggregates all repeated choice observations for
that persona before computing the posterior class responsibility. For persona
``u`` and class ``k`` the model is

    P(y_u | x_u) = sum_k pi_k prod_t P_MNL(y_ut | x_ut, beta_k),

where ``t`` indexes repeated choice tasks. This is the appropriate likelihood
for the controlled benchmark because one sampled persona contributes multiple
choice observations.

Prediction remains ex-ante and population-level: target/held-out probabilities
are the class-prior weighted mixture of class-specific MNL probabilities. No
held-out realized choices are used to infer a persona's latent class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class _LongChoiceData:
    observation_ids: np.ndarray
    features: tuple[str, ...]
    x: np.ndarray
    chosen: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    row_observation_index: np.ndarray
    sorted_to_original: np.ndarray
    panel_ids: np.ndarray
    observation_panel_index: np.ndarray
    panel_id_column: str | None

    @property
    def n_observations(self) -> int:
        return int(len(self.observation_ids))

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_panels(self) -> int:
        return int(len(self.panel_ids))


@dataclass(frozen=True)
class LatentClassMNLFit:
    """Estimated finite-mixture MNL fit."""

    features: tuple[str, ...]
    class_weights: np.ndarray
    beta: np.ndarray
    log_likelihood: float
    penalized_log_likelihood: float
    n_observations: int
    n_panels: int
    panel_id_column: str | None
    n_classes: int
    em_iterations: int
    converged: bool
    selected_restart: int
    n_restarts: int
    l2: float
    seed: int

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_classes))

    @property
    def beta_by_class(self) -> dict[int, dict[str, float]]:
        return {
            k: {feature: float(self.beta[k, j]) for j, feature in enumerate(self.features)}
            for k in self.class_ids
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "estimated_panel_latent_class_mnl",
            "features": list(self.features),
            "n_classes": int(self.n_classes),
            "class_weights": [float(x) for x in self.class_weights],
            "beta_by_class": {
                str(k): values for k, values in self.beta_by_class.items()
            },
            "log_likelihood": float(self.log_likelihood),
            "penalized_log_likelihood": float(self.penalized_log_likelihood),
            "n_observations": int(self.n_observations),
            "n_panels": int(self.n_panels),
            "panel_id_column": self.panel_id_column,
            "em_iterations": int(self.em_iterations),
            "converged": bool(self.converged),
            "selected_restart": int(self.selected_restart),
            "n_restarts": int(self.n_restarts),
            "l2": float(self.l2),
            "seed": int(self.seed),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _prepare_long_choice_data(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    require_choice: bool,
    panel_id_column: str | None = None,
) -> _LongChoiceData:
    features = tuple(str(x) for x in features)
    required = {"observation_id", "alternative_id", *features}
    if require_choice:
        required.add("chosen")
    if panel_id_column is not None:
        required.add(panel_id_column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"latent-class MNL data missing required columns: {missing}")
    if df.empty:
        raise ValueError("latent-class MNL data must not be empty")

    codes, uniques = pd.factorize(df["observation_id"], sort=False)
    if (codes < 0).any():
        raise ValueError("observation_id must not contain missing values")
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    counts = np.bincount(sorted_codes, minlength=len(uniques)).astype(int)
    if (counts <= 0).any():
        raise ValueError("each observation must contain at least one alternative")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)

    x = df.loc[:, features].to_numpy(float)[order]
    if not np.isfinite(x).all():
        raise ValueError("latent-class MNL feature matrix contains non-finite values")

    if require_choice:
        chosen = df["chosen"].to_numpy(float)[order]
        chosen_per_obs = np.add.reduceat(chosen, starts)
        if not np.allclose(chosen_per_obs, 1.0):
            raise ValueError("each observation must have exactly one chosen alternative")
    else:
        chosen = np.zeros(len(df), dtype=float)

    row_obs = np.repeat(np.arange(len(uniques), dtype=int), counts)

    # Map each observation to one persistent panel/persona. When no panel column
    # is supplied, each observation is its own panel, reproducing the v1.9
    # observation-level latent mixture exactly.
    if panel_id_column is None:
        panel_ids = np.asarray(uniques)
        observation_panel_index = np.arange(len(uniques), dtype=int)
    else:
        panel_values_sorted = df[panel_id_column].to_numpy()[order]
        if pd.isna(panel_values_sorted).any():
            raise ValueError(f"{panel_id_column} must not contain missing values")

        observation_panel_values: list[Any] = []
        for start, length in zip(starts, counts):
            vals = panel_values_sorted[start : start + length]
            first = vals[0]
            if not np.all(vals == first):
                raise ValueError(
                    f"each observation_id must map to exactly one {panel_id_column}"
                )
            observation_panel_values.append(first)

        observation_panel_index, panel_ids = pd.factorize(
            np.asarray(observation_panel_values, dtype=object), sort=False
        )
        if (observation_panel_index < 0).any():
            raise ValueError(f"{panel_id_column} must not contain missing values")

    return _LongChoiceData(
        observation_ids=np.asarray(uniques),
        features=features,
        x=x,
        chosen=chosen,
        starts=starts,
        lengths=counts,
        row_observation_index=row_obs,
        sorted_to_original=order,
        panel_ids=np.asarray(panel_ids),
        observation_panel_index=np.asarray(observation_panel_index, dtype=int),
        panel_id_column=panel_id_column,
    )


def _softmax_rows_by_observation(utilities: np.ndarray, data: _LongChoiceData) -> np.ndarray:
    maxima = np.maximum.reduceat(utilities, data.starts)
    centered = utilities - np.repeat(maxima, data.lengths)
    exp_u = np.exp(centered)
    denom = np.add.reduceat(exp_u, data.starts)
    return exp_u / np.repeat(denom, data.lengths)


def _observation_log_likelihood(beta: np.ndarray, data: _LongChoiceData) -> np.ndarray:
    utilities = data.x @ beta
    maxima = np.maximum.reduceat(utilities, data.starts)
    centered = utilities - np.repeat(maxima, data.lengths)
    denom = np.add.reduceat(np.exp(centered), data.starts)
    log_denom = maxima + np.log(denom)
    chosen_utility = np.add.reduceat(utilities * data.chosen, data.starts)
    return chosen_utility - log_denom


def _panel_log_likelihood_matrix(
    class_observation_ll: np.ndarray,
    data: _LongChoiceData,
) -> np.ndarray:
    """Aggregate class-specific observation log likelihoods by panel."""

    out = np.zeros((data.n_panels, class_observation_ll.shape[1]), dtype=float)
    np.add.at(out, data.observation_panel_index, class_observation_ll)
    return out


def _weighted_mnl_objective(
    beta: np.ndarray,
    data: _LongChoiceData,
    observation_weights: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    utilities = data.x @ beta
    probabilities = _softmax_rows_by_observation(utilities, data)
    obs_ll = _observation_log_likelihood(beta, data)

    row_weights = observation_weights[data.row_observation_index]
    residual = data.chosen - probabilities
    grad_log_likelihood = data.x.T @ (row_weights * residual)

    nll = -float(np.dot(observation_weights, obs_ll))
    grad = -grad_log_likelihood
    if l2 > 0:
        nll += 0.5 * l2 * float(np.dot(beta, beta))
        grad = grad + l2 * beta
    return nll, np.asarray(grad, dtype=float)


def _fit_weighted_class(
    data: _LongChoiceData,
    observation_weights: np.ndarray,
    beta0: np.ndarray,
    *,
    l2: float,
    max_iter: int,
) -> np.ndarray:
    effective_n = float(np.sum(observation_weights))
    if effective_n <= 1e-8:
        return np.asarray(beta0, dtype=float).copy()

    result = minimize(
        fun=lambda b: _weighted_mnl_objective(b, data, observation_weights, l2),
        x0=np.asarray(beta0, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-7},
    )
    beta = np.asarray(result.x, dtype=float)
    if not np.isfinite(beta).all():
        raise RuntimeError(
            f"latent-class MNL M-step returned non-finite coefficients: {result.message}"
        )
    return beta


def _class_log_likelihood_matrix(beta: np.ndarray, data: _LongChoiceData) -> np.ndarray:
    return np.column_stack([
        _observation_log_likelihood(beta[k], data) for k in range(beta.shape[0])
    ])


def _observed_panel_log_likelihood(
    class_weights: np.ndarray,
    class_panel_ll: np.ndarray,
) -> float:
    return float(
        np.sum(logsumexp(np.log(class_weights)[None, :] + class_panel_ll, axis=1))
    )


def fit_latent_class_mnl(
    df: pd.DataFrame,
    config: Any,
    *,
    n_classes: int = 3,
    seed: int = 0,
    n_restarts: int = 6,
    em_max_iter: int = 60,
    em_tol: float = 1e-6,
    mstep_max_iter: int | None = None,
    init_scale: float = 0.8,
    min_class_weight: float = 1e-3,
    initial_beta: Sequence[float] | np.ndarray | None = None,
    panel_id_column: str | None = "persona_id",
) -> LatentClassMNLFit:
    """Estimate a panel finite-mixture MNL without true segment labels.

    ``panel_id_column`` identifies the entity whose latent class persists across
    repeated choice observations. For the controlled benchmark this is
    ``persona_id``. Set it to ``None`` to recover the old v1.9 observation-level
    mixture as a backwards-compatible diagnostic.
    """

    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    features = tuple(_config_value(config, "features", ()))
    if not features:
        raise ValueError("latent-class MNL requires a non-empty config.features")
    l2 = float(_config_value(config, "l2", 0.0))
    max_iter_cfg = int(_config_value(config, "max_iter", 200))
    mstep_max_iter = int(mstep_max_iter or min(max_iter_cfg, 120))

    data = _prepare_long_choice_data(
        df,
        features,
        require_choice=True,
        panel_id_column=panel_id_column,
    )
    p = len(features)
    base = (
        np.zeros(p, dtype=float)
        if initial_beta is None
        else np.asarray(initial_beta, dtype=float)
    )
    if base.shape != (p,):
        raise ValueError(f"initial_beta must have shape {(p,)}, got {base.shape}")

    rng_master = np.random.default_rng(seed)
    best: LatentClassMNLFit | None = None

    for restart in range(n_restarts):
        restart_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(restart_seed)
        scale = float(init_scale * (0.75 + 0.25 * restart))
        beta = np.vstack([
            base + rng.normal(0.0, scale, size=p) for _ in range(n_classes)
        ])
        class_weights = rng.dirichlet(np.ones(n_classes))
        class_weights = np.maximum(class_weights, min_class_weight)
        class_weights = class_weights / class_weights.sum()

        previous_ll = -np.inf
        converged = False
        final_iter = 0

        for em_iter in range(1, em_max_iter + 1):
            # E-step: combine all repeated choice tasks from the same persona
            # before computing class membership probabilities.
            class_obs_ll = _class_log_likelihood_matrix(beta, data)
            class_panel_ll = _panel_log_likelihood_matrix(class_obs_ll, data)
            log_joint = np.log(class_weights)[None, :] + class_panel_ll
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            panel_responsibilities = np.exp(log_joint - log_norm)

            # M-step: priors are panel proportions. Each observation inherits
            # its persona's posterior class responsibility.
            class_weights = panel_responsibilities.mean(axis=0)
            class_weights = np.maximum(class_weights, min_class_weight)
            class_weights = class_weights / class_weights.sum()

            for k in range(n_classes):
                observation_weights = panel_responsibilities[
                    data.observation_panel_index, k
                ]
                beta[k] = _fit_weighted_class(
                    data,
                    observation_weights,
                    beta[k],
                    l2=l2,
                    max_iter=mstep_max_iter,
                )

            class_obs_ll = _class_log_likelihood_matrix(beta, data)
            class_panel_ll = _panel_log_likelihood_matrix(class_obs_ll, data)
            observed_ll = _observed_panel_log_likelihood(class_weights, class_panel_ll)
            final_iter = em_iter

            if np.isfinite(previous_ll):
                improvement = observed_ll - previous_ll
                threshold = em_tol * (1.0 + abs(previous_ll))
                if abs(improvement) <= threshold:
                    converged = True
                    break
            previous_ll = observed_ll

        class_obs_ll = _class_log_likelihood_matrix(beta, data)
        class_panel_ll = _panel_log_likelihood_matrix(class_obs_ll, data)
        observed_ll = _observed_panel_log_likelihood(class_weights, class_panel_ll)
        penalized_ll = observed_ll - 0.5 * l2 * float(np.sum(beta * beta))

        candidate = LatentClassMNLFit(
            features=features,
            class_weights=np.asarray(class_weights, dtype=float).copy(),
            beta=np.asarray(beta, dtype=float).copy(),
            log_likelihood=float(observed_ll),
            penalized_log_likelihood=float(penalized_ll),
            n_observations=data.n_observations,
            n_panels=data.n_panels,
            panel_id_column=panel_id_column,
            n_classes=int(n_classes),
            em_iterations=int(final_iter),
            converged=bool(converged),
            selected_restart=int(restart),
            n_restarts=int(n_restarts),
            l2=float(l2),
            seed=int(seed),
        )
        if best is None or candidate.penalized_log_likelihood > best.penalized_log_likelihood:
            best = candidate

    assert best is not None
    return best


def predict_latent_class_mnl_long(
    df: pd.DataFrame,
    fit: LatentClassMNLFit,
    *,
    probability_column: str = "mnl_prob",
) -> pd.DataFrame:
    """Predict ex-ante population choice probabilities from the fitted mixture.

    Prediction uses class priors rather than target choices/posteriors, avoiding
    leakage from anchor or held-out realized outcomes.
    """

    data = _prepare_long_choice_data(
        df,
        fit.features,
        require_choice=False,
        panel_id_column=None,
    )
    mixture_sorted = np.zeros(data.n_rows, dtype=float)
    for k in fit.class_ids:
        utilities = data.x @ fit.beta[k]
        class_probs = _softmax_rows_by_observation(utilities, data)
        mixture_sorted += fit.class_weights[k] * class_probs

    mixture_original = np.empty_like(mixture_sorted)
    mixture_original[data.sorted_to_original] = mixture_sorted

    out = df.copy()
    out[probability_column] = mixture_original
    return out


def latent_class_coefficient_table(fit: LatentClassMNLFit) -> pd.DataFrame:
    """Return one tidy row per estimated class-feature coefficient."""

    rows: list[dict[str, Any]] = []
    for class_id in fit.class_ids:
        for j, feature in enumerate(fit.features):
            rows.append(
                {
                    "class_id": int(class_id),
                    "feature": feature,
                    "beta": float(fit.beta[class_id, j]),
                    "class_weight": float(fit.class_weights[class_id]),
                    "n_observations": int(fit.n_observations),
                    "n_panels": int(fit.n_panels),
                    "panel_id_column": fit.panel_id_column,
                    "log_likelihood": float(fit.log_likelihood),
                    "em_iterations": int(fit.em_iterations),
                    "selected_restart": int(fit.selected_restart),
                }
            )
    return pd.DataFrame(rows)


def latent_class_panel_posteriors(
    df: pd.DataFrame,
    fit: LatentClassMNLFit,
    *,
    panel_id_column: str | None = None,
) -> pd.DataFrame:
    """Return fitted posterior class probabilities for the training panels.

    This is intended for controlled-benchmark diagnostics only. It requires the
    realized training choices and must not be used to score held-out target
    choices.
    """

    panel_col = panel_id_column if panel_id_column is not None else fit.panel_id_column
    data = _prepare_long_choice_data(
        df,
        fit.features,
        require_choice=True,
        panel_id_column=panel_col,
    )
    class_obs_ll = _class_log_likelihood_matrix(fit.beta, data)
    class_panel_ll = _panel_log_likelihood_matrix(class_obs_ll, data)
    log_joint = np.log(fit.class_weights)[None, :] + class_panel_ll
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    rows: list[dict[str, Any]] = []
    for panel_idx, panel_id in enumerate(data.panel_ids):
        best_class = int(np.argmax(posterior[panel_idx]))
        row: dict[str, Any] = {
            "panel_id": panel_id,
            "estimated_class_id": best_class,
            "max_posterior": float(posterior[panel_idx, best_class]),
        }
        for k in fit.class_ids:
            row[f"class_{k}_posterior"] = float(posterior[panel_idx, k])
        rows.append(row)
    return pd.DataFrame(rows)
