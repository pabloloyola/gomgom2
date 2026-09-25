from __future__ import annotations

import numpy as np
import pandas as pd

from eipg.econ import (
    MNLConfig,
    fit_latent_class_mnl,
    latent_class_panel_posteriors,
    predict_latent_class_mnl_long,
)


def _toy_panel_data(n_personas: int = 12, tasks_per_persona: int = 5) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    obs_idx = 0
    for persona_id in range(n_personas):
        latent = persona_id % 2
        for _ in range(tasks_per_persona):
            obs = f"o{obs_idx}"
            obs_idx += 1
            # Two alternatives with varying price/quality tradeoffs.
            price1 = 1.0 + rng.uniform(0.0, 1.0)
            quality1 = rng.uniform(0.0, 1.0)
            price0 = 1.0 + rng.uniform(0.0, 1.0)
            quality0 = rng.uniform(0.0, 1.0)
            utilities = np.array([
                (-2.5 * price0 + 0.4 * quality0) if latent == 0 else (-0.4 * price0 + 2.5 * quality0),
                (-2.5 * price1 + 0.4 * quality1) if latent == 0 else (-0.4 * price1 + 2.5 * quality1),
            ])
            chosen_alt = int(np.argmax(utilities))
            specs = [(price0, quality0), (price1, quality1)]
            for alt, (price, quality) in enumerate(specs):
                rows.append(
                    {
                        "observation_id": obs,
                        "persona_id": persona_id,
                        "alternative_id": alt,
                        "chosen": int(alt == chosen_alt),
                        "price": price,
                        "quality": quality,
                    }
                )
    return pd.DataFrame(rows)


def test_panel_latent_class_uses_one_latent_membership_per_persona() -> None:
    df = _toy_panel_data()
    cfg = MNLConfig(features=("price", "quality"), l2=1e-3, max_iter=150)
    fit = fit_latent_class_mnl(
        df,
        cfg,
        n_classes=2,
        seed=3,
        n_restarts=2,
        em_max_iter=20,
        panel_id_column="persona_id",
    )
    assert fit.n_panels == df["persona_id"].nunique()
    assert fit.n_observations == df["observation_id"].nunique()
    assert fit.panel_id_column == "persona_id"
    assert np.isclose(fit.class_weights.sum(), 1.0)


def test_panel_posteriors_are_one_row_per_persona_and_normalized() -> None:
    df = _toy_panel_data()
    cfg = MNLConfig(features=("price", "quality"), l2=1e-3, max_iter=150)
    fit = fit_latent_class_mnl(
        df,
        cfg,
        n_classes=2,
        seed=4,
        n_restarts=2,
        em_max_iter=20,
        panel_id_column="persona_id",
    )
    posterior = latent_class_panel_posteriors(df, fit)
    assert len(posterior) == df["persona_id"].nunique()
    probs = posterior[["class_0_posterior", "class_1_posterior"]].to_numpy(float)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_prediction_is_ex_ante_mixture_and_normalizes_by_observation() -> None:
    df = _toy_panel_data()
    cfg = MNLConfig(features=("price", "quality"), l2=1e-3, max_iter=150)
    fit = fit_latent_class_mnl(
        df,
        cfg,
        n_classes=2,
        seed=5,
        n_restarts=2,
        em_max_iter=20,
        panel_id_column="persona_id",
    )
    pred = predict_latent_class_mnl_long(df, fit)
    sums = pred.groupby("observation_id")["mnl_prob"].sum().to_numpy(float)
    assert np.allclose(sums, 1.0)


def test_observation_level_fallback_remains_available() -> None:
    df = _toy_panel_data(n_personas=6, tasks_per_persona=3)
    cfg = MNLConfig(features=("price", "quality"), l2=1e-3, max_iter=100)
    fit = fit_latent_class_mnl(
        df,
        cfg,
        n_classes=2,
        seed=6,
        n_restarts=1,
        em_max_iter=5,
        panel_id_column=None,
    )
    assert fit.n_panels == fit.n_observations
    assert fit.panel_id_column is None
