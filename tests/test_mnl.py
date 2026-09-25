import numpy as np
import pandas as pd

from eipg.econ import MNLConfig, MultinomialLogitModel, predict_probabilities_long


def _toy_choice_data() -> pd.DataFrame:
    rows = []
    # Four deterministic-ish observations with two alternatives.
    for i in range(12):
        # Alt 1 has better quality; choose it most of the time.
        rows.append(
            {
                "observation_id": f"obs_{i}",
                "alternative_id": 0,
                "chosen": 1 if i in {3, 7} else 0,
                "price": 1.0,
                "quality": 0.0,
            }
        )
        rows.append(
            {
                "observation_id": f"obs_{i}",
                "alternative_id": 1,
                "chosen": 0 if i in {3, 7} else 1,
                "price": 1.2,
                "quality": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_mnl_fit_returns_beta_and_scores():
    df = _toy_choice_data()
    model = MultinomialLogitModel(MNLConfig(features=("price", "quality"), l2=1e-4, max_iter=200))
    fit = model.fit(df)

    assert fit.success
    assert fit.n_observations == 12
    assert fit.n_rows == 24
    assert set(fit.beta_by_feature) == {"price", "quality"}
    assert np.isfinite(fit.train_nll)
    assert fit.train_nll_per_observation > 0


def test_mnl_probabilities_sum_to_one_per_observation():
    df = _toy_choice_data()
    beta = np.array([-0.5, 1.0])
    pred = predict_probabilities_long(df, beta=beta, features=("price", "quality"))

    sums = pred.groupby("observation_id")["mnl_prob"].sum().to_numpy()
    assert np.allclose(sums, 1.0)
    assert "mnl_utility" in pred.columns
