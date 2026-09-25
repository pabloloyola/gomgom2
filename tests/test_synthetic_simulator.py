import numpy as np
import pandas as pd

from eipg.personas.schema import PersonaLatent
from eipg.simulators.synthetic import (
    RandomUtilityChoiceSimulator,
    SyntheticSimulatorConfig,
    summarize_choice_dataset,
)


def test_choice_probabilities_favor_lower_price_for_price_sensitive_persona():
    slate = pd.DataFrame(
        {
            "context_set": ["X_sim", "X_sim"],
            "context_id": ["ctx_0", "ctx_0"],
            "alternative_id": [0, 1],
            "product_id": ["item_00", "item_01"],
            "price": [0.5, 2.0],
            "quality": [0.5, 0.5],
            "sustain": [0.0, 0.0],
            "novelty": [0.0, 0.0],
            "brand": [0.0, 0.0],
        }
    )
    persona = PersonaLatent(
        persona_id=0,
        segment_id=0,
        segment_label="budget_sensitive",
        z=np.array([-2.0, 0.0, 0.0, 0.0, 0.0]),
        features=("price", "quality", "sustain", "novelty", "brand"),
    )
    sim = RandomUtilityChoiceSimulator(SyntheticSimulatorConfig(choice_temperature=1.0), seed=1)
    probs = sim.choice_probabilities(slate, persona)
    low_price_prob = float(probs.loc[probs["alternative_id"] == 0, "choice_prob"].iloc[0])
    high_price_prob = float(probs.loc[probs["alternative_id"] == 1, "choice_prob"].iloc[0])
    assert low_price_prob > high_price_prob


def test_simulate_long_dataset_has_one_choice_per_observation():
    contexts = pd.DataFrame(
        {
            "context_set": ["X_sim", "X_sim", "X_sim", "X_sim"],
            "context_id": ["ctx_0", "ctx_0", "ctx_1", "ctx_1"],
            "alternative_id": [0, 1, 0, 1],
            "product_id": ["item_00", "item_01", "item_00", "item_01"],
            "price": [0.5, 1.5, 0.8, 1.2],
            "quality": [0.5, 0.5, 0.4, 0.9],
            "sustain": [0.0, 0.0, 0.0, 0.0],
            "novelty": [0.0, 0.0, 0.0, 0.0],
            "brand": [0.0, 0.0, 0.0, 0.0],
        }
    )
    persona = PersonaLatent(
        persona_id=0,
        segment_id=0,
        segment_label="budget_sensitive",
        z=np.array([-1.0, 1.0, 0.0, 0.0, 0.0]),
        features=("price", "quality", "sustain", "novelty", "brand"),
    )
    sim = RandomUtilityChoiceSimulator(seed=2)
    df = sim.simulate_long_dataset(
        contexts=contexts,
        personas=[persona],
        n_observations=10,
        dataset_label="D_test",
    )
    assert df["observation_id"].nunique() == 10
    assert len(df) == 20
    chosen_counts = df.groupby("observation_id")["chosen"].sum()
    assert chosen_counts.eq(1).all()
    summary = summarize_choice_dataset(df)
    assert summary["n_observations"] == 10
    assert summary["n_chosen_rows"] == 10
