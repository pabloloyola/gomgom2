import numpy as np
import pandas as pd

from eipg.outeropt import EvolutionSearchConfig, mutate_generator_params, run_evolutionary_search
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig
from eipg.econ import MNLConfig
from eipg.objectives import CalibrationMomentConfig, RegularizationConfig


def _tiny_contexts():
    rows = []
    for ctx in range(6):
        for alt in range(3):
            rows.append(
                {
                    "context_id": f"ctx_{ctx}",
                    "alternative_id": alt,
                    "price": float(1 + alt),
                    "quality": float(alt),
                    "sustain": float(ctx % 2),
                    "novelty": float((ctx + alt) % 2),
                    "brand": float(alt == 2),
                }
            )
    return pd.DataFrame(rows)


def _params():
    return MixtureGeneratorParams(
        weights=np.array([0.5, 0.5]),
        means=np.array([[-1.5, 0.5, 0.0, 0.0, 0.0], [-0.5, 1.2, 0.5, 0.0, 0.5]]),
        features=("price", "quality", "sustain", "novelty", "brand"),
        within_component_std=0.1,
        segment_labels=("a", "b"),
    )


def test_mutate_generator_params_preserves_shape_and_normalization():
    rng = np.random.default_rng(123)
    mutated = mutate_generator_params(
        _params(), rng=rng, mean_scale=0.2, weight_logit_scale=0.1, mean_clip=3.0
    )
    assert mutated.means.shape == (2, 5)
    assert mutated.weights.shape == (2,)
    assert np.isclose(mutated.weights.sum(), 1.0)
    assert np.max(np.abs(mutated.means)) <= 3.0


def test_run_evolutionary_search_smoke():
    params = _params()
    contexts = _tiny_contexts()
    personas = MixturePersonaGenerator(params, seed=1).sample(10)
    sim_cfg = SyntheticSimulatorConfig(choice_temperature=1.0)
    simulator = RandomUtilityChoiceSimulator(sim_cfg, seed=2)
    d_h = simulator.simulate_long_dataset(contexts=contexts, personas=personas, n_observations=12, dataset_label="D_H")
    d_cal = simulator.simulate_long_dataset(contexts=contexts, personas=personas, n_observations=12, dataset_label="D_cal")
    d_cf = simulator.simulate_long_dataset(contexts=contexts, personas=personas, n_observations=12, dataset_label="D_cf")

    result = run_evolutionary_search(
        initial_params=params,
        x_sim=contexts,
        d_h=d_h,
        d_calib_int=d_cal,
        d_cf=d_cf,
        simulator_config=sim_cfg,
        mnl_config=MNLConfig(features=params.features, max_iter=50),
        calibration_config=CalibrationMomentConfig(attribute_features=params.features),
        regularization_config=RegularizationConfig(entropy_weight=0.01, dispersion_weight=0.01),
        search_config=EvolutionSearchConfig(budget=2, population_size=2),
        seed=99,
        n_personas=10,
        n_observations=12,
    )
    assert len(result.evaluations) == 4
    assert result.best.objective_value == min(ev.objective_value for ev in result.evaluations)
    assert not result.history.empty
