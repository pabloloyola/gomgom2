import numpy as np

from eipg.personas import MixturePersonaGenerator, PersonaRenderer, params_from_config


def _section():
    return {
        "k_components": 3,
        "latent_dim": 5,
        "features": ["price", "quality", "sustain", "novelty", "brand"],
        "within_component_std": 0.30,
        "init": "paper_smoke",
    }


def test_params_from_config_shape_and_weights():
    params = params_from_config(_section())
    assert params.k_components == 3
    assert params.latent_dim == 5
    assert params.means.shape == (3, 5)
    np.testing.assert_allclose(params.weights.sum(), 1.0)
    assert params.features == ("price", "quality", "sustain", "novelty", "brand")


def test_sampling_is_reproducible():
    params = params_from_config(_section())
    g1 = MixturePersonaGenerator(params, seed=123)
    g2 = MixturePersonaGenerator(params, seed=123)
    p1 = g1.sample(5)
    p2 = g2.sample(5)
    assert [p.segment_id for p in p1] == [p.segment_id for p in p2]
    np.testing.assert_allclose(np.stack([p.z for p in p1]), np.stack([p.z for p in p2]))


def test_prototypes_and_renderer():
    params = params_from_config(_section())
    g = MixturePersonaGenerator(params, seed=0)
    prototypes = g.prototypes()
    assert len(prototypes) == 3
    text = PersonaRenderer().render_profile(prototypes[0])
    assert "You are" in text
    assert "price" in text or "prices" in text
