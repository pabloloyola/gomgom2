from eipg.personas.anchored_mu import (
    FEATURES,
    coordinate_neighbor,
    initial_anchored_personas,
    neighborhood,
    render_population,
)


def test_zero_mu_renders_exact_original_prompts():
    personas = initial_anchored_personas()
    rendered = render_population(personas)
    assert len(rendered) == 3
    for structured, text_persona in zip(personas, rendered):
        assert text_persona.prompt == structured.base_prompt


def test_coordinate_neighbor_changes_exactly_one_value():
    personas = initial_anchored_personas()
    candidate = coordinate_neighbor(
        personas,
        persona_index=1,
        feature_index=FEATURES.index("quality"),
        delta=0.5,
        lower=-1.5,
        upper=1.5,
    )
    assert candidate[1].as_dict()["quality"] == 0.5
    assert candidate[0] == personas[0]
    assert candidate[2] == personas[2]


def test_nonzero_mu_appends_relative_adjustment():
    personas = initial_anchored_personas()
    candidate = coordinate_neighbor(
        personas,
        persona_index=0,
        feature_index=FEATURES.index("price"),
        delta=0.5,
        lower=-1.5,
        upper=1.5,
    )
    rendered = render_population(candidate)
    assert rendered[0].prompt.startswith(personas[0].base_prompt)
    assert "relative to that baseline description" in rendered[0].prompt
    assert "more price-sensitive" in rendered[0].prompt


def test_neighborhood_has_plus_minus_for_all_coordinates_at_origin():
    personas = initial_anchored_personas()
    neighbors = neighborhood(
        personas,
        step=0.5,
        lower=-1.5,
        upper=1.5,
    )
    assert len(neighbors) == len(personas) * len(FEATURES) * 2
