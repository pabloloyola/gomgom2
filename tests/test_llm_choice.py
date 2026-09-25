from eipg.simulators.llm_choice import TextPersonaChoiceSimulator


def test_parse_choice_accepts_common_provider_wrappers():
    valid = [0, 1, 2]
    assert TextPersonaChoiceSimulator.parse_choice('{"alternative_id": 1}', valid) == 1
    assert TextPersonaChoiceSimulator.parse_choice('```json\n{"alternative_id": 2}\n```', valid) == 2
    assert TextPersonaChoiceSimulator.parse_choice('alternative_id: 0', valid) == 0
    assert TextPersonaChoiceSimulator.parse_choice('Alternative = 2', valid) == 2
    assert TextPersonaChoiceSimulator.parse_choice('1', valid) == 1


def test_parse_choice_rejects_freeform_reasoning_with_unlabelled_numbers():
    valid = [0, 1, 2]
    text = 'I compared 3 options and prefer the first one because price is 1.0.'
    try:
        TextPersonaChoiceSimulator.parse_choice(text, valid)
    except ValueError:
        pass
    else:
        raise AssertionError('free-form numeric text should not be parsed as a choice')
