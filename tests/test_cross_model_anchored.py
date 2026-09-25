from __future__ import annotations

import json

from eipg.simulators.huggingface_local import HuggingFaceLocalConfig, _select_loader_kind
from scripts.run_cross_model_anchored_calibration import _budget_trace
from scripts.derive_cross_model_residual_linucb import _action_features, _desired_sign
from eipg.experiments.gepa_persona import AnchoredSuffixRenderer, EIPGGEPAAdapter
from scripts.run_llm_prompt_refinement_calibration_v2 import initial_personas


def test_loader_selection_multimodal_families() -> None:
    assert _select_loader_kind(
        requested="auto",
        model_type="gemma3",
        architectures=("Gemma3ForConditionalGeneration",),
    ) == "multimodal"
    assert _select_loader_kind(
        requested="auto",
        model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
    ) == "multimodal"


def test_loader_selection_causal_family() -> None:
    assert _select_loader_kind(
        requested="auto",
        model_type="phi3",
        architectures=("Phi3ForCausalLM",),
    ) == "causal"
    assert _select_loader_kind(
        requested="causal",
        model_type="gemma3",
        architectures=("Gemma3ForConditionalGeneration",),
    ) == "causal"


def test_budget_trace_uses_only_completed_prefix(tmp_path) -> None:
    history = [
        {
            "evaluation_index": 0,
            "objective": 0.10,
            "personas": [{"persona_id": "budget", "segment_label": "b", "prompt": "p0"}],
        },
        {
            "evaluation_index": 1,
            "objective": 0.09,
            "personas": [{"persona_id": "budget", "segment_label": "b", "prompt": "p1"}],
        },
        {
            "evaluation_index": 2,
            "objective": 0.07,
            "personas": [{"persona_id": "budget", "segment_label": "b", "prompt": "p2"}],
        },
        {
            "evaluation_index": 3,
            "objective": 0.08,
            "personas": [{"persona_id": "budget", "segment_label": "b", "prompt": "p3"}],
        },
    ]
    summary = {
        "seed": 1729,
        "logical_calls_per_evaluation": 15,
        "evaluation_calls": 4,
        "logical_simulator_calls": 60,
        "final_objective": 0.07,
        "selected_personas": history[2]["personas"],
    }
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    trace = _budget_trace(tmp_path, [1, 2, 3, 4, 5])
    by_label = {row["label"]: row for row in trace["checkpoints"]}

    assert by_label["eval_budget_1"]["selected_objective"] == 0.10
    assert by_label["eval_budget_2"]["selected_objective"] == 0.09
    assert by_label["eval_budget_3"]["selected_objective"] == 0.07
    assert by_label["eval_budget_4"]["selected_objective"] == 0.07
    assert "eval_budget_5" not in by_label
    assert by_label["eval_budget_3"]["logical_simulator_call_budget"] == 45
    assert by_label["full_search"]["selected_objective"] == 0.07


def test_chat_template_kwargs_are_configurable() -> None:
    cfg = HuggingFaceLocalConfig.from_config({
        "model": "Qwen/Qwen3.5-9B",
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert cfg.chat_template_kwargs == {"enable_thinking": False}


def test_adaptive_residual_direction_convention() -> None:
    # Positive chosen-price residual means chosen prices are too high, so the
    # anchored edit should increase price sensitivity (positive coordinate).
    assert _desired_sign("price", 0.2) == 1
    assert _desired_sign("price", -0.2) == -1

    # Positive non-price chosen-feature residual means the simulator chooses
    # too much of that feature, so its preference weight should decrease.
    assert _desired_sign("quality", 0.2) == -1
    assert _desired_sign("sustain", -0.2) == 1
    assert _desired_sign("brand", 0.0) == 0


def test_adaptive_action_embedding_is_fixed() -> None:
    x = _action_features({
        "persona_id": "quality",
        "feature": "brand",
        "signed_step": -0.5,
    })
    assert x.shape == (9,)
    assert x[1] == 1.0
    assert x[3 + 4] == 1.0
    assert x[-1] == -1.0
    assert float((x != 0).sum()) == 3.0


def test_gepa_empty_suffix_preserves_authored_prompts_exactly() -> None:
    renderer = AnchoredSuffixRenderer(
        max_suffix_chars=900,
        forbidden_terms=["calibration", "residual", "heldout"],
        no_numeric_target_copying=True,
        target_values=[0.123456],
    )
    candidate = {
        "budget_suffix": "",
        "quality_suffix": "",
        "sustain_suffix": "",
    }
    rendered = renderer.render(candidate)
    baseline = initial_personas()
    assert [p.prompt for p in rendered] == [p.prompt for p in baseline]


def test_gepa_suffix_rejects_calibration_leakage() -> None:
    renderer = AnchoredSuffixRenderer(
        max_suffix_chars=900,
        forbidden_terms=["calibration", "residual", "heldout"],
        no_numeric_target_copying=True,
        target_values=[0.123456],
    )
    ok, error = renderer.validate({
        "budget_suffix": "Use the calibration residual to prefer cheaper products.",
        "quality_suffix": "",
        "sustain_suffix": "",
    })
    assert not ok
    assert error is not None
    assert "forbidden" in error


def test_gepa_adapter_exposes_optional_proposer_attribute() -> None:
    # GEPA 0.1.4 accesses this protocol field directly.
    assert hasattr(EIPGGEPAAdapter, "propose_new_texts")
    assert EIPGGEPAAdapter.propose_new_texts is None


def test_gepa_adapter_state_roundtrip_preserves_budget_accounting() -> None:
    adapter = EIPGGEPAAdapter.__new__(EIPGGEPAAdapter)
    adapter.evaluation_counter = 7
    adapter.logical_calls = 888
    adapter.candidate_discovery_calls = {0: 120, 2: 744}

    state = adapter.get_adapter_state()

    restored = EIPGGEPAAdapter.__new__(EIPGGEPAAdapter)
    restored.evaluation_counter = 0
    restored.logical_calls = 0
    restored.candidate_discovery_calls = {}
    restored.set_adapter_state(state)

    assert restored.evaluation_counter == 7
    assert restored.logical_calls == 888
    assert restored.candidate_discovery_calls == {0: 120, 2: 744}
