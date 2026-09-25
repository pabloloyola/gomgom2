"""Simulator-side components."""

from eipg.simulators.contexts import ChoiceContextConfig, ContextSets, generate_context_sets
from eipg.simulators.synthetic import (
    RandomUtilityChoiceSimulator,
    SyntheticSimulatorConfig,
    summarize_choice_dataset,
)

__all__ = [
    "ChoiceContextConfig",
    "ContextSets",
    "RandomUtilityChoiceSimulator",
    "SyntheticSimulatorConfig",
    "generate_context_sets",
    "summarize_choice_dataset",
]
