"""Block-normalized outer optimization for symbolic EIPG search.

This module leaves the existing paper-facing evolutionary search untouched and
adds an experimental objective for configuration search. Calibration moments
are grouped into economically interpretable blocks and normalized within block
before weighting. This prevents a block from receiving more influence merely
because it contains more scalar moments.

The held-out counterfactual report is never used in candidate selection. It is
carried through ``BaselineResult`` only for diagnostics after the candidate has
been scored on calibration data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from eipg.baselines import BaselineCandidate, BaselineResult, evaluate_baseline_candidate
from eipg.econ import MNLConfig
from eipg.objectives.calibration import CalibrationMomentConfig, CalibrationReport
from eipg.objectives.regularization import RegularizationConfig
from eipg.outeropt.evolution import (
    CandidateEvaluation,
    EvolutionSearchResult,
    mutate_generator_params,
)
from eipg.personas import MixtureGeneratorParams
from eipg.simulators import SyntheticSimulatorConfig


DEFAULT_BLOCK_WEIGHTS: dict[str, float] = {
    "level_shares": 1.0,
    "level_attributes": 1.0,
    "substitution": 1.0,
    "intervention_attributes": 1.0,
}

ProgressCallback = Callable[[dict[str, Any]], None]


def calibration_block(moment_type: str, short_name: str) -> str:
    """Map a scalar calibration moment to an interpretable objective block."""

    moment_type = str(moment_type)
    short_name = str(short_name)
    if moment_type in {"item_shares", "group_shares"}:
        return "level_shares"
    if moment_type == "chosen_attributes":
        return "level_attributes"
    if moment_type == "intervention_response":
        if (
            short_name.startswith("delta_alt_")
            or short_name.startswith("delta_intervened_alt_")
            or short_name.startswith("intervened_alt_")
        ):
            return "substitution"
        return "intervention_attributes"
    return "other"


def block_weighted_calibration_score(
    report: CalibrationReport,
    *,
    block_weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return sqrt(weighted mean block MSE) and block-level diagnostics.

    Each block contributes its *mean* squared error, so moment count does not
    implicitly determine block weight. Blocks with zero weight are excluded.
    """

    weights = dict(DEFAULT_BLOCK_WEIGHTS)
    if block_weights:
        weights.update({str(k): float(v) for k, v in block_weights.items()})
    if any(v < 0 for v in weights.values()):
        raise ValueError("calibration block weights must be non-negative")

    table = report.table.copy()
    if table.empty:
        return 0.0, {"weighted_calibration": 0.0}
    required = {"moment_type", "short_name", "diff"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"calibration table missing columns: {sorted(missing)}")

    table["objective_block"] = [
        calibration_block(mt, sn)
        for mt, sn in zip(table["moment_type"], table["short_name"])
    ]

    diagnostics: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for block, group in table.groupby("objective_block", sort=True):
        diff = group["diff"].to_numpy(dtype=float)
        mse = float(np.mean(diff * diff))
        rmse = float(np.sqrt(mse))
        diagnostics[f"block_{block}_rmse"] = rmse
        diagnostics[f"block_{block}_n"] = float(len(group))
        weight = float(weights.get(block, 0.0))
        diagnostics[f"block_{block}_weight"] = weight
        if weight > 0:
            weighted_sum += weight * mse
            total_weight += weight

    if total_weight <= 0:
        raise ValueError("at least one calibration block must have positive weight")
    score = float(np.sqrt(weighted_sum / total_weight))
    diagnostics["weighted_calibration"] = score
    diagnostics["active_block_weight_sum"] = float(total_weight)
    return score, diagnostics


@dataclass(frozen=True)
class BlockWeightedSearchConfig:
    """Evolutionary-search settings plus block-normalized calibration weights."""

    budget: int = 6
    population_size: int = 4
    mean_mutation_scale: float = 0.25
    weight_logit_mutation_scale: float = 0.20
    mean_clip: float = 4.0
    sigma_decay: float = 0.90
    include_current_best: bool = True
    regularization_multiplier: float = 0.05
    block_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BLOCK_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.budget <= 0 or self.population_size <= 0:
            raise ValueError("budget and population_size must be positive")
        if self.mean_mutation_scale < 0 or self.weight_logit_mutation_scale < 0:
            raise ValueError("mutation scales must be non-negative")
        if self.mean_clip <= 0:
            raise ValueError("mean_clip must be positive")
        if not (0 < self.sigma_decay <= 1.0):
            raise ValueError("sigma_decay must be in (0, 1]")
        if self.regularization_multiplier < 0:
            raise ValueError("regularization_multiplier must be non-negative")
        for key, value in self.block_weights.items():
            if float(value) < 0:
                raise ValueError(f"negative block weight for {key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "block_weighted_evolutionary_search",
            "budget": int(self.budget),
            "population_size": int(self.population_size),
            "mean_mutation_scale": float(self.mean_mutation_scale),
            "weight_logit_mutation_scale": float(self.weight_logit_mutation_scale),
            "mean_clip": float(self.mean_clip),
            "sigma_decay": float(self.sigma_decay),
            "include_current_best": bool(self.include_current_best),
            "regularization_multiplier": float(self.regularization_multiplier),
            "block_weights": {k: float(v) for k, v in self.block_weights.items()},
        }


def compute_block_weighted_objective(
    result: BaselineResult,
    *,
    config: BlockWeightedSearchConfig,
) -> tuple[float, dict[str, float]]:
    """Compute the search objective without using held-out counterfactual data."""

    calibration, block_diag = block_weighted_calibration_score(
        result.calibration_report,
        block_weights=config.block_weights,
    )
    regularization = float(result.regularization_report.objective())
    objective = calibration + float(config.regularization_multiplier) * regularization
    components = {
        "weighted_calibration": float(calibration),
        "regularization": regularization,
        "regularization_multiplier": float(config.regularization_multiplier),
        **block_diag,
    }
    return float(objective), components


def run_block_weighted_search(
    *,
    initial_params: MixtureGeneratorParams,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    d_calib_int: pd.DataFrame,
    d_cf: pd.DataFrame,
    simulator_config: SyntheticSimulatorConfig,
    mnl_config: MNLConfig,
    calibration_config: CalibrationMomentConfig,
    regularization_config: RegularizationConfig,
    search_config: BlockWeightedSearchConfig,
    seed: int,
    n_personas: int,
    n_observations: int,
    evaluation_persona_seed: int | None = None,
    evaluation_simulator_seed: int | None = None,
    candidate_prefix: str = "weighted",
    progress_callback: ProgressCallback | None = None,
) -> EvolutionSearchResult:
    """Run evolutionary search using block-normalized calibration loss.

    ``progress_callback`` is called after every evaluated candidate. It receives
    generation/candidate indices, local completed/total counts, objective values,
    and diagnostic calibration/CF scores. The callback is reporting-only and
    never affects candidate selection.
    """

    rng = np.random.default_rng(seed)
    persona_seed = int(seed + 100_000 if evaluation_persona_seed is None else evaluation_persona_seed)
    simulator_seed = int(seed + 110_000 if evaluation_simulator_seed is None else evaluation_simulator_seed)
    current_center = initial_params
    best_eval: CandidateEvaluation | None = None
    evaluations: list[CandidateEvaluation] = []
    total_evaluations = int(search_config.budget * search_config.population_size)

    for generation in range(search_config.budget):
        scale_factor = float(search_config.sigma_decay) ** generation
        mean_scale = float(search_config.mean_mutation_scale) * scale_factor
        weight_scale = float(search_config.weight_logit_mutation_scale) * scale_factor

        candidate_params: list[MixtureGeneratorParams] = []
        if search_config.include_current_best:
            candidate_params.append(current_center)
        while len(candidate_params) < search_config.population_size:
            candidate_params.append(
                mutate_generator_params(
                    current_center,
                    rng=rng,
                    mean_scale=mean_scale,
                    weight_logit_scale=weight_scale,
                    mean_clip=search_config.mean_clip,
                )
            )

        generation_evals: list[CandidateEvaluation] = []
        for candidate_index, params in enumerate(candidate_params[: search_config.population_size]):
            name = f"{candidate_prefix}_g{generation:03d}_c{candidate_index:03d}"
            candidate = BaselineCandidate(
                name=name,
                params=params,
                description="Block-weighted EIPG candidate.",
            )
            result = evaluate_baseline_candidate(
                candidate,
                x_sim=x_sim,
                d_h=d_h,
                d_calib_int=d_calib_int,
                d_cf=d_cf,
                simulator_config=simulator_config,
                mnl_config=mnl_config,
                calibration_config=calibration_config,
                regularization_config=regularization_config,
                seed=seed,
                n_personas=n_personas,
                n_observations=n_observations,
                persona_seed=persona_seed,
                simulator_seed=simulator_seed,
            )
            objective, components = compute_block_weighted_objective(
                result,
                config=search_config,
            )
            ev = CandidateEvaluation(
                generation=generation,
                candidate_index=candidate_index,
                candidate_name=name,
                params=params,
                result=result,
                objective_value=objective,
                objective_components=components,
            )
            evaluations.append(ev)
            generation_evals.append(ev)
            if best_eval is None or ev.objective_value < best_eval.objective_value:
                best_eval = ev

            if progress_callback is not None:
                summary = result.summary_row()
                progress_callback(
                    {
                        "candidate": name,
                        "generation": int(generation),
                        "candidate_index": int(candidate_index),
                        "completed": int(len(evaluations)),
                        "total": total_evaluations,
                        "objective": float(objective),
                        "weighted_calibration": float(components["weighted_calibration"]),
                        "calibration_l2": float(summary["calibration_l2_error"]),
                        "cf_l2": float(summary["cf_l2_error"]),
                    }
                )

        generation_best = min(generation_evals, key=lambda ev: ev.objective_value)
        current_center = generation_best.params

    assert best_eval is not None
    return EvolutionSearchResult(
        config=search_config,  # type: ignore[arg-type]
        evaluations=tuple(evaluations),
        best=best_eval,
    )
