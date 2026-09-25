"""Black-box outer optimization for EIPG.

Paper alignment
---------------
EIPG optimizes the persona generator parameters ``phi`` through a noisy,
non-differentiable objective:

    J(phi) = L_cal(beta*(phi); D_H, X_calib_int) + lambda R(phi),

where the inner model ``beta*(phi)`` is obtained by fitting an economic model to
synthetic choices generated from ``G_phi``.  This module implements the first
clean outer loop: a small evolutionary search over mixture weights and component
means.

The implementation is deliberately simple and auditable.  Each candidate
``phi`` is evaluated by the same full pass used for baselines:

    sample z ~ p_phi(z)
    simulate D_phi with pi_theta
    fit MNL m_beta to D_phi
    compute calibration moments against D_H and calibration interventions
    add generator regularization
    rank candidates by the resulting objective

This is sufficient for local smoke tests and for validating the full bilevel
logic before adding LLM simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from eipg.baselines import BaselineCandidate, BaselineResult, evaluate_baseline_candidate
from eipg.econ import MNLConfig
from eipg.objectives.calibration import CalibrationMomentConfig
from eipg.objectives.regularization import RegularizationConfig
from eipg.personas import MixtureGeneratorParams
from eipg.simulators import SyntheticSimulatorConfig


@dataclass(frozen=True)
class EvolutionSearchConfig:
    """Configuration for the v0.8 evolutionary outer loop."""

    budget: int = 4
    population_size: int = 2
    mean_mutation_scale: float = 0.30
    weight_logit_mutation_scale: float = 0.25
    mean_clip: float = 4.0
    sigma_decay: float = 0.85
    include_current_best: bool = True
    objective_metric: str = "calibration_l2_plus_regularization"
    regularization_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.mean_mutation_scale < 0:
            raise ValueError("mean_mutation_scale must be non-negative")
        if self.weight_logit_mutation_scale < 0:
            raise ValueError("weight_logit_mutation_scale must be non-negative")
        if self.mean_clip <= 0:
            raise ValueError("mean_clip must be positive")
        if not (0 < self.sigma_decay <= 1.0):
            raise ValueError("sigma_decay must be in (0, 1]")
        allowed = {
            "calibration_l2",
            "calibration_l2_plus_regularization",
            "calibration_rmse_plus_regularization",
        }
        if self.objective_metric not in allowed:
            raise ValueError(f"unknown objective_metric={self.objective_metric!r}; allowed={sorted(allowed)}")

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "EvolutionSearchConfig":
        """Construct from the ``outer_optimizer`` config section."""

        return cls(
            budget=int(section.get("budget", 4)),
            population_size=int(section.get("population", section.get("population_size", 2))),
            mean_mutation_scale=float(section.get("mean_mutation_scale", 0.30)),
            weight_logit_mutation_scale=float(section.get("weight_logit_mutation_scale", 0.25)),
            mean_clip=float(section.get("mean_clip", 4.0)),
            sigma_decay=float(section.get("sigma_decay", 0.85)),
            include_current_best=bool(section.get("include_current_best", True)),
            objective_metric=str(
                section.get("objective_metric", "calibration_l2_plus_regularization")
            ),
            regularization_multiplier=float(section.get("regularization_multiplier", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "evolutionary_search",
            "budget": int(self.budget),
            "population_size": int(self.population_size),
            "mean_mutation_scale": float(self.mean_mutation_scale),
            "weight_logit_mutation_scale": float(self.weight_logit_mutation_scale),
            "mean_clip": float(self.mean_clip),
            "sigma_decay": float(self.sigma_decay),
            "include_current_best": bool(self.include_current_best),
            "objective_metric": self.objective_metric,
            "regularization_multiplier": float(self.regularization_multiplier),
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """One evaluated outer-loop candidate."""

    generation: int
    candidate_index: int
    candidate_name: str
    params: MixtureGeneratorParams
    result: BaselineResult
    objective_value: float
    objective_components: dict[str, float]

    def summary_row(self) -> dict[str, Any]:
        row = self.result.summary_row()
        row.update(
            {
                "generation": int(self.generation),
                "candidate_index": int(self.candidate_index),
                "candidate": self.candidate_name,
                "objective_value": float(self.objective_value),
            }
        )
        for key, value in self.objective_components.items():
            row[f"objective_{key}"] = float(value)
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": int(self.generation),
            "candidate_index": int(self.candidate_index),
            "candidate_name": self.candidate_name,
            "objective_value": float(self.objective_value),
            "objective_components": {k: float(v) for k, v in self.objective_components.items()},
            "params": self.params.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class EvolutionSearchResult:
    """Complete result of the evolutionary outer loop."""

    config: EvolutionSearchConfig
    evaluations: tuple[CandidateEvaluation, ...]
    best: CandidateEvaluation

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame([ev.summary_row() for ev in self.evaluations])

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "best": self.best.to_dict(),
            "history_rows": self.history.to_dict(orient="records"),
            "n_evaluations": int(len(self.evaluations)),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def compute_outer_objective(
    result: BaselineResult,
    *,
    config: EvolutionSearchConfig,
) -> tuple[float, dict[str, float]]:
    """Return the scalar outer objective and named components."""

    calib_summary = result.calibration_report.summary()
    calibration_l2 = float(calib_summary["l2_error"])
    calibration_rmse = float(calib_summary["rmse"])
    regularization = float(result.regularization_report.objective())

    if config.objective_metric == "calibration_l2":
        objective = calibration_l2
    elif config.objective_metric == "calibration_rmse_plus_regularization":
        objective = calibration_rmse + float(config.regularization_multiplier) * regularization
    else:
        objective = calibration_l2 + float(config.regularization_multiplier) * regularization

    return float(objective), {
        "calibration_l2": calibration_l2,
        "calibration_rmse": calibration_rmse,
        "regularization": regularization,
        "regularization_multiplier": float(config.regularization_multiplier),
    }


def mutate_generator_params(
    params: MixtureGeneratorParams,
    *,
    rng: np.random.Generator,
    mean_scale: float,
    weight_logit_scale: float,
    mean_clip: float,
) -> MixtureGeneratorParams:
    """Create a nearby generator candidate by perturbing logits and means."""

    logits = params.log_weights.copy()
    means = params.means.copy()

    if weight_logit_scale > 0:
        logits = logits + rng.normal(0.0, float(weight_logit_scale), size=logits.shape)
    if mean_scale > 0:
        means = means + rng.normal(0.0, float(mean_scale), size=means.shape)
    means = np.clip(means, -float(mean_clip), float(mean_clip))

    phi = np.concatenate([logits, means.ravel()])
    return MixtureGeneratorParams.from_flat_phi(
        phi,
        k_components=params.k_components,
        features=params.features,
        within_component_std=params.within_component_std,
        segment_labels=params.segment_labels,
    )


def run_evolutionary_search(
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
    search_config: EvolutionSearchConfig,
    seed: int,
    n_personas: int,
    n_observations: int,
    evaluation_persona_seed: int | None = None,
    evaluation_simulator_seed: int | None = None,
) -> EvolutionSearchResult:
    """Run a small evolutionary search over persona-generator parameters."""

    rng = np.random.default_rng(seed)
    # Candidate evaluations use common random numbers by default.  The search
    # mutation RNG remains independent from the simulation RNGs.
    if evaluation_persona_seed is None:
        evaluation_persona_seed = seed + 100_000
    if evaluation_simulator_seed is None:
        evaluation_simulator_seed = seed + 110_000
    current_center = initial_params
    best_eval: CandidateEvaluation | None = None
    evaluations: list[CandidateEvaluation] = []

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
            name = f"eipg_g{generation:03d}_c{candidate_index:03d}"
            candidate = BaselineCandidate(
                name=name,
                params=params,
                description="Outer-loop EIPG candidate generated by evolutionary search.",
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
                persona_seed=evaluation_persona_seed,
                simulator_seed=evaluation_simulator_seed,
            )
            objective, components = compute_outer_objective(result, config=search_config)
            evaluation = CandidateEvaluation(
                generation=generation,
                candidate_index=candidate_index,
                candidate_name=name,
                params=params,
                result=result,
                objective_value=objective,
                objective_components=components,
            )
            evaluations.append(evaluation)
            generation_evals.append(evaluation)
            if best_eval is None or evaluation.objective_value < best_eval.objective_value:
                best_eval = evaluation

        # Move the search center to the best candidate seen in this generation.
        generation_best = min(generation_evals, key=lambda ev: ev.objective_value)
        current_center = generation_best.params

    assert best_eval is not None
    return EvolutionSearchResult(
        config=search_config,
        evaluations=tuple(evaluations),
        best=best_eval,
    )
