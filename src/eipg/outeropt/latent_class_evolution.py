"""Outer-loop EIPG search with a panel latent-class MNL inner model.

This module is intentionally parallel to ``eipg.outeropt.evolution``.  It is a
controlled-benchmark experiment rather than a replacement for the default
homogeneous-MNL path.

For each candidate generator ``phi`` we:

1. sample a synthetic persona population with common random numbers;
2. simulate repeated choices on ``X_sim``;
3. fit a panel latent-class MNL to those choices;
4. predict the human anchor and calibration-intervention sets using the
   *population mixture* (no target persona labels/posteriors);
5. compute the same calibration moments and generator regularizer used by the
   standard EIPG objective;
6. record held-out counterfactual error for diagnostics only.

The held-out counterfactual set is never used for candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from eipg.econ import (
    MNLConfig,
    MultinomialLogitModel,
    fit_latent_class_mnl,
    predict_latent_class_mnl_long,
)
from eipg.objectives.calibration import (
    CalibrationMomentConfig,
    CalibrationReport,
    build_calibration_report,
)
from eipg.objectives.regularization import (
    RegularizationConfig,
    RegularizationReport,
    regularization_report,
)
from eipg.outeropt.evolution import EvolutionSearchConfig, mutate_generator_params
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig


@dataclass(frozen=True)
class LatentClassInnerConfig:
    """Configuration for the panel latent-class inner model."""

    n_classes: int = 3
    panel_id_column: str = "persona_id"
    n_restarts: int = 2
    em_max_iter: int = 25
    em_tol: float = 1e-5
    mstep_max_iter: int = 80
    init_scale: float = 0.8
    min_class_weight: float = 1e-3
    seed: int = 0

    @classmethod
    def from_config(
        cls,
        section: dict[str, Any] | None,
        *,
        default_n_classes: int,
        seed: int,
    ) -> "LatentClassInnerConfig":
        section = section or {}
        return cls(
            n_classes=int(section.get("n_classes", default_n_classes)),
            panel_id_column=str(section.get("panel_id_column", "persona_id")),
            n_restarts=int(section.get("n_restarts", 2)),
            em_max_iter=int(section.get("em_max_iter", 25)),
            em_tol=float(section.get("em_tol", 1e-5)),
            mstep_max_iter=int(section.get("mstep_max_iter", 80)),
            init_scale=float(section.get("init_scale", 0.8)),
            min_class_weight=float(section.get("min_class_weight", 1e-3)),
            seed=int(section.get("seed", seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_classes": int(self.n_classes),
            "panel_id_column": self.panel_id_column,
            "n_restarts": int(self.n_restarts),
            "em_max_iter": int(self.em_max_iter),
            "em_tol": float(self.em_tol),
            "mstep_max_iter": int(self.mstep_max_iter),
            "init_scale": float(self.init_scale),
            "min_class_weight": float(self.min_class_weight),
            "seed": int(self.seed),
        }


def _probability_metrics(
    target: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    probability_column: str,
) -> dict[str, float]:
    """Evaluate long-format choice probabilities against realized choices."""

    required = {"observation_id", "chosen", probability_column}
    missing = sorted(required - set(predicted.columns))
    if missing:
        raise ValueError(f"prediction table missing required columns: {missing}")
    if len(target) != len(predicted):
        raise ValueError("target and predicted tables must have the same number of rows")

    work = predicted.reset_index(drop=True)
    chosen_mask = work["chosen"].to_numpy(float) > 0.5
    chosen_prob = work.loc[chosen_mask, probability_column].to_numpy(float)
    chosen_prob = np.clip(chosen_prob, 1e-15, 1.0)
    nll = -float(np.mean(np.log(chosen_prob)))

    # Top-choice accuracy is computed at the choice-occasion level.
    idx = work.groupby("observation_id", sort=False)[probability_column].idxmax()
    top_accuracy = float(work.loc[idx, "chosen"].astype(float).mean())

    return {
        "nll_per_observation": nll,
        "top_choice_accuracy": top_accuracy,
    }


@dataclass(frozen=True)
class LatentClassCandidateResult:
    """Evaluation of one generator under the panel latent-class inner model."""

    candidate_name: str
    params: MixtureGeneratorParams
    fit: Any
    calibration_report: CalibrationReport
    counterfactual_report: CalibrationReport
    regularization_report: RegularizationReport
    evaluation: dict[str, Any]
    n_personas: int
    n_observations: int
    persona_seed: int
    simulator_seed: int
    inner_model_seed: int

    @property
    def train_nll_per_observation(self) -> float:
        return -float(self.fit.log_likelihood) / float(self.fit.n_observations)

    def summary_row(self) -> dict[str, Any]:
        calib = self.calibration_report.summary()
        cf = self.counterfactual_report.summary()
        reg = self.regularization_report.terms
        datasets = self.evaluation.get("datasets", {})
        class_weights = np.asarray(self.fit.class_weights, dtype=float)
        class_entropy = -float(np.sum(class_weights * np.log(np.clip(class_weights, 1e-15, 1.0))))
        return {
            "candidate": self.candidate_name,
            "inner_model": "panel_latent_class_mnl",
            "n_personas": int(self.n_personas),
            "n_observations": int(self.n_observations),
            "persona_seed": int(self.persona_seed),
            "simulator_seed": int(self.simulator_seed),
            "inner_model_seed": int(self.inner_model_seed),
            "train_nll_per_observation": self.train_nll_per_observation,
            "anchor_nll_per_observation": float(datasets["D_H"]["nll_per_observation"]),
            "calib_int_nll_per_observation": float(
                datasets["D_calib_int_truth"]["nll_per_observation"]
            ),
            "cf_nll_per_observation": float(datasets["D_cf_truth"]["nll_per_observation"]),
            "anchor_accuracy": float(datasets["D_H"]["top_choice_accuracy"]),
            "cf_accuracy": float(datasets["D_cf_truth"]["top_choice_accuracy"]),
            "calibration_l2_error": float(calib["l2_error"]),
            "calibration_rmse": float(calib["rmse"]),
            "cf_l2_error": float(cf["l2_error"]),
            "cf_rmse": float(cf["rmse"]),
            "mixture_entropy": float(reg["mixture_entropy"]),
            "normalized_mixture_entropy": float(reg["normalized_mixture_entropy"]),
            "avg_pairwise_mean_distance": float(reg["avg_pairwise_mean_distance"]),
            "regularization_objective": float(reg["regularization_objective"]),
            "lc_n_classes": int(self.fit.n_classes),
            "lc_class_entropy": class_entropy,
            "lc_converged": bool(self.fit.converged),
            "lc_em_iterations": int(self.fit.em_iterations),
            "lc_selected_restart": int(self.fit.selected_restart),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_name,
            "inner_model": "panel_latent_class_mnl",
            "params": self.params.to_dict(),
            "summary": self.summary_row(),
            "fit": self.fit.to_dict(),
            "evaluation": self.evaluation,
            "calibration": self.calibration_report.to_dict(),
            "counterfactual": self.counterfactual_report.to_dict(),
            "regularization": self.regularization_report.to_dict(),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def evaluate_latent_class_candidate(
    *,
    candidate_name: str,
    params: MixtureGeneratorParams,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    d_calib_int: pd.DataFrame,
    d_cf: pd.DataFrame,
    simulator_config: SyntheticSimulatorConfig,
    mnl_config: MNLConfig,
    calibration_config: CalibrationMomentConfig,
    regularization_config: RegularizationConfig,
    lc_config: LatentClassInnerConfig,
    n_personas: int,
    n_observations: int,
    persona_seed: int,
    simulator_seed: int,
) -> LatentClassCandidateResult:
    """Evaluate one EIPG generator using panel latent-class feedback."""

    generator = MixturePersonaGenerator(params, seed=int(persona_seed))
    personas = generator.sample(int(n_personas))
    simulator = RandomUtilityChoiceSimulator(simulator_config, seed=int(simulator_seed))
    d_phi = simulator.simulate_long_dataset(
        contexts=x_sim,
        personas=personas,
        n_observations=int(n_observations),
        dataset_label=f"D_phi_{candidate_name}",
    )

    # Candidate-specific homogeneous MNL provides a stable center for the EM
    # initializations.  The actual feedback model remains the latent-class MNL.
    base_fit = MultinomialLogitModel(mnl_config).fit(d_phi)
    fit = fit_latent_class_mnl(
        d_phi,
        mnl_config,
        n_classes=lc_config.n_classes,
        seed=lc_config.seed,
        n_restarts=lc_config.n_restarts,
        em_max_iter=lc_config.em_max_iter,
        em_tol=lc_config.em_tol,
        mstep_max_iter=lc_config.mstep_max_iter,
        init_scale=lc_config.init_scale,
        min_class_weight=lc_config.min_class_weight,
        initial_beta=base_fit.beta,
        panel_id_column=lc_config.panel_id_column,
    )

    probability_column = calibration_config.probability_column
    pred_h = predict_latent_class_mnl_long(
        d_h, fit, probability_column=probability_column
    )
    pred_calib = predict_latent_class_mnl_long(
        d_calib_int, fit, probability_column=probability_column
    )
    pred_cf = predict_latent_class_mnl_long(
        d_cf, fit, probability_column=probability_column
    )

    calibration_report = build_calibration_report(
        anchor_target=d_h,
        anchor_model=pred_h,
        intervention_target=d_calib_int,
        intervention_model=pred_calib,
        config=calibration_config,
    )
    counterfactual_report = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=pred_cf,
        config=calibration_config,
    )
    reg_report = regularization_report(params, regularization_config)

    evaluation = {
        "inner_model": "panel_latent_class_mnl",
        "datasets": {
            "D_H": _probability_metrics(
                d_h, pred_h, probability_column=probability_column
            ),
            "D_calib_int_truth": _probability_metrics(
                d_calib_int, pred_calib, probability_column=probability_column
            ),
            "D_cf_truth": _probability_metrics(
                d_cf, pred_cf, probability_column=probability_column
            ),
        },
    }

    return LatentClassCandidateResult(
        candidate_name=candidate_name,
        params=params,
        fit=fit,
        calibration_report=calibration_report,
        counterfactual_report=counterfactual_report,
        regularization_report=reg_report,
        evaluation=evaluation,
        n_personas=int(n_personas),
        n_observations=int(n_observations),
        persona_seed=int(persona_seed),
        simulator_seed=int(simulator_seed),
        inner_model_seed=int(lc_config.seed),
    )


def _outer_objective(
    result: LatentClassCandidateResult,
    search_config: EvolutionSearchConfig,
) -> tuple[float, dict[str, float]]:
    calib = result.calibration_report.summary()
    calibration_l2 = float(calib["l2_error"])
    calibration_rmse = float(calib["rmse"])
    regularization = float(result.regularization_report.objective())

    if search_config.objective_metric == "calibration_l2":
        objective = calibration_l2
    elif search_config.objective_metric == "calibration_rmse_plus_regularization":
        objective = calibration_rmse + search_config.regularization_multiplier * regularization
    else:
        objective = calibration_l2 + search_config.regularization_multiplier * regularization

    return float(objective), {
        "calibration_l2": calibration_l2,
        "calibration_rmse": calibration_rmse,
        "regularization": regularization,
        "regularization_multiplier": float(search_config.regularization_multiplier),
    }


@dataclass(frozen=True)
class LatentClassCandidateEvaluation:
    generation: int
    candidate_index: int
    candidate_name: str
    params: MixtureGeneratorParams
    result: LatentClassCandidateResult
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
            "objective_components": self.objective_components,
            "params": self.params.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class LatentClassEvolutionSearchResult:
    search_config: EvolutionSearchConfig
    inner_config: LatentClassInnerConfig
    evaluations: tuple[LatentClassCandidateEvaluation, ...]
    best: LatentClassCandidateEvaluation

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame([ev.summary_row() for ev in self.evaluations])

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_config": self.search_config.to_dict(),
            "inner_config": self.inner_config.to_dict(),
            "n_evaluations": int(len(self.evaluations)),
            "best": self.best.to_dict(),
            "history_rows": self.history.to_dict(orient="records"),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def run_latent_class_evolutionary_search(
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
    lc_config: LatentClassInnerConfig,
    seed: int,
    n_personas: int,
    n_observations: int,
    evaluation_persona_seed: int,
    evaluation_simulator_seed: int,
) -> LatentClassEvolutionSearchResult:
    """Run EIPG with panel latent-class MNL feedback."""

    rng = np.random.default_rng(seed)
    current_center = initial_params
    best_eval: LatentClassCandidateEvaluation | None = None
    evaluations: list[LatentClassCandidateEvaluation] = []

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

        generation_evals: list[LatentClassCandidateEvaluation] = []
        for candidate_index, params in enumerate(candidate_params[: search_config.population_size]):
            name = f"eipg_lc_g{generation:03d}_c{candidate_index:03d}"
            result = evaluate_latent_class_candidate(
                candidate_name=name,
                params=params,
                x_sim=x_sim,
                d_h=d_h,
                d_calib_int=d_calib_int,
                d_cf=d_cf,
                simulator_config=simulator_config,
                mnl_config=mnl_config,
                calibration_config=calibration_config,
                regularization_config=regularization_config,
                lc_config=lc_config,
                n_personas=n_personas,
                n_observations=n_observations,
                persona_seed=evaluation_persona_seed,
                simulator_seed=evaluation_simulator_seed,
            )
            objective, components = _outer_objective(result, search_config)
            ev = LatentClassCandidateEvaluation(
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

        current_center = min(generation_evals, key=lambda x: x.objective_value).params

    assert best_eval is not None
    return LatentClassEvolutionSearchResult(
        search_config=search_config,
        inner_config=lc_config,
        evaluations=tuple(evaluations),
        best=best_eval,
    )
