"""Simple baseline ladder for the controlled synthetic benchmark.

This module provides a small, auditable baseline ladder that uses the same
evaluation path as the outer optimizer:

1. ``static``: the initial generator from the config;
2. ``diversity_only``: a heuristic anchor-free generator with uniform weights and
   spread-out component means, included to inspect what diversity without
   behavioral grounding looks like;
3. ``oracle_truth``: the ground-truth synthetic population, available only in the
   controlled benchmark.

Each candidate is evaluated by the same pass:

    sample z ~ p_phi(z)
    simulate D_phi with pi_theta
    fit MNL m_beta to D_phi
    compare model-implied moments to D_H and calibration interventions
    evaluate the fitted model on held-out counterfactual contexts X_cf
    report regularization diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from eipg.econ import MNLConfig, MNLFitResult, MultinomialLogitModel, evaluate_fitted_mnl, predict_probabilities_long
from eipg.objectives.calibration import CalibrationMomentConfig, CalibrationReport, build_calibration_report
from eipg.objectives.regularization import RegularizationConfig, RegularizationReport, regularization_report
from eipg.personas import MixtureGeneratorParams, MixturePersonaGenerator
from eipg.simulators import RandomUtilityChoiceSimulator, SyntheticSimulatorConfig


@dataclass(frozen=True)
class BaselineCandidate:
    """One generator candidate in the baseline ladder."""

    name: str
    params: MixtureGeneratorParams
    description: str


@dataclass(frozen=True)
class BaselineResult:
    """Evaluation result for one baseline candidate."""

    candidate: BaselineCandidate
    mnl_fit: MNLFitResult
    mnl_evaluation: dict[str, Any]
    calibration_report: CalibrationReport
    counterfactual_report: CalibrationReport
    regularization_report: RegularizationReport
    n_personas: int
    n_observations: int
    persona_seed: int
    simulator_seed: int

    def summary_row(self) -> dict[str, Any]:
        calib_summary = self.calibration_report.summary()
        cf_summary = self.counterfactual_report.summary()
        reg_terms = self.regularization_report.terms
        datasets = self.mnl_evaluation.get("datasets", {})
        return {
            "candidate": self.candidate.name,
            "description": self.candidate.description,
            "n_personas": int(self.n_personas),
            "n_observations": int(self.n_observations),
            "persona_seed": int(self.persona_seed),
            "simulator_seed": int(self.simulator_seed),
            "train_nll_per_observation": float(self.mnl_fit.train_nll_per_observation),
            "anchor_nll_per_observation": _nested_metric(datasets, "D_H", "nll_per_observation"),
            "calib_int_nll_per_observation": _nested_metric(
                datasets, "D_calib_int_truth", "nll_per_observation"
            ),
            "cf_nll_per_observation": _nested_metric(datasets, "D_cf_truth", "nll_per_observation"),
            "anchor_accuracy": _nested_metric(datasets, "D_H", "top_choice_accuracy"),
            "cf_accuracy": _nested_metric(datasets, "D_cf_truth", "top_choice_accuracy"),
            "calibration_l2_error": float(calib_summary["l2_error"]),
            "calibration_rmse": float(calib_summary["rmse"]),
            "cf_l2_error": float(cf_summary["l2_error"]),
            "cf_rmse": float(cf_summary["rmse"]),
            "mixture_entropy": float(reg_terms["mixture_entropy"]),
            "normalized_mixture_entropy": float(reg_terms["normalized_mixture_entropy"]),
            "avg_pairwise_mean_distance": float(reg_terms["avg_pairwise_mean_distance"]),
            "regularization_objective": float(reg_terms["regularization_objective"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": {
                "name": self.candidate.name,
                "description": self.candidate.description,
                "params": self.candidate.params.to_dict(),
            },
            "summary": self.summary_row(),
            "mnl_fit": self.mnl_fit.to_dict(),
            "mnl_evaluation": self.mnl_evaluation,
            "calibration": self.calibration_report.to_dict(),
            "counterfactual": self.counterfactual_report.to_dict(),
            "regularization": self.regularization_report.to_dict(),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def _nested_metric(datasets: dict[str, Any], dataset_name: str, metric: str) -> float:
    value = datasets.get(dataset_name, {}).get(metric, float("nan"))
    return float(value)


def make_diversity_only_params(
    base: MixtureGeneratorParams,
    *,
    radius: float = 2.5,
) -> MixtureGeneratorParams:
    """Construct a bounded anchor-free diversity heuristic.

    This is a deliberately simple diagnostic baseline.  It sets weights to
    uniform and pushes component means outward from their centroid while keeping
    their original directions.  It is not a full optimizer; the full outer loop
    is introduced later.
    """

    k = base.k_components
    weights = np.ones(k, dtype=float) / k
    means = base.means.copy()
    center = means.mean(axis=0)
    directions = means - center

    for idx in range(k):
        norm = float(np.linalg.norm(directions[idx]))
        if norm < 1.0e-8:
            direction = np.zeros(base.latent_dim, dtype=float)
            direction[idx % base.latent_dim] = 1.0
        else:
            direction = directions[idx] / norm
        means[idx] = center + float(radius) * direction

    return MixtureGeneratorParams(
        weights=weights,
        means=means,
        features=base.features,
        within_component_std=base.within_component_std,
        segment_labels=base.segment_labels,
    )


def default_baseline_candidates(
    *,
    initial_params: MixtureGeneratorParams,
    truth_params: MixtureGeneratorParams | None = None,
    diversity_radius: float = 2.5,
) -> list[BaselineCandidate]:
    """Return the controlled-synthetic baseline ladder."""

    candidates = [
        BaselineCandidate(
            name="static",
            params=initial_params,
            description="Initial uncalibrated persona generator from the config.",
        ),
        BaselineCandidate(
            name="diversity_only",
            params=make_diversity_only_params(initial_params, radius=diversity_radius),
            description="Anchor-free diversity heuristic with uniform weights and spread-out means.",
        ),
    ]
    if truth_params is not None:
        candidates.append(
            BaselineCandidate(
                name="oracle_truth",
                params=truth_params,
                description="Ground-truth synthetic population; controlled-benchmark oracle reference.",
            )
        )
    return candidates


def evaluate_baseline_candidate(
    candidate: BaselineCandidate,
    *,
    x_sim: pd.DataFrame,
    d_h: pd.DataFrame,
    d_calib_int: pd.DataFrame,
    d_cf: pd.DataFrame,
    simulator_config: SyntheticSimulatorConfig,
    mnl_config: MNLConfig,
    calibration_config: CalibrationMomentConfig,
    regularization_config: RegularizationConfig,
    seed: int,
    n_personas: int,
    n_observations: int,
    persona_seed: int | None = None,
    simulator_seed: int | None = None,
) -> BaselineResult:
    """Evaluate one generator candidate through the full controlled-synthetic pass."""

    # Common-random-number support: callers can hold these seeds fixed across
    # candidates so that differences in scores are driven by phi rather than by
    # fresh simulation draws.  The legacy ``seed`` remains the fallback.
    persona_seed = int(seed if persona_seed is None else persona_seed)
    simulator_seed = int(seed + 10_000 if simulator_seed is None else simulator_seed)

    generator = MixturePersonaGenerator(candidate.params, seed=persona_seed)
    personas = generator.sample(n_personas)
    simulator = RandomUtilityChoiceSimulator(simulator_config, seed=simulator_seed)
    d_phi = simulator.simulate_long_dataset(
        contexts=x_sim,
        personas=personas,
        n_observations=n_observations,
        dataset_label=f"D_phi_{candidate.name}",
    )

    mnl = MultinomialLogitModel(mnl_config)
    fit = mnl.fit(d_phi)
    pred_h = predict_probabilities_long(d_h, beta=fit.beta, features=fit.features)
    pred_calib = predict_probabilities_long(d_calib_int, beta=fit.beta, features=fit.features)
    pred_cf = predict_probabilities_long(d_cf, beta=fit.beta, features=fit.features)

    mnl_eval = evaluate_fitted_mnl(
        fit,
        {
            f"D_phi_{candidate.name}": d_phi,
            "D_H": d_h,
            "D_calib_int_truth": d_calib_int,
            "D_cf_truth": d_cf,
        },
    )

    calib_report = build_calibration_report(
        anchor_target=d_h,
        anchor_model=pred_h,
        intervention_target=d_calib_int,
        intervention_model=pred_calib,
        config=calibration_config,
    )
    cf_report = build_calibration_report(
        anchor_target=d_cf,
        anchor_model=pred_cf,
        config=calibration_config,
    )
    reg_report = regularization_report(candidate.params, regularization_config)

    return BaselineResult(
        candidate=candidate,
        mnl_fit=fit,
        mnl_evaluation=mnl_eval,
        calibration_report=calib_report,
        counterfactual_report=cf_report,
        regularization_report=reg_report,
        n_personas=n_personas,
        n_observations=n_observations,
        persona_seed=persona_seed,
        simulator_seed=simulator_seed,
    )


def baseline_results_table(results: list[BaselineResult]) -> pd.DataFrame:
    """Return one summary row per baseline candidate."""

    return pd.DataFrame([result.summary_row() for result in results])
