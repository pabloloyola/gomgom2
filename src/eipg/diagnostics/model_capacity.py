"""Economic-model capacity diagnostics for the controlled benchmark.

The controlled benchmark exposes the ground-truth random-utility simulator and
persona latents. This module separates sources of discrepancy that are otherwise
confounded in ordinary EIPG evaluation.

The v1.9.2 diagnostic distinguishes two oracle segmented paths:

* ``oracle_segmented_mnl``: train with true segment labels *and* route each
  target observation through its true target segment. This is a privileged
  routed upper bound.
* ``oracle_mixture_mnl``: train the same segment-specific MNLs with true labels,
  but at prediction time withhold target labels and average segment predictions
  using training-population weights. This is the fair oracle comparator for an
  estimated latent-class MNL.

The direct oracle reuses the simulator probabilities stored for the exact
sampled persona-context tasks. The ordinary ``oracle_truth`` path fits one
homogeneous MNL to the same ground-truth generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eipg.objectives.calibration import (
    CalibrationMomentConfig,
    CalibrationReport,
    build_calibration_report,
    compare_calibration_reports,
)


@dataclass(frozen=True)
class EconomicModelCapacityDiagnostic:
    """Direct, oracle-routed, oracle-mixture, latent-class, and homogeneous outputs."""

    direct_oracle_calibration: CalibrationReport
    direct_oracle_counterfactual: CalibrationReport
    oracle_through_mnl_calibration: CalibrationReport
    oracle_through_mnl_counterfactual: CalibrationReport
    moment_comparison: pd.DataFrame
    oracle_segmented_mnl_calibration: CalibrationReport | None = None
    oracle_segmented_mnl_counterfactual: CalibrationReport | None = None
    oracle_mixture_mnl_calibration: CalibrationReport | None = None
    oracle_mixture_mnl_counterfactual: CalibrationReport | None = None
    estimated_latent_class_mnl_calibration: CalibrationReport | None = None
    estimated_latent_class_mnl_counterfactual: CalibrationReport | None = None

    def summary(self) -> dict[str, Any]:
        direct_cal = self.direct_oracle_calibration.summary()
        direct_cf = self.direct_oracle_counterfactual.summary()
        mnl_cal = self.oracle_through_mnl_calibration.summary()
        mnl_cf = self.oracle_through_mnl_counterfactual.summary()
        out: dict[str, Any] = {
            "direct_oracle": {
                "calibration_l2_error": float(direct_cal["l2_error"]),
                "calibration_rmse": float(direct_cal["rmse"]),
                "cf_l2_error": float(direct_cf["l2_error"]),
                "cf_rmse": float(direct_cf["rmse"]),
            },
            "oracle_through_mnl": {
                "calibration_l2_error": float(mnl_cal["l2_error"]),
                "calibration_rmse": float(mnl_cal["rmse"]),
                "cf_l2_error": float(mnl_cf["l2_error"]),
                "cf_rmse": float(mnl_cf["rmse"]),
            },
            "excess_l2_after_mnl": {
                "calibration": float(mnl_cal["l2_error"] - direct_cal["l2_error"]),
                "counterfactual": float(mnl_cf["l2_error"] - direct_cf["l2_error"]),
            },
        }

        seg_cal = seg_cf = None
        if (
            self.oracle_segmented_mnl_calibration is not None
            and self.oracle_segmented_mnl_counterfactual is not None
        ):
            seg_cal = self.oracle_segmented_mnl_calibration.summary()
            seg_cf = self.oracle_segmented_mnl_counterfactual.summary()
            out["oracle_segmented_mnl"] = {
                "calibration_l2_error": float(seg_cal["l2_error"]),
                "calibration_rmse": float(seg_cal["rmse"]),
                "cf_l2_error": float(seg_cf["l2_error"]),
                "cf_rmse": float(seg_cf["rmse"]),
            }
            out["excess_l2_after_segmented_mnl"] = {
                "calibration": float(seg_cal["l2_error"] - direct_cal["l2_error"]),
                "counterfactual": float(seg_cf["l2_error"] - direct_cf["l2_error"]),
            }
            out["homogeneous_minus_segmented_l2"] = {
                "calibration": float(mnl_cal["l2_error"] - seg_cal["l2_error"]),
                "counterfactual": float(mnl_cf["l2_error"] - seg_cf["l2_error"]),
            }

        mix_cal = mix_cf = None
        if (
            self.oracle_mixture_mnl_calibration is not None
            and self.oracle_mixture_mnl_counterfactual is not None
        ):
            mix_cal = self.oracle_mixture_mnl_calibration.summary()
            mix_cf = self.oracle_mixture_mnl_counterfactual.summary()
            out["oracle_mixture_mnl"] = {
                "calibration_l2_error": float(mix_cal["l2_error"]),
                "calibration_rmse": float(mix_cal["rmse"]),
                "cf_l2_error": float(mix_cf["l2_error"]),
                "cf_rmse": float(mix_cf["rmse"]),
            }
            out["excess_l2_after_oracle_mixture_mnl"] = {
                "calibration": float(mix_cal["l2_error"] - direct_cal["l2_error"]),
                "counterfactual": float(mix_cf["l2_error"] - direct_cf["l2_error"]),
            }
            out["homogeneous_minus_oracle_mixture_l2"] = {
                "calibration": float(mnl_cal["l2_error"] - mix_cal["l2_error"]),
                "counterfactual": float(mnl_cf["l2_error"] - mix_cf["l2_error"]),
            }
            if seg_cal is not None and seg_cf is not None:
                out["oracle_mixture_minus_routed_segmented_l2"] = {
                    "calibration": float(mix_cal["l2_error"] - seg_cal["l2_error"]),
                    "counterfactual": float(mix_cf["l2_error"] - seg_cf["l2_error"]),
                }

        if (
            self.estimated_latent_class_mnl_calibration is not None
            and self.estimated_latent_class_mnl_counterfactual is not None
        ):
            lc_cal = self.estimated_latent_class_mnl_calibration.summary()
            lc_cf = self.estimated_latent_class_mnl_counterfactual.summary()
            out["estimated_latent_class_mnl"] = {
                "calibration_l2_error": float(lc_cal["l2_error"]),
                "calibration_rmse": float(lc_cal["rmse"]),
                "cf_l2_error": float(lc_cf["l2_error"]),
                "cf_rmse": float(lc_cf["rmse"]),
            }
            out["excess_l2_after_estimated_latent_class_mnl"] = {
                "calibration": float(lc_cal["l2_error"] - direct_cal["l2_error"]),
                "counterfactual": float(lc_cf["l2_error"] - direct_cf["l2_error"]),
            }
            out["homogeneous_minus_estimated_latent_class_l2"] = {
                "calibration": float(mnl_cal["l2_error"] - lc_cal["l2_error"]),
                "counterfactual": float(mnl_cf["l2_error"] - lc_cf["l2_error"]),
            }
            if seg_cal is not None and seg_cf is not None:
                out["estimated_minus_oracle_segmented_l2"] = {
                    "calibration": float(lc_cal["l2_error"] - seg_cal["l2_error"]),
                    "counterfactual": float(lc_cf["l2_error"] - seg_cf["l2_error"]),
                }
            if mix_cal is not None and mix_cf is not None:
                # This is the fair estimation-cost comparison: neither path
                # receives target segment labels.
                out["estimated_minus_oracle_mixture_l2"] = {
                    "calibration": float(lc_cal["l2_error"] - mix_cal["l2_error"]),
                    "counterfactual": float(lc_cf["l2_error"] - mix_cf["l2_error"]),
                }
        return out


def direct_oracle_probability_rows(
    target_df: pd.DataFrame,
    *,
    simulator_probability_column: str = "choice_prob",
    output_probability_column: str = "mnl_prob",
) -> pd.DataFrame:
    """Convert stored ground-truth simulator probabilities into model rows."""

    if simulator_probability_column not in target_df.columns:
        raise ValueError(
            f"target data must contain {simulator_probability_column!r}; "
            "regenerate the controlled benchmark with stored simulator probabilities"
        )
    out = target_df.copy()
    probs = out[simulator_probability_column].astype(float)
    if not np.isfinite(probs).all():
        raise ValueError("direct-oracle probabilities contain non-finite values")
    out[output_probability_column] = probs
    return out


def build_direct_oracle_report(
    *,
    anchor_target: pd.DataFrame,
    intervention_target: pd.DataFrame | None,
    config: CalibrationMomentConfig,
    weighting: str = "diagonal_uniform",
) -> CalibrationReport:
    """Build a calibration report without fitting an inner MNL model."""

    anchor_model = direct_oracle_probability_rows(
        anchor_target,
        output_probability_column=config.probability_column,
    )
    intervention_model = None
    if intervention_target is not None:
        intervention_model = direct_oracle_probability_rows(
            intervention_target,
            output_probability_column=config.probability_column,
        )
    return build_calibration_report(
        anchor_target=anchor_target,
        anchor_model=anchor_model,
        intervention_target=intervention_target,
        intervention_model=intervention_model,
        config=config,
        weighting=weighting,
    )


def build_economic_model_capacity_diagnostic(
    *,
    d_h: pd.DataFrame,
    d_calib_int: pd.DataFrame,
    d_cf: pd.DataFrame,
    oracle_through_mnl_calibration: CalibrationReport,
    oracle_through_mnl_counterfactual: CalibrationReport,
    config: CalibrationMomentConfig,
    oracle_segmented_mnl_calibration: CalibrationReport | None = None,
    oracle_segmented_mnl_counterfactual: CalibrationReport | None = None,
    oracle_mixture_mnl_calibration: CalibrationReport | None = None,
    oracle_mixture_mnl_counterfactual: CalibrationReport | None = None,
    estimated_latent_class_mnl_calibration: CalibrationReport | None = None,
    estimated_latent_class_mnl_counterfactual: CalibrationReport | None = None,
    extra_reports: dict[str, CalibrationReport] | None = None,
) -> EconomicModelCapacityDiagnostic:
    """Compare direct, routed, mixture, estimated-LC, and homogeneous paths."""

    paired = [
        (
            "segmented",
            oracle_segmented_mnl_calibration,
            oracle_segmented_mnl_counterfactual,
        ),
        (
            "oracle-mixture",
            oracle_mixture_mnl_calibration,
            oracle_mixture_mnl_counterfactual,
        ),
        (
            "estimated latent-class",
            estimated_latent_class_mnl_calibration,
            estimated_latent_class_mnl_counterfactual,
        ),
    ]
    for label, cal, cf in paired:
        if (cal is None) != (cf is None):
            raise ValueError(
                f"{label} calibration and counterfactual reports must either both be provided or both be omitted"
            )

    direct_cal = build_direct_oracle_report(
        anchor_target=d_h,
        intervention_target=d_calib_int,
        config=config,
    )
    direct_cf = build_direct_oracle_report(
        anchor_target=d_cf,
        intervention_target=None,
        config=config,
    )

    reports: dict[str, CalibrationReport] = {"direct_oracle": direct_cal}
    if oracle_segmented_mnl_calibration is not None:
        reports["oracle_segmented_mnl"] = oracle_segmented_mnl_calibration
    if oracle_mixture_mnl_calibration is not None:
        reports["oracle_mixture_mnl"] = oracle_mixture_mnl_calibration
    if estimated_latent_class_mnl_calibration is not None:
        reports["estimated_latent_class_mnl"] = estimated_latent_class_mnl_calibration
    reports["oracle_through_mnl"] = oracle_through_mnl_calibration
    if extra_reports:
        reports.update(extra_reports)
    comparison = compare_calibration_reports(reports)

    return EconomicModelCapacityDiagnostic(
        direct_oracle_calibration=direct_cal,
        direct_oracle_counterfactual=direct_cf,
        oracle_segmented_mnl_calibration=oracle_segmented_mnl_calibration,
        oracle_segmented_mnl_counterfactual=oracle_segmented_mnl_counterfactual,
        oracle_mixture_mnl_calibration=oracle_mixture_mnl_calibration,
        oracle_mixture_mnl_counterfactual=oracle_mixture_mnl_counterfactual,
        estimated_latent_class_mnl_calibration=estimated_latent_class_mnl_calibration,
        estimated_latent_class_mnl_counterfactual=estimated_latent_class_mnl_counterfactual,
        oracle_through_mnl_calibration=oracle_through_mnl_calibration,
        oracle_through_mnl_counterfactual=oracle_through_mnl_counterfactual,
        moment_comparison=comparison,
    )
