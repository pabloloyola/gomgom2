"""Calibration moments for EIPG.

Paper alignment
---------------
This module implements the milestone where the fitted economic model
``m_{beta*(phi)}`` is compared against human-anchor behavior.

In the paper notation, we build two vectors:

    M_tar  : target moments derived from D_H and calibration interventions
    M_phi  : model-implied moments from p_{beta*(phi)}(. | x)

The calibration loss is then a weighted discrepancy

    (M_phi - M_tar)^T W (M_phi - M_tar).

The implementation is intentionally transparent: every scalar moment is also
returned as a row in a table so it can be inspected in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json

import numpy as np
import pandas as pd


DEFAULT_ATTRIBUTE_FEATURES: tuple[str, ...] = (
    "price",
    "quality",
    "sustain",
    "novelty",
    "brand",
)

DEFAULT_MOMENT_TYPES: tuple[str, ...] = (
    "item_shares",
    "group_shares",
    "chosen_attributes",
    "intervention_response",
)


@dataclass(frozen=True)
class CalibrationMomentConfig:
    """Configuration for calibration moments.

    The config mirrors the paper's moment-vector description but keeps the first
    implementation small and auditable.
    """

    moment_types: tuple[str, ...] = DEFAULT_MOMENT_TYPES
    attribute_features: tuple[str, ...] = DEFAULT_ATTRIBUTE_FEATURES
    group_column: str = "brand"
    probability_column: str = "mnl_prob"
    chosen_column: str = "chosen"
    intervened_alternative_id: int = 0
    drop_redundant_moments: bool = True
    intervention_share_response: str = "full_substitution"
    substitution_reference_alternative_id: int | None = None

    def __post_init__(self) -> None:
        if not self.moment_types:
            raise ValueError("moment_types must be non-empty")
        unknown = set(self.moment_types) - set(DEFAULT_MOMENT_TYPES)
        if unknown:
            raise ValueError(f"unknown calibration moment types: {sorted(unknown)}")
        if not self.attribute_features:
            raise ValueError("attribute_features must be non-empty")
        allowed_response = {"intervened_only", "full_substitution"}
        if self.intervention_share_response not in allowed_response:
            raise ValueError(
                "intervention_share_response must be one of "
                f"{sorted(allowed_response)}"
            )

    @classmethod
    def from_config(
        cls,
        section: dict[str, Any],
        *,
        default_features: Iterable[str] | None = None,
        intervened_alternative_id: int = 0,
    ) -> "CalibrationMomentConfig":
        return cls(
            moment_types=tuple(section.get("moments") or DEFAULT_MOMENT_TYPES),
            attribute_features=tuple(
                section.get("attribute_features") or tuple(default_features or DEFAULT_ATTRIBUTE_FEATURES)
            ),
            group_column=str(section.get("group_column", "brand")),
            probability_column=str(section.get("probability_column", "mnl_prob")),
            chosen_column=str(section.get("chosen_column", "chosen")),
            intervened_alternative_id=int(
                section.get("intervened_alternative_id", intervened_alternative_id)
            ),
            drop_redundant_moments=bool(section.get("drop_redundant_moments", True)),
            intervention_share_response=str(
                section.get("intervention_share_response", "full_substitution")
            ),
            substitution_reference_alternative_id=(
                int(section["substitution_reference_alternative_id"])
                if section.get("substitution_reference_alternative_id") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "moment_types": list(self.moment_types),
            "attribute_features": list(self.attribute_features),
            "group_column": self.group_column,
            "probability_column": self.probability_column,
            "chosen_column": self.chosen_column,
            "intervened_alternative_id": int(self.intervened_alternative_id),
            "drop_redundant_moments": bool(self.drop_redundant_moments),
            "intervention_share_response": self.intervention_share_response,
            "substitution_reference_alternative_id": self.substitution_reference_alternative_id,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Complete moment-comparison result."""

    table: pd.DataFrame
    config: CalibrationMomentConfig
    weighting: str = "diagonal_uniform"

    def summary(self) -> dict[str, Any]:
        if self.table.empty:
            return {
                "n_moments": 0,
                "l2_error": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "max_abs_error": 0.0,
            }
        diff = self.table["diff"].to_numpy(dtype=float)
        abs_diff = np.abs(diff)
        squared = diff * diff
        return {
            "n_moments": int(len(diff)),
            "l2_error": float(np.sqrt(np.sum(squared))),
            "rmse": float(np.sqrt(np.mean(squared))),
            "mae": float(np.mean(abs_diff)),
            "max_abs_error": float(np.max(abs_diff)),
            "weighting": self.weighting,
            "config": self.config.to_dict(),
            "error_by_block": _error_by_column(self.table, "block"),
            "error_by_moment_type": _error_by_column(self.table, "moment_type"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "moments": self.table.to_dict(orient="records"),
        }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out


def _error_by_column(table: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for value, group in table.groupby(column, sort=True):
        diff = group["diff"].to_numpy(dtype=float)
        abs_diff = np.abs(diff)
        squared = diff * diff
        out[str(value)] = {
            "n_moments": int(len(diff)),
            "l2_error": float(np.sqrt(np.sum(squared))),
            "rmse": float(np.sqrt(np.mean(squared))),
            "mae": float(np.mean(abs_diff)),
            "max_abs_error": float(np.max(abs_diff)),
        }
    return out


def _validate_choice_df(df: pd.DataFrame, *, weight_col: str | None = None) -> None:
    required = {"observation_id", "alternative_id"}
    if weight_col is not None:
        required.add(weight_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"data missing required columns: {missing}")
    if df.empty:
        raise ValueError("data must be non-empty")


def _weighted_rows(df: pd.DataFrame, *, weight_col: str) -> pd.DataFrame:
    _validate_choice_df(df, weight_col=weight_col)
    out = df.copy()
    out["_moment_weight"] = out[weight_col].astype(float)
    return out


def _empirical_rows(df: pd.DataFrame, *, chosen_col: str) -> pd.DataFrame:
    _validate_choice_df(df, weight_col=chosen_col)
    out = df.copy()
    out["_moment_weight"] = out[chosen_col].astype(float)
    return out


def _n_observations(df: pd.DataFrame) -> int:
    return max(int(df["observation_id"].nunique()), 1)


def moment_series_from_long(
    df: pd.DataFrame,
    *,
    block: str,
    config: CalibrationMomentConfig,
    use_probabilities: bool,
) -> pd.Series:
    """Compute a named moment vector from long-format choice rows.

    If ``use_probabilities`` is False, moments are empirical moments using the
    ``chosen`` column. If True, moments are model-implied moments using
    ``mnl_prob``. Item-level shares are normalized by the number of contexts in
    which the alternative is available, matching the availability-conditional
    definition used in the paper. Other aggregate moments retain their natural
    observation-level normalization.
    """

    if use_probabilities:
        rows = _weighted_rows(df, weight_col=config.probability_column)
    else:
        rows = _empirical_rows(df, chosen_col=config.chosen_column)

    n_obs = _n_observations(rows)
    values: dict[str, float] = {}

    if "item_shares" in config.moment_types:
        for alt in sorted(rows["alternative_id"].dropna().unique().tolist()):
            mask = rows["alternative_id"] == alt
            # Item-level shares are conditional on availability. In long format,
            # an alternative is available in exactly those observations for which
            # it has a row. This keeps an item from being penalized simply because
            # it is absent from some choice sets.
            n_available = max(int(rows.loc[mask, "observation_id"].nunique()), 1)
            values[f"{block}/item_shares/alt_{int(alt)}"] = float(
                rows.loc[mask, "_moment_weight"].sum() / n_available
            )

    if "group_shares" in config.moment_types and config.group_column in rows.columns:
        group_values = sorted(rows[config.group_column].dropna().unique().tolist())
        # Group shares sum to one when every chosen alternative belongs to one
        # group. With uniform diagonal weighting, including every category gives
        # the same composition constraint multiple times. In the paper-facing
        # objective we therefore use K-1 shares, dropping the first sorted value
        # as a reference category. Setting drop_redundant_moments=False restores
        # the fully expanded diagnostic vector.
        if config.drop_redundant_moments and len(group_values) > 1:
            group_values = group_values[1:]
        for group_value in group_values:
            mask = rows[config.group_column] == group_value
            label = _clean_value_label(group_value)
            values[f"{block}/group_shares/{config.group_column}_{label}"] = float(
                rows.loc[mask, "_moment_weight"].sum() / n_obs
            )

    if "chosen_attributes" in config.moment_types:
        for attr in config.attribute_features:
            if attr not in rows.columns:
                continue
            # In the current benchmark ``brand`` is a binary 0/1 group label.
            # Its chosen mean is therefore exactly the brand_1 group share. Do
            # not count the same scalar twice in the uniform-weight objective.
            if (
                config.drop_redundant_moments
                and "group_shares" in config.moment_types
                and attr == config.group_column
                and _is_binary_indicator(rows[attr])
            ):
                continue
            weighted_sum = (rows[attr].astype(float) * rows["_moment_weight"]).sum()
            total_weight = rows["_moment_weight"].sum()
            values[f"{block}/chosen_attributes/mean_{attr}"] = float(
                weighted_sum / max(float(total_weight), 1.0e-12)
            )

    return pd.Series(values, dtype=float)



def _is_binary_indicator(series: pd.Series) -> bool:
    """Return True when a column contains only numeric 0/1 values.

    This is used only for exact redundancy removal. A non-binary grouping
    variable may have a meaningful numeric mean, so it is not dropped.
    """

    values = pd.to_numeric(series.dropna(), errors="coerce")
    if values.empty or values.isna().any():
        return False
    unique = set(values.astype(float).unique().tolist())
    return unique.issubset({0.0, 1.0}) and bool(unique)

def _clean_value_label(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(" ", "_").replace("/", "_")


def intervention_response_series(
    *,
    anchor_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    block: str,
    config: CalibrationMomentConfig,
    use_probabilities: bool,
) -> pd.Series:
    """Compute intervention-response moments.

    Besides the response of the directly intervened alternative, the
    ``full_substitution`` mode records how choice share moves toward or away
    from the other alternatives.  These moments are intended to constrain the
    substitution pattern rather than only the marginal response of the treated
    item.

    When ``drop_redundant_moments`` is active, one non-intervened alternative
    is omitted as a reference because the complete vector of share changes sums
    to zero when every alternative is available in every slate.
    """

    if use_probabilities:
        anchor_rows = _weighted_rows(anchor_df, weight_col=config.probability_column)
        intervention_rows = _weighted_rows(intervention_df, weight_col=config.probability_column)
    else:
        anchor_rows = _empirical_rows(anchor_df, chosen_col=config.chosen_column)
        intervention_rows = _empirical_rows(intervention_df, chosen_col=config.chosen_column)

    def _availability_share(rows: pd.DataFrame, alt_id: int) -> float:
        mask = rows["alternative_id"].astype(int) == int(alt_id)
        n_available = max(int(rows.loc[mask, "observation_id"].nunique()), 1)
        return float(rows.loc[mask, "_moment_weight"].sum() / n_available)

    alternatives = sorted(
        set(anchor_rows["alternative_id"].astype(int).unique().tolist())
        | set(intervention_rows["alternative_id"].astype(int).unique().tolist())
    )
    alt = int(config.intervened_alternative_id)
    if alt not in alternatives:
        raise ValueError(f"intervened alternative {alt} is absent from intervention data")

    anchor_shares = {a: _availability_share(anchor_rows, a) for a in alternatives}
    intervention_shares = {a: _availability_share(intervention_rows, a) for a in alternatives}

    values: dict[str, float] = {
        f"{block}/intervention_response/delta_intervened_alt_{alt}_share":
            intervention_shares[alt] - anchor_shares[alt],
    }

    if config.intervention_share_response == "full_substitution":
        non_intervened = [a for a in alternatives if a != alt]
        reference: int | None = None
        if config.drop_redundant_moments and non_intervened:
            requested = config.substitution_reference_alternative_id
            if requested is not None and requested in non_intervened:
                reference = int(requested)
            else:
                # Deterministic reference category: the largest non-intervened id.
                reference = int(non_intervened[-1])
        for other in non_intervened:
            if reference is not None and int(other) == reference:
                continue
            values[f"{block}/intervention_response/delta_alt_{int(other)}_share"] = (
                intervention_shares[int(other)] - anchor_shares[int(other)]
            )

    # The post-intervention share is already present as
    # ``calib_int/item_shares/alt_<id>`` when item-share moments are active.
    if not (config.drop_redundant_moments and "item_shares" in config.moment_types):
        values[f"{block}/intervention_response/intervened_alt_{alt}_share_after"] = (
            intervention_shares[alt]
        )

    for attr in config.attribute_features:
        if attr not in anchor_rows.columns or attr not in intervention_rows.columns:
            continue
        anchor_total = anchor_rows["_moment_weight"].sum()
        int_total = intervention_rows["_moment_weight"].sum()
        anchor_mean = float(
            (anchor_rows[attr].astype(float) * anchor_rows["_moment_weight"]).sum()
            / max(float(anchor_total), 1.0e-12)
        )
        int_mean = float(
            (intervention_rows[attr].astype(float) * intervention_rows["_moment_weight"]).sum()
            / max(float(int_total), 1.0e-12)
        )
        values[f"{block}/intervention_response/delta_mean_{attr}"] = int_mean - anchor_mean

    return pd.Series(values, dtype=float)


def build_calibration_report(
    *,
    anchor_target: pd.DataFrame,
    anchor_model: pd.DataFrame,
    intervention_target: pd.DataFrame | None = None,
    intervention_model: pd.DataFrame | None = None,
    config: CalibrationMomentConfig | None = None,
    weighting: str = "diagonal_uniform",
) -> CalibrationReport:
    """Build the calibration moment comparison table.

    Parameters
    ----------
    anchor_target:
        Empirical human-anchor choices ``D_H``.
    anchor_model:
        MNL predicted probabilities on the same anchor contexts.
    intervention_target:
        Empirical/synthetic target choices for calibration-intervention contexts.
    intervention_model:
        MNL predicted probabilities on calibration-intervention contexts.
    config:
        Moment configuration.
    weighting:
        Currently only ``diagonal_uniform`` is implemented; the field is stored
        so later versions can add GMM weights without changing the output schema.
    """

    config = config or CalibrationMomentConfig()
    if weighting != "diagonal_uniform":
        raise ValueError("v0.6 implements only diagonal_uniform weighting")

    target_parts: list[pd.Series] = [
        moment_series_from_long(anchor_target, block="anchor", config=config, use_probabilities=False)
    ]
    model_parts: list[pd.Series] = [
        moment_series_from_long(anchor_model, block="anchor", config=config, use_probabilities=True)
    ]

    has_intervention = intervention_target is not None and intervention_model is not None
    if has_intervention:
        assert intervention_target is not None
        assert intervention_model is not None
        target_parts.append(
            moment_series_from_long(
                intervention_target, block="calib_int", config=config, use_probabilities=False
            )
        )
        model_parts.append(
            moment_series_from_long(
                intervention_model, block="calib_int", config=config, use_probabilities=True
            )
        )
        if "intervention_response" in config.moment_types:
            target_parts.append(
                intervention_response_series(
                    anchor_df=anchor_target,
                    intervention_df=intervention_target,
                    block="calib_int",
                    config=config,
                    use_probabilities=False,
                )
            )
            model_parts.append(
                intervention_response_series(
                    anchor_df=anchor_model,
                    intervention_df=intervention_model,
                    block="calib_int",
                    config=config,
                    use_probabilities=True,
                )
            )

    target = pd.concat(target_parts)
    model = pd.concat(model_parts)
    all_names = sorted(set(target.index).union(set(model.index)))

    records: list[dict[str, Any]] = []
    for name in all_names:
        target_value = float(target.get(name, 0.0))
        model_value = float(model.get(name, 0.0))
        diff = model_value - target_value
        block, moment_type, short_name = _split_moment_name(name)
        records.append(
            {
                "name": name,
                "block": block,
                "moment_type": moment_type,
                "short_name": short_name,
                "target": target_value,
                "model": model_value,
                "diff": float(diff),
                "abs_diff": float(abs(diff)),
                "squared_diff": float(diff * diff),
                "weight": 1.0,
                "weighted_squared_diff": float(diff * diff),
            }
        )

    table = pd.DataFrame.from_records(records)
    return CalibrationReport(table=table, config=config, weighting=weighting)


def _split_moment_name(name: str) -> tuple[str, str, str]:
    parts = name.split("/", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", "", name


def compute_moment_error(table: pd.DataFrame) -> float:
    """Return the uniform-weight GMM-style calibration loss."""

    if table.empty:
        return 0.0
    diff = table["diff"].to_numpy(dtype=float)
    weights = table.get("weight", pd.Series(np.ones(len(table)))).to_numpy(dtype=float)
    return float(np.sum(weights * diff * diff))


def compare_calibration_reports(reports: dict[str, CalibrationReport]) -> pd.DataFrame:
    """Merge moment-level calibration reports for side-by-side diagnostics.

    Parameters
    ----------
    reports:
        Mapping from a short candidate label (for example ``static``,
        ``eipg``, or ``oracle``) to the corresponding :class:`CalibrationReport`.

    Returns
    -------
    pandas.DataFrame
        One row per behavioral moment. The shared target is stored once, while
        each candidate contributes ``<label>_model``, ``<label>_diff``, and
        ``<label>_abs_error`` columns. If both ``static`` and ``eipg`` are
        present, the table also includes positive-valued improvement columns,
        where a positive value means EIPG reduced the error relative to static.

    Notes
    -----
    The function validates that all reports refer to the same moment names and
    target values. This matters for paper diagnostics: a before/after comparison
    is meaningful only when every candidate is scored against the identical
    calibration target.
    """

    if not reports:
        raise ValueError("reports must contain at least one CalibrationReport")

    key_cols = ["name", "block", "moment_type", "short_name"]
    merged: pd.DataFrame | None = None

    for label, report in reports.items():
        clean_label = str(label).strip().replace(" ", "_")
        table = report.table.copy()
        required = set(key_cols + ["target", "model", "diff", "abs_diff"])
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"report {label!r} missing required columns: {missing}")

        current = table[key_cols + ["target", "model", "diff", "abs_diff"]].copy()
        current = current.rename(
            columns={
                "target": f"{clean_label}_target",
                "model": f"{clean_label}_model",
                "diff": f"{clean_label}_diff",
                "abs_diff": f"{clean_label}_abs_error",
            }
        )
        merged = current if merged is None else merged.merge(current, on=key_cols, how="outer", validate="one_to_one")

    assert merged is not None
    target_cols = [c for c in merged.columns if c.endswith("_target")]
    if merged[target_cols].isna().any().any():
        raise ValueError("candidate reports do not contain the same moment names")

    target_values = merged[target_cols].to_numpy(dtype=float)
    if target_values.shape[1] > 1 and not np.allclose(
        target_values, target_values[:, [0]], rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("candidate reports use inconsistent target moment values")

    merged.insert(4, "target", target_values[:, 0])
    merged = merged.drop(columns=target_cols)

    if {"static_abs_error", "eipg_abs_error"}.issubset(merged.columns):
        merged["abs_error_improvement_static_to_eipg"] = (
            merged["static_abs_error"] - merged["eipg_abs_error"]
        )
        merged["squared_error_improvement_static_to_eipg"] = (
            merged["static_diff"] ** 2 - merged["eipg_diff"] ** 2
        )

    return merged.sort_values(key_cols).reset_index(drop=True)
