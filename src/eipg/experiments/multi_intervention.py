"""Multiple-calibration-intervention identification diagnostics.

The goal is to vary the *information content* of calibration while holding the
inner economic model and candidate bank fixed.  Interventions are applied to a
shared set of anchor contexts and each intervention contributes its own moment
rows.  Anchor moments are included exactly once when reports are combined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from eipg.objectives.calibration import CalibrationReport


@dataclass(frozen=True)
class InterventionSpec:
    """One deterministic perturbation applied to one alternative in each slate."""

    name: str
    attribute: str
    mode: str
    value: float
    intervened_alternative_id: int = 0
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("intervention name must be non-empty")
        if self.mode not in {"multiply", "add", "set"}:
            raise ValueError("mode must be one of: multiply, add, set")


def apply_intervention(
    base: pd.DataFrame,
    spec: InterventionSpec,
    *,
    prefix: str | None = None,
    context_set: str | None = None,
    n_contexts: int | None = None,
) -> pd.DataFrame:
    """Return a perturbed copy of long-format base contexts.

    The first ``n_contexts`` sorted context ids are used, making candidate and
    intervention comparisons deterministic.  ``base_context_id`` preserves the
    source context id for auditing.
    """

    required = {"context_id", "alternative_id", spec.attribute}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"base contexts missing columns: {sorted(missing)}")
    if base.empty:
        raise ValueError("base contexts must be non-empty")

    ids = sorted(base["context_id"].astype(str).unique().tolist())
    if n_contexts is not None:
        if int(n_contexts) <= 0:
            raise ValueError("n_contexts must be positive when supplied")
        ids = ids[: int(n_contexts)]
    out = base[base["context_id"].astype(str).isin(ids)].copy()

    stem = prefix or f"calib_{spec.name}"
    id_map = {old: f"{stem}_{i:04d}" for i, old in enumerate(ids)}
    out["base_context_id"] = out["context_id"].astype(str)
    out["context_id"] = out["context_id"].astype(str).map(id_map)
    out["context_set"] = context_set or f"X_calib_{spec.name}"
    out["intervention_type"] = spec.name
    out["intervened_alternative_id"] = int(spec.intervened_alternative_id)

    mask = out["alternative_id"].astype(int) == int(spec.intervened_alternative_id)
    current = out.loc[mask, spec.attribute].astype(float)
    if spec.mode == "multiply":
        changed = current * float(spec.value)
    elif spec.mode == "add":
        changed = current + float(spec.value)
    else:
        changed = np.full(len(current), float(spec.value), dtype=float)
    if spec.clip_min is not None or spec.clip_max is not None:
        lower = -np.inf if spec.clip_min is None else float(spec.clip_min)
        upper = np.inf if spec.clip_max is None else float(spec.clip_max)
        changed = np.clip(np.asarray(changed, dtype=float), lower, upper)
    out.loc[mask, spec.attribute] = np.asarray(changed, dtype=float)
    return out.reset_index(drop=True)


def combine_intervention_reports(
    reports: Iterable[tuple[str, CalibrationReport]],
) -> CalibrationReport:
    """Combine per-intervention reports, including shared anchor moments once.

    Each input report must use the same calibration config.  Intervention rows
    are namespaced by intervention name so moments from distinct perturbations
    cannot collide.  This allows the existing block-normalized objective to be
    reused without changing the paper-facing single-intervention implementation.
    """

    pairs = list(reports)
    if not pairs:
        raise ValueError("reports must be non-empty")
    config = pairs[0][1].config
    anchor: pd.DataFrame | None = None
    pieces: list[pd.DataFrame] = []

    for name, report in pairs:
        if report.config != config:
            raise ValueError("all reports must share the same CalibrationMomentConfig")
        table = report.table.copy()
        this_anchor = table[table["block"].astype(str) == "anchor"].copy()
        if anchor is None:
            anchor = this_anchor
        else:
            left = anchor[["name", "target", "model"]].reset_index(drop=True)
            right = this_anchor[["name", "target", "model"]].reset_index(drop=True)
            if len(left) != len(right) or not np.allclose(
                left[["target", "model"]].to_numpy(float),
                right[["target", "model"]].to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError("anchor moments differ across intervention reports")

        intervention = table[table["block"].astype(str) != "anchor"].copy()
        intervention["block"] = f"calib_int/{name}"
        intervention["name"] = [
            f"calib_int/{name}/{mt}/{sn}"
            for mt, sn in zip(intervention["moment_type"], intervention["short_name"])
        ]
        pieces.append(intervention)

    assert anchor is not None
    combined = pd.concat([anchor, *pieces], ignore_index=True)
    return CalibrationReport(table=combined, config=config, weighting=pairs[0][1].weighting)
