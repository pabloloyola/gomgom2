"""Diagnostics for the fixed-pair LLM simulator transfer gate.

These utilities intentionally operate on already-produced choice records so the
metrics can be tested offline before any remote-model credential is configured.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransferGateThresholds:
    min_parse_success: float = 0.98
    min_repeat_consistency: float = 0.85
    min_directional_accuracy: float = 0.75


@dataclass(frozen=True)
class TransferGateReport:
    n_requests: int
    parse_success: float
    repeat_consistency: float
    directional_accuracy: float
    choice_shares: dict[int, float]
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _choice_rows(records: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "repeat_id", "alternative_id", "chosen"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    chosen = records.loc[records["chosen"].astype(int) == 1].copy()
    if chosen.empty:
        return chosen
    counts = chosen.groupby(["pair_id", "repeat_id"], sort=False).size()
    if (counts > 1).any():
        raise ValueError("a pair/repeat has more than one chosen alternative")
    return chosen


def parse_success_rate(request_log: pd.DataFrame) -> float:
    """Fraction of attempted requests that produced a valid parsed choice."""
    if request_log.empty:
        return 0.0
    if "parse_ok" not in request_log.columns:
        raise ValueError("request_log must contain parse_ok")
    return float(request_log["parse_ok"].astype(bool).mean())


def repeat_consistency(records: pd.DataFrame) -> float:
    """Mean modal-choice agreement within each fixed persona-slate pair."""
    chosen = _choice_rows(records)
    if chosen.empty:
        return 0.0
    scores: list[float] = []
    for _, group in chosen.groupby("pair_id", sort=False):
        counts = group["alternative_id"].astype(int).value_counts()
        scores.append(float(counts.max() / counts.sum()))
    return float(np.mean(scores)) if scores else 0.0


def aggregate_choice_shares(records: pd.DataFrame) -> dict[int, float]:
    chosen = _choice_rows(records)
    if chosen.empty:
        return {}
    counts = chosen["alternative_id"].astype(int).value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in counts.items()}


def directional_accuracy(direction_checks: Iterable[bool]) -> float:
    values = [bool(x) for x in direction_checks]
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_transfer_gate(
    *,
    request_log: pd.DataFrame,
    choice_records: pd.DataFrame,
    direction_checks: Iterable[bool],
    thresholds: TransferGateThresholds | None = None,
) -> TransferGateReport:
    thresholds = thresholds or TransferGateThresholds()
    parse = parse_success_rate(request_log)
    repeat = repeat_consistency(choice_records)
    direction = directional_accuracy(direction_checks)
    shares = aggregate_choice_shares(choice_records)
    passed = (
        parse >= thresholds.min_parse_success
        and repeat >= thresholds.min_repeat_consistency
        and direction >= thresholds.min_directional_accuracy
    )
    return TransferGateReport(
        n_requests=int(len(request_log)),
        parse_success=parse,
        repeat_consistency=repeat,
        directional_accuracy=direction,
        choice_shares=shares,
        passed=bool(passed),
    )
