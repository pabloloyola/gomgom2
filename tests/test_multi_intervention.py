import numpy as np
import pandas as pd

from eipg.experiments import InterventionSpec, apply_intervention, combine_intervention_reports
from eipg.objectives.calibration import CalibrationMomentConfig, CalibrationReport


def _base_contexts():
    return pd.DataFrame(
        {
            "context_id": ["c0", "c0", "c1", "c1"],
            "alternative_id": [0, 1, 0, 1],
            "price": [1.0, 1.2, 0.8, 1.1],
            "quality": [0.4, 0.6, 0.9, 0.2],
            "sustain": [0.2, 0.7, 0.3, 0.8],
            "novelty": [0.1, 0.5, 0.4, 0.6],
            "brand": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_apply_intervention_changes_only_treated_alternative():
    base = _base_contexts()
    spec = InterventionSpec("quality", "quality", "add", 0.35, clip_min=0.0, clip_max=1.0)
    out = apply_intervention(base, spec)
    treated = out["alternative_id"] == 0
    untreated = ~treated
    assert np.allclose(out.loc[treated, "quality"].to_numpy(), [0.75, 1.0])
    assert np.allclose(out.loc[untreated, "quality"].to_numpy(), base.loc[untreated, "quality"].to_numpy())
    assert set(out["base_context_id"]) == {"c0", "c1"}
    assert set(out["intervention_type"]) == {"quality"}


def _report(intervention_shift: float) -> CalibrationReport:
    cfg = CalibrationMomentConfig()
    table = pd.DataFrame(
        [
            {
                "name": "anchor/item_shares/alt_0",
                "block": "anchor",
                "moment_type": "item_shares",
                "short_name": "alt_0",
                "target": 0.5,
                "model": 0.45,
                "diff": -0.05,
                "abs_diff": 0.05,
                "squared_diff": 0.0025,
                "weight": 1.0,
                "weighted_squared_diff": 0.0025,
            },
            {
                "name": "calib_int/intervention_response/delta_alt_1_share",
                "block": "calib_int",
                "moment_type": "intervention_response",
                "short_name": "delta_alt_1_share",
                "target": 0.1,
                "model": 0.1 + intervention_shift,
                "diff": intervention_shift,
                "abs_diff": abs(intervention_shift),
                "squared_diff": intervention_shift**2,
                "weight": 1.0,
                "weighted_squared_diff": intervention_shift**2,
            },
        ]
    )
    return CalibrationReport(table=table, config=cfg)


def test_combine_intervention_reports_keeps_anchor_once_and_namespaces_rows():
    combined = combine_intervention_reports([("price", _report(0.02)), ("quality", _report(0.04))])
    assert int((combined.table["block"] == "anchor").sum()) == 1
    intervention_blocks = set(combined.table.loc[combined.table["block"] != "anchor", "block"])
    assert intervention_blocks == {"calib_int/price", "calib_int/quality"}
    assert combined.table["name"].is_unique
