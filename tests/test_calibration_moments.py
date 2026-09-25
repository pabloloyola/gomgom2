from __future__ import annotations

import numpy as np
import pandas as pd

from eipg.objectives import (
    CalibrationMomentConfig,
    build_calibration_report,
    compute_moment_error,
    compare_calibration_reports,
    moment_series_from_long,
)


def _toy_long_df(with_prob: bool = False) -> pd.DataFrame:
    rows = []
    # Two observations, two alternatives each.
    for obs, chosen_alt in [("o1", 0), ("o2", 1)]:
        for alt in [0, 1]:
            rec = {
                "observation_id": obs,
                "alternative_id": alt,
                "chosen": 1 if alt == chosen_alt else 0,
                "price": 1.0 + alt,
                "quality": 0.2 + alt,
                "sustain": 0.0,
                "novelty": 0.0,
                "brand": float(alt),
            }
            if with_prob:
                rec["mnl_prob"] = 0.5
            rows.append(rec)
    return pd.DataFrame(rows)


def _varying_availability_df(with_prob: bool = False) -> pd.DataFrame:
    """Two observations where alternative 1 is available only once."""

    rows = [
        {
            "observation_id": "o1",
            "alternative_id": 0,
            "chosen": 0,
            "price": 1.0,
            "quality": 0.2,
            "sustain": 0.0,
            "novelty": 0.0,
            "brand": 0.0,
        },
        {
            "observation_id": "o1",
            "alternative_id": 1,
            "chosen": 1,
            "price": 2.0,
            "quality": 1.2,
            "sustain": 0.0,
            "novelty": 0.0,
            "brand": 1.0,
        },
        {
            "observation_id": "o2",
            "alternative_id": 0,
            "chosen": 1,
            "price": 1.1,
            "quality": 0.3,
            "sustain": 0.0,
            "novelty": 0.0,
            "brand": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    if with_prob:
        df["mnl_prob"] = [0.25, 0.75, 1.0]
    return df


def test_empirical_moment_series_item_shares_sum_to_one() -> None:
    cfg = CalibrationMomentConfig(moment_types=("item_shares",))
    moments = moment_series_from_long(
        _toy_long_df(), block="anchor", config=cfg, use_probabilities=False
    )
    assert np.isclose(moments.sum(), 1.0)
    assert np.isclose(moments["anchor/item_shares/alt_0"], 0.5)
    assert np.isclose(moments["anchor/item_shares/alt_1"], 0.5)


def test_model_moments_use_mnl_probabilities() -> None:
    cfg = CalibrationMomentConfig(moment_types=("item_shares",))
    moments = moment_series_from_long(
        _toy_long_df(with_prob=True), block="anchor", config=cfg, use_probabilities=True
    )
    assert np.isclose(moments["anchor/item_shares/alt_0"], 0.5)
    assert np.isclose(moments["anchor/item_shares/alt_1"], 0.5)


def test_item_shares_are_conditional_on_alternative_availability() -> None:
    cfg = CalibrationMomentConfig(moment_types=("item_shares",))

    empirical = moment_series_from_long(
        _varying_availability_df(),
        block="anchor",
        config=cfg,
        use_probabilities=False,
    )
    model = moment_series_from_long(
        _varying_availability_df(with_prob=True),
        block="anchor",
        config=cfg,
        use_probabilities=True,
    )

    # Alternative 0 is available in both observations and chosen once.
    assert np.isclose(empirical["anchor/item_shares/alt_0"], 0.5)
    # Alternative 1 is available only in o1 and is chosen there, so its
    # availability-conditional share is 1 rather than 1/2.
    assert np.isclose(empirical["anchor/item_shares/alt_1"], 1.0)
    # The model-implied share uses the same availability denominator.
    assert np.isclose(model["anchor/item_shares/alt_1"], 0.75)


def test_calibration_report_has_error_rows() -> None:
    cfg = CalibrationMomentConfig(moment_types=("item_shares", "chosen_attributes"))
    target = _toy_long_df()
    model = _toy_long_df(with_prob=True)
    report = build_calibration_report(anchor_target=target, anchor_model=model, config=cfg)
    assert not report.table.empty
    assert "diff" in report.table.columns
    assert report.summary()["n_moments"] == len(report.table)
    assert compute_moment_error(report.table) >= 0.0


def test_redundant_binary_group_moments_are_dropped_by_default() -> None:
    cfg = CalibrationMomentConfig(
        moment_types=("group_shares", "chosen_attributes"),
        attribute_features=("price", "brand"),
        group_column="brand",
    )
    moments = moment_series_from_long(
        _toy_long_df(), block="anchor", config=cfg, use_probabilities=False
    )

    # brand_0 is the reference category; mean_brand would exactly duplicate
    # brand_1 because brand is encoded as a binary 0/1 feature.
    assert "anchor/group_shares/brand_0" not in moments.index
    assert "anchor/group_shares/brand_1" in moments.index
    assert "anchor/chosen_attributes/mean_brand" not in moments.index
    assert "anchor/chosen_attributes/mean_price" in moments.index


def test_expanded_moment_vector_can_be_requested_for_diagnostics() -> None:
    cfg = CalibrationMomentConfig(
        moment_types=("group_shares", "chosen_attributes"),
        attribute_features=("price", "brand"),
        group_column="brand",
        drop_redundant_moments=False,
    )
    moments = moment_series_from_long(
        _toy_long_df(), block="anchor", config=cfg, use_probabilities=False
    )

    assert "anchor/group_shares/brand_0" in moments.index
    assert "anchor/group_shares/brand_1" in moments.index
    assert "anchor/chosen_attributes/mean_brand" in moments.index


def test_intervention_after_share_is_not_duplicated_when_item_shares_active() -> None:
    from eipg.objectives.calibration import intervention_response_series

    cfg = CalibrationMomentConfig(
        moment_types=("item_shares", "intervention_response"),
        intervened_alternative_id=0,
    )
    anchor = _toy_long_df()
    intervention = _toy_long_df().copy()
    response = intervention_response_series(
        anchor_df=anchor,
        intervention_df=intervention,
        block="calib_int",
        config=cfg,
        use_probabilities=False,
    )

    assert "calib_int/intervention_response/delta_intervened_alt_0_share" in response.index
    assert "calib_int/intervention_response/intervened_alt_0_share_after" not in response.index


def _three_alt_response_df(chosen_by_obs: dict[str, int]) -> pd.DataFrame:
    rows = []
    for obs, chosen_alt in chosen_by_obs.items():
        for alt in [0, 1, 2]:
            rows.append(
                {
                    "observation_id": obs,
                    "alternative_id": alt,
                    "chosen": 1 if alt == chosen_alt else 0,
                    "price": 1.0 + 0.1 * alt,
                    "quality": 0.2 + 0.1 * alt,
                    "sustain": 0.0,
                    "novelty": 0.0,
                    "brand": float(alt % 2),
                }
            )
    return pd.DataFrame(rows)


def test_full_substitution_response_adds_nonintervened_share_changes() -> None:
    from eipg.objectives.calibration import intervention_response_series

    cfg = CalibrationMomentConfig(
        moment_types=("intervention_response",),
        attribute_features=("price",),
        intervened_alternative_id=0,
        intervention_share_response="full_substitution",
        drop_redundant_moments=True,
    )
    anchor = _three_alt_response_df({"o1": 0, "o2": 2})
    intervention = _three_alt_response_df({"o1": 1, "o2": 2})
    response = intervention_response_series(
        anchor_df=anchor,
        intervention_df=intervention,
        block="calib_int",
        config=cfg,
        use_probabilities=False,
    )

    assert np.isclose(
        response["calib_int/intervention_response/delta_intervened_alt_0_share"], -0.5
    )
    assert np.isclose(
        response["calib_int/intervention_response/delta_alt_1_share"], 0.5
    )
    # alt_2 is the deterministic reference category, so the linearly dependent
    # final share-change coordinate is omitted.
    assert "calib_int/intervention_response/delta_alt_2_share" not in response.index


def test_intervened_only_response_preserves_legacy_behavior() -> None:
    from eipg.objectives.calibration import intervention_response_series

    cfg = CalibrationMomentConfig(
        moment_types=("intervention_response",),
        attribute_features=("price",),
        intervened_alternative_id=0,
        intervention_share_response="intervened_only",
    )
    anchor = _three_alt_response_df({"o1": 0, "o2": 2})
    intervention = _three_alt_response_df({"o1": 1, "o2": 2})
    response = intervention_response_series(
        anchor_df=anchor,
        intervention_df=intervention,
        block="calib_int",
        config=cfg,
        use_probabilities=False,
    )
    assert not any("delta_alt_1_share" in name for name in response.index)



def test_compare_calibration_reports_builds_moment_level_before_after_table() -> None:
    cfg = CalibrationMomentConfig(moment_types=("item_shares",))
    target = _toy_long_df()

    static_model = _toy_long_df(with_prob=True)
    # A deliberately better model for the two observed choices.
    eipg_model = _toy_long_df(with_prob=True)
    eipg_model["mnl_prob"] = [0.75, 0.25, 0.25, 0.75]

    static = build_calibration_report(
        anchor_target=target,
        anchor_model=static_model,
        config=cfg,
    )
    eipg = build_calibration_report(
        anchor_target=target,
        anchor_model=eipg_model,
        config=cfg,
    )

    comparison = compare_calibration_reports({"static": static, "eipg": eipg})
    assert len(comparison) == len(static.table)
    assert {"target", "static_model", "eipg_model"}.issubset(comparison.columns)
    assert {
        "static_abs_error",
        "eipg_abs_error",
        "abs_error_improvement_static_to_eipg",
    }.issubset(comparison.columns)
    assert np.allclose(comparison["target"], static.table.sort_values("name")["target"])
    assert (comparison["abs_error_improvement_static_to_eipg"] >= 0).all()
