from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoembeddings.visualization_diagnostics import calculate_mobility_diagnostics, summarize_diagnostics


def _users() -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": "u1", "age_group": "18-29", "household_type": "single"},
        {"user_id": "u2", "age_group": "65+", "household_type": "family"},
        {"user_id": "u3", "age_group": None, "household_type": None},
    ])


def _events() -> pd.DataFrame:
    return pd.DataFrame([
        # Input order is deliberately opposite to UTC/local chronological order.
        {"user_id": "u1", "timestamp": "2026-01-01T01:00:00Z", "latitude": 0.0, "longitude": 1.0, "region_id": "b", "observation_mode": "passive"},
        {"user_id": "u1", "timestamp": "2026-01-01T00:00:00Z", "latitude": 0.0, "longitude": 0.0, "region_id": "a", "observation_mode": "passive"},
        {"user_id": "u1", "timestamp": "2026-01-01T01:00:00Z", "latitude": 0.0, "longitude": 2.0, "region_id": "b", "observation_mode": "passive"},
        {"user_id": "u2", "timestamp": "2025-12-31T15:30:00Z", "latitude": np.nan, "longitude": np.nan, "region_id": "a", "observation_mode": "passive"},
    ])


def test_distance_speed_timezone_order_zero_duration_and_missing_coordinates() -> None:
    per_user, transitions = calculate_mobility_diagnostics(_events(), _users())
    u1 = transitions.loc[transitions.user_id.eq("u1")].reset_index(drop=True)
    assert u1.loc[0, "distance_km"] == pytest.approx(111.195, rel=1e-4)
    assert u1.loc[0, "elapsed_hours"] == 1.0
    assert u1.loc[0, "speed_kmh"] == pytest.approx(111.195, rel=1e-4)
    assert bool(u1.loc[1, "zero_duration"])
    assert np.isnan(u1.loc[1, "speed_kmh"])
    assert per_user.set_index("user_id").loc["u2", "unique_location_count"] != 0
    # UTC midnight does not split both u1 points in Asia/Tokyo.
    assert per_user.set_index("user_id").loc["u1", "events_per_day"] == 3.0


def test_demographic_support_sizes_and_missingness_are_machine_readable() -> None:
    per_user, transitions = calculate_mobility_diagnostics(_events(), _users())
    report = summarize_diagnostics(per_user, {"straight_line_speed_kmh": {"max": 50}}, transitions)
    age = report["distributions"]["strata"]["age_group"]
    assert age["18-29"]["sample_size"] == 1
    assert age["<missing>"]["demographic_missing_count"] == 1
    assert report["behavioral_warnings"]
    assert any(item["check"] == "event_coordinates_present" for item in report["integrity_failures"])
    assert "not evidence" in report["interpretation"]


def test_truth_metrics_require_explicit_truth_argument() -> None:
    per_user, _ = calculate_mobility_diagnostics(_events(), _users())
    assert per_user["home_work_distance_km"].isna().all()
    assert per_user["gps_error_m"].isna().all()
    latents = pd.DataFrame([{"user_id": "u1", "home_latitude": 0, "home_longitude": 0,
                             "work_latitude": 0, "work_longitude": 1}])
    trajectories = pd.DataFrame([{"user_id": "u1", "timestamp": "2026-01-01T00:00:00Z",
                                   "true_latitude": 0, "true_longitude": 0}])
    observation = pd.DataFrame([{"user_id": "u1", "record_probability": .75}])
    protected, _ = calculate_mobility_diagnostics(
        _events(), _users(), truth={"latents": latents, "trajectories": trajectories, "observation": observation}
    )
    u1 = protected.set_index("user_id").loc["u1"]
    assert u1.home_work_distance_km == pytest.approx(111.195, rel=1e-4)
    assert u1.gps_error_m == pytest.approx(0.0)
    assert u1.missing_event_rate == pytest.approx(.25)


def test_protected_visualization_module_is_not_used_by_modeling_apis() -> None:
    for name in ("prepare.py", "baseline.py", "training.py", "export.py"):
        source = (Path("src/geoembeddings") / name).read_text(encoding="utf-8")
        assert "visualization_diagnostics" not in source


def test_negative_elapsed_is_integrity_failure_not_behavioral_warning() -> None:
    per_user, transitions = calculate_mobility_diagnostics(_events(), _users())
    transitions.loc[0, "elapsed_hours"] = -1
    report = summarize_diagnostics(per_user, {}, transitions)
    assert report["integrity_failures"][0]["severity"] == "integrity_failure"
