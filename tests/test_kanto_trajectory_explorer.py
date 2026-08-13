from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.kanto_trajectory_explorer import (
    EmptySelectionError,
    ExplorerData,
    filter_events,
    render_trajectories,
    split_trajectories,
)


def _data() -> ExplorerData:
    users = pd.DataFrame(
        [
            {"user_id": "u1", "age_group": "20-29", "household_type": "single"},
            {"user_id": "u2", "age_group": "40-49", "household_type": "family"},
        ]
    )
    events = pd.DataFrame(
        [
            {"user_id": "u2", "timestamp": "2026-01-02T01:00:00Z", "service_id": "travel", "action_type": "search", "observation_mode": "active", "region_id": "hakone", "latitude": 35.2, "longitude": 139.0},
            {"user_id": "u1", "timestamp": "2026-01-01T15:30:00Z", "service_id": "location", "action_type": "ping", "observation_mode": "passive", "region_id": "tokyo", "latitude": 35.1, "longitude": 139.1},
            {"user_id": "u1", "timestamp": "2026-01-01T14:30:00Z", "service_id": "location", "action_type": "ping", "observation_mode": "passive", "region_id": "tokyo", "latitude": 35.0, "longitude": 139.0},
            {"user_id": "u1", "timestamp": "2026-01-03T00:00:00Z", "service_id": "travel", "action_type": "book", "observation_mode": "active", "region_id": "hakone", "latitude": 35.3, "longitude": 139.2},
        ]
    )
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
    return ExplorerData(users, events)


def test_filter_composition_and_requested_value_validation() -> None:
    selected = filter_events(
        _data(),
        {"age_group": ["20-29"], "household_type": ["single"], "service": ["location"], "region": ["tokyo"]},
    )
    assert selected["user_id"].tolist() == ["u1", "u1"]
    assert selected["timestamp"].is_monotonic_increasing
    with pytest.raises(ValueError, match="Invalid service"):
        filter_events(_data(), {"service": ["not-from-this-run"]})


def test_empty_selection_is_explicit_and_not_rendered(tmp_path: Path) -> None:
    selected = filter_events(_data(), {"age_group": ["20-29"], "service": ["travel"], "region": ["tokyo"]})
    assert selected.empty
    output = tmp_path / "empty.png"
    with pytest.raises(EmptySelectionError, match="No observed events"):
        render_trajectories(selected, output)
    assert not output.exists()


def test_stable_chronological_order_and_gap_splitting() -> None:
    selected = filter_events(_data())
    assert selected[["user_id", "timestamp"]].values.tolist() == sorted(
        selected[["user_id", "timestamp"]].values.tolist(), key=lambda row: (row[0], row[1])
    )
    parts = split_trajectories(selected, "6h")
    assert [part["user_id"].nunique() for part in parts] == [1, 1, 1]
    assert [len(part) for part in parts] == [2, 1, 1]


def test_date_filter_uses_tokyo_calendar_date() -> None:
    # 15:30 UTC is 00:30 on January 2 in Tokyo; 14:30 UTC remains January 1.
    selected = filter_events(_data(), start_date="2026-01-02", end_date="2026-01-02")
    assert selected["timestamp"].dt.date.astype(str).unique().tolist() == ["2026-01-02"]
    assert len(selected) == 2


def test_maximum_number_of_trajectories_is_enforced(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    output = tmp_path / "capped.png"
    figure, metadata = render_trajectories(_data().events, output, max_gap="6h", max_trajectories=2)
    assert metadata == {"rendered_trajectories": 2, "available_trajectories": 3}
    assert output.is_file()
    import matplotlib.pyplot as plt

    plt.close(figure)
