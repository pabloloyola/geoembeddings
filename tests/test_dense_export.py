from __future__ import annotations

import pytest
import pandas as pd

from geoembeddings.data import DenseUserCutoffDataset, _dense_cutoff_offsets
from geoembeddings.prepare import prepare_data


@pytest.mark.parametrize(
    ("event_count", "event_stride", "expected"),
    [
        (0, 1, []),
        (1, 1, [0]),
        (5, 1, [0, 1, 2, 3, 4]),
        (5, 2, [0, 2, 4]),
        (6, 2, [0, 2, 4, 5]),
        (3, 10, [0, 2]),
    ],
)
def test_dense_cutoffs_include_first_and_last_observed_events(
    event_count: int, event_stride: int, expected: list[int]
) -> None:
    assert _dense_cutoff_offsets(event_count, event_stride) == expected


def test_dense_cutoffs_reject_invalid_stride() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _dense_cutoff_offsets(3, 0)


def test_dense_dataset_uses_only_observed_prepared_contract(tmp_path) -> None:
    observed = tmp_path / "run" / "observed"
    observed.mkdir(parents=True)
    (observed.parent / "manifest.json").write_text(
        '{"dataset_contract":{"name":"geoembeddings-dataset","version":"1.0"}}'
    )
    pd.DataFrame(
        [
            {
                "user_id": "user_1",
                "age_group": "adult",
                "household_type": "single",
                "home_prefecture": "Tokyo",
                "home_region_id": "tokyo",
                "geo_split": "train",
            }
        ]
    ).to_csv(observed / "users_observed.csv.gz", index=False)
    events = []
    for index in range(5):
        events.append(
            {
                "user_id": "user_1",
                "timestamp": f"2026-01-0{index + 1}T00:00:00Z",
                "service_id": "location",
                "action_type": "ping",
                "observation_mode": "passive",
                "object_id": f"object_{index}",
                "object_category": "place",
                "region_id": "tokyo",
                "prefecture": "Tokyo",
                "latitude": 35.0 + index / 100,
                "longitude": 139.0 + index / 100,
                "geohash_5": "xn76g",
                "geohash_7": f"xn76g{index:02d}",
                "location_accuracy_m": 10.0,
                "session_id": f"session_{index}",
            }
        )
    pd.DataFrame(events).to_csv(observed / "observed_events.csv.gz", index=False)
    config = {
        "data": {
            "max_sequence_length": 2,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "include_object_id": False,
            "categorical_fields": ["service_id", "action_type"],
            "continuous_fields": ["latitude", "longitude"],
        }
    }
    prepared = tmp_path / "experiment" / "prepared"
    prepare_data(observed, prepared, config)

    dataset = DenseUserCutoffDataset(observed, prepared, config, event_stride=2)

    assert len(dataset) == 3
    assert [dataset[index]["history_event_count"] for index in range(3)] == [1, 3, 5]
    assert [dataset[index]["timestamp"] for index in range(3)] == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00",
        "2026-01-05T00:00:00+00:00",
    ]
    assert dataset[2]["categorical"].shape == (2, 2)
    assert not (tmp_path / "run" / "truth").exists()
