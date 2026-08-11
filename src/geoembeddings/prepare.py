from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import dump_config
from .io import sha256_file, write_json
from .schema import EVENT_FILE, USER_FILE, load_observed


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def prepare_data(
    observed_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    observed_dir = Path(observed_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    users, events = load_observed(observed_dir)
    source_contract = json.loads((observed_dir.parent / "manifest.json").read_text(encoding="utf-8"))["dataset_contract"]

    train_end, validation_end = _temporal_boundaries(events, config["data"])
    train_events = events.loc[events["timestamp"] <= train_end]

    categorical_fields = list(config["data"]["categorical_fields"])
    if bool(config["data"].get("include_object_id", False)):
        categorical_fields.append("object_id")
    missing_fields = set(categorical_fields) - set(events.columns)
    if missing_fields:
        raise ValueError(f"Configured categorical fields absent from events: {sorted(missing_fields)}")

    vocabularies = {
        field: _make_vocabulary(train_events[field]) for field in categorical_fields
    }
    continuous_stats = _continuous_statistics(train_events)

    per_split = _count_targets_by_split(events, train_end, validation_end)
    report = {
        "dataset_contract": {
            "name": source_contract["name"],
            "version": source_contract["version"],
        },
        "run_dir": str(observed_dir.parent),
        "observed_dir": str(observed_dir),
        "source_files": {
            USER_FILE: sha256_file(observed_dir / USER_FILE),
            EVENT_FILE: sha256_file(observed_dir / EVENT_FILE),
        },
        "rows": {"users": int(len(users)), "events": int(len(events))},
        "users_with_events": int(events["user_id"].nunique()),
        "timestamp_min": events["timestamp"].min().isoformat(),
        "timestamp_max": events["timestamp"].max().isoformat(),
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "target_events_by_split": per_split,
        "categorical_fields": categorical_fields,
        "continuous_fields": list(config["data"]["continuous_fields"]),
        "vocabulary_sizes": {field: len(values) for field, values in vocabularies.items()},
        "continuous_statistics": continuous_stats,
        "information_boundary": (
            "Vocabularies and normalization statistics were fit using observed training events only. "
            "No truth/ file is accepted or read by prepare or train."
        ),
    }

    write_json(vocabularies, output_dir / "vocabularies.json")
    write_json(report, output_dir / "prepared_metadata.json")
    dump_config(config, output_dir / "config.resolved.yaml")
    return report


def _temporal_boundaries(
    events: pd.DataFrame,
    data_config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = np.sort(events["timestamp"].drop_duplicates().to_numpy())
    if len(timestamps) < 3:
        raise ValueError("At least three distinct event timestamps are required")
    train_fraction = float(data_config["train_fraction"])
    validation_fraction = float(data_config["validation_fraction"])
    train_index = min(max(int(len(timestamps) * train_fraction) - 1, 0), len(timestamps) - 3)
    validation_index = min(
        max(int(len(timestamps) * (train_fraction + validation_fraction)) - 1, train_index + 1),
        len(timestamps) - 2,
    )
    train_end = pd.Timestamp(timestamps[train_index])
    validation_end = pd.Timestamp(timestamps[validation_index])
    return train_end, validation_end


def _make_vocabulary(values: pd.Series) -> dict[str, int]:
    normalized = values.fillna(UNK_TOKEN).astype(str)
    unique = sorted(value for value in normalized.unique() if value not in {PAD_TOKEN, UNK_TOKEN, ""})
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    vocabulary.update({value: index + 2 for index, value in enumerate(unique)})
    return vocabulary


def _continuous_statistics(events: pd.DataFrame) -> dict[str, dict[str, float]]:
    derived = derive_continuous_features(events)
    statistics: dict[str, dict[str, float]] = {}
    for column in derived.columns:
        values = derived[column].astype(float)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        statistics[column] = {
            "mean": mean,
            "std": standard_deviation if standard_deviation > 1e-8 else 1.0,
        }
    return statistics


def derive_continuous_features(events: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(events["timestamp"], utc=True)
    delta_minutes = (
        events.assign(_timestamp=timestamp)
        .groupby("user_id", sort=False)["_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_of_week = timestamp.dt.dayofweek.astype(float)
    accuracy = pd.to_numeric(events["location_accuracy_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.DataFrame(
        {
            "latitude": pd.to_numeric(events["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(events["longitude"], errors="coerce"),
            "log_delta_minutes": np.log1p(delta_minutes),
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
            "log_location_accuracy_m": np.log1p(accuracy),
        },
        index=events.index,
    )


def _count_targets_by_split(
    events: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, int]:
    timestamps = events["timestamp"]
    return {
        "train": int((timestamps <= train_end).sum()),
        "validation": int(((timestamps > train_end) & (timestamps <= validation_end)).sum()),
        "test": int((timestamps > validation_end).sum()),
    }
