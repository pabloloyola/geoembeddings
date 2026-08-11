from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contract import OBSERVED_FILES


USER_FILE = OBSERVED_FILES["users"]
EVENT_FILE = OBSERVED_FILES["events"]

REQUIRED_USER_COLUMNS = {
    "user_id",
    "age_group",
    "household_type",
    "home_prefecture",
    "home_region_id",
    "geo_split",
}

REQUIRED_EVENT_COLUMNS = {
    "user_id",
    "timestamp",
    "service_id",
    "action_type",
    "observation_mode",
    "object_id",
    "object_category",
    "region_id",
    "prefecture",
    "latitude",
    "longitude",
    "geohash_5",
    "geohash_7",
    "location_accuracy_m",
    "session_id",
}

FORBIDDEN_TRAINING_COLUMN_PARTS = {
    "latent",
    "utility",
    "episode_id",
    "decision_id",
    "is_chosen",
    "true_latitude",
    "true_longitude",
    "price_sensitivity",
    "distance_sensitivity",
    "novelty_seeking",
    "family_orientation",
    "travel_propensity",
}


def load_observed(observed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_dir = Path(observed_dir)
    if observed_dir.name != "observed":
        raise ValueError(
            "--observed-dir must point directly to the simulator's observed/ directory; "
            "training code does not accept the dataset root or truth/"
        )
    users_path = observed_dir / USER_FILE
    events_path = observed_dir / EVENT_FILE
    missing_files = [str(path) for path in (users_path, events_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing observed input files: {missing_files}")

    users = pd.read_csv(users_path, dtype=str)
    events = pd.read_csv(events_path, low_memory=False)
    validate_observed(users, events)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="raise")
    events = events.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    return users, events


def validate_observed(users: pd.DataFrame, events: pd.DataFrame) -> None:
    missing_users = REQUIRED_USER_COLUMNS - set(users.columns)
    missing_events = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing_users:
        raise ValueError(f"users_observed is missing columns: {sorted(missing_users)}")
    if missing_events:
        raise ValueError(f"observed_events is missing columns: {sorted(missing_events)}")

    suspicious = []
    for column in list(users.columns) + list(events.columns):
        lower = column.lower()
        if any(part in lower for part in FORBIDDEN_TRAINING_COLUMN_PARTS):
            suspicious.append(column)
    if suspicious:
        raise ValueError(f"Protected truth-like columns found in training inputs: {sorted(set(suspicious))}")
    if users["user_id"].duplicated().any():
        raise ValueError("users_observed must contain one row per user")
    unknown_users = set(events["user_id"].astype(str)) - set(users["user_id"].astype(str))
    if unknown_users:
        example = sorted(unknown_users)[:5]
        raise ValueError(f"Events refer to users absent from users_observed: {example}")
    if events.empty:
        raise ValueError("observed_events contains no rows")
