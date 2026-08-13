from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {"data", "model", "objectives", "training", "evaluation"}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    _validate_config(config)
    return config


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    """Load a versioned auxiliary YAML without imposing embedding sections."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("schema_version"), str):
        raise ValueError(f"Auxiliary configuration must be a versioned mapping: {path}")
    return config


def _validate_config(config: dict[str, Any]) -> None:
    from .user_roles import protocol_config
    protocol_config(config)
    data = config["data"]
    train_fraction = float(data["train_fraction"])
    validation_fraction = float(data["validation_fraction"])
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("data.train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if int(data["min_history_events"]) < 1:
        raise ValueError("data.min_history_events must be positive")
    if int(data["max_sequence_length"]) < int(data["min_history_events"]):
        raise ValueError("max_sequence_length must be >= min_history_events")
    if not config["objectives"]:
        raise ValueError("At least one objective weight is required")
    if any(float(value) < 0 for value in config["objectives"].values()):
        raise ValueError("Objective weights must be non-negative")


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
