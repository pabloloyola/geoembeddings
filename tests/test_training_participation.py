from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from geoembeddings.config import load_config
from geoembeddings.data import EventWindowDataset, SampleReference, participation_roles
from geoembeddings.io import read_json
from geoembeddings.training import train_model


def _dataset(events: pd.DataFrame, split_users: list[str]) -> EventWindowDataset:
    dataset = EventWindowDataset.__new__(EventWindowDataset)
    dataset.events = events
    dataset.metadata = {
        "train_end": "2025-01-01T12:00:00+00:00",
        "validation_end": "2025-01-02T12:00:00+00:00",
        "timestamp_max": "2025-01-03T12:00:00+00:00",
    }
    dataset.references = [SampleReference(user, (0,), index) for index, user in enumerate(split_users)]
    return dataset


def _hash(users: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(users)).encode()).hexdigest()


def test_participation_uses_actual_target_windows_not_cutoffs_or_export_users() -> None:
    events = pd.DataFrame({
        "user_id": ["train", "validation", "export_only", "probe_only"],
        "timestamp": pd.to_datetime([
            "2025-01-01T10:00Z", "2025-01-01T11:00Z",
            "2025-01-01T09:00Z", "2025-01-03T10:00Z",
        ], utc=True),
    })
    train = _dataset(events, ["train"])
    validation = _dataset(events, ["validation"])

    roles = participation_roles(train, validation)

    assert roles["eligible_training_windows"] == {
        "count": 1, "identity_sha256": _hash(["train"]), "window_count": 1,
    }
    assert roles["validation_checkpoint_selection_windows"]["identity_sha256"] == _hash(["validation"])
    # Both users have cutoff/export availability, but neither contributed an eligible target window.
    assert roles["exported_only_after_checkpoint_freezing"]["identity_sha256"] == _hash(
        ["export_only", "probe_only"]
    )
    # A hypothetical evaluator label does not enter the API and cannot alter membership.
    reversed_roles = participation_roles(_dataset(events.iloc[::-1], ["train"]),
                                         _dataset(events.iloc[::-1], ["validation"]))
    assert reversed_roles == roles


def test_training_writes_versioned_immutable_participation_artifact(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/embedding/single_vector.yaml")
    config["training"].update({"device": "cpu", "epochs": 1, "batch_size": 512})
    config["model"].update({"categorical_embedding_dim": 2, "event_dim": 4,
                            "hidden_dim": 4, "user_embedding_dim": 4})
    output = tmp_path / "model"
    train_model(root / "smoke/run/observed", root / "smoke/experiment/prepared", output, config)

    artifact = read_json(output / "training_participation.json")
    assert artifact["schema_version"] == "geoembeddings-training-participation/1.0"
    assert artifact["participation_definition"]["version"] == "eligible-target-windows/1.0"
    assert artifact["checkpoint_identity"]["sha256"]
    assert artifact["preparation_identity"]["observed_source_hashes"]
    assert artifact["roles"]["eligible_training_windows"]["window_count"] > 0

    with pytest.raises(FileExistsError, match="Immutable training participation"):
        train_model(root / "smoke/run/observed", root / "smoke/experiment/prepared", output, config)
