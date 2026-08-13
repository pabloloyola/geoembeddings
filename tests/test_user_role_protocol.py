from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from geoembeddings.config import load_config
from geoembeddings.data import EventWindowDataset
from geoembeddings.export import export_embeddings
from geoembeddings.io import read_json
from geoembeddings.prepare import prepare_data
from geoembeddings.training import train_model
from geoembeddings.user_roles import (PROTOCOL_SCHEMA, assign_users,
                                      authenticate_roles, role_summary)


def _config(root: Path) -> dict:
    config = load_config(root / "configs/embedding/user_role_diagnostic_v2.yaml")
    config["training"].update({"device": "cpu", "epochs": 1, "batch_size": 512})
    config["model"].update({"categorical_embedding_dim": 2, "event_dim": 4,
                            "hidden_dim": 4, "user_embedding_dim": 4})
    return config


def test_seeded_assignment_is_canonical_disjoint_and_drift_rejected() -> None:
    protocol = {"schema_version": PROTOCOL_SCHEMA, "seed": 7, "fractions": {
        "target_train": .5, "target_validation": .25, "target_test": .25}}
    users = [f"u{i}" for i in range(40)]
    forward = assign_users(users, protocol)
    assert forward == assign_users(reversed(users), protocol)
    summary = role_summary(forward)
    assert sum(summary[role]["count"] for role in
               ("target_train", "target_validation", "target_test")) == len(users)
    metadata = {"preparation_schema_version": "geoembeddings-preparation/2.0",
                "user_role_protocol": {"schema_version": PROTOCOL_SCHEMA, "seed": 7,
                 "fractions": protocol["fractions"], "assignment_sha256": "post-hoc",
                 "roles": summary}}
    with pytest.raises(ValueError, match="drifted"):
        authenticate_roles(metadata, {"data": {"user_role_protocol": protocol}}, users)


def test_cross_stage_test_nonmembers_have_no_fitting_window(tmp_path: Path) -> None:
    """Preparation, training and export preserve the same frozen role identity."""
    root = Path(__file__).resolve().parents[1]
    observed = root / "smoke/run/observed"
    experiment = tmp_path / "experiment"
    config = _config(root)
    prepared = experiment / "prepared"
    metadata = prepare_data(observed, prepared, config)
    train = EventWindowDataset(observed, prepared, "train", config)
    validation = EventWindowDataset(observed, prepared, "validation", config)
    assignments = train.user_roles
    assert assignments is not None
    test_users = {user for user, role in assignments.items() if role == "target_test"}
    assert test_users
    assert test_users.isdisjoint(reference.user_id for reference in train.references)
    assert test_users.isdisjoint(reference.user_id for reference in validation.references)
    assert metadata["preprocessing_participants"]["identity_sha256"] == \
        metadata["user_role_protocol"]["roles"]["target_train"]["identity_sha256"]

    report = train_model(observed, prepared, experiment / "model", config)
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["preparation_identity"]["user_role_protocol"] == metadata["user_role_protocol"]
    output = experiment / "embeddings.npz"
    exported = export_embeddings(observed, prepared, report["checkpoint"], output, config)
    arrays = np.load(output, allow_pickle=False)
    exported_users = set(arrays["user_id"].tolist())
    assert exported_users and exported_users <= test_users  # users without cutoff history may be omitted
    assert exported["user_role_protocol"] == metadata["user_role_protocol"]
    participation = read_json(experiment / "model/training_participation.json")
    assert participation["user_role_protocol"] == metadata["user_role_protocol"]

    changed = json.loads(json.dumps(config))
    changed["data"]["user_role_protocol"]["seed"] += 1
    with pytest.raises(ValueError, match="drifted"):
        EventWindowDataset(observed, prepared, "train", changed)
