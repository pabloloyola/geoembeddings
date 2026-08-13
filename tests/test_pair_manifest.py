from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from geoembeddings import simulator
from geoembeddings.contract import PAIR_MANIFEST_SCHEMA, PairManifest, validate_pair_manifest
from geoembeddings.pair_manifest import create_pair_manifest


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _run(root: Path, observation_seed: int) -> None:
    config = simulator.load_config(CONFIG)
    config["run"].update(users=10, days=2, seed=20260811, output=str(root),
                         random_streams={"observation": observation_seed})
    simulator.activate_config(config)
    simulator.simulate(argparse.Namespace(output=str(root), overwrite=False, seed=20260811,
        users=10, days=2, start_date=config["run"]["start_date"],
        scenario=config["run"]["scenario"], full_kanto=False, config=str(CONFIG)))


def _valid() -> dict:
    digest = hashlib.sha256(b"x").hexdigest()
    run = {"run_dir": "/run", "simulator_version": "1", "dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"},
           "manifest_sha256": digest, "config_sha256": digest, "source_hashes": {"observed/x": digest},
           "identity_schema": "geoembeddings-simulation-identity/1.0", "entity_hashes": {name: digest for name in ("users", "regions", "pois", "episodes", "choices", "trajectories")}}
    return {"schema_version": PAIR_MANIFEST_SCHEMA, "reference": run, "intervention": {**run, "run_dir": "/other"},
            "intervention_type": "observation", "intervention_parameters": {},
            "invariant_entity_classes": ["users"], "allowed_to_change_fields": ["observed.*"],
            "matching_keys": {"users": ["user_id"], "regions": ["region_id"], "pois": ["poi_id"],
                "episodes": ["episode_id"], "choices": ["decision_id"], "trajectories": ["trajectory_id"]},
            "stream_lineage": {"reference": {}}, "creation_provenance": {"created_at": "now"}}


def test_pair_manifest_schema_round_trip_and_rejections() -> None:
    value = _valid()
    assert PairManifest.from_dict(value).to_dict() == value
    broken = json.loads(json.dumps(value)); broken["schema_version"] = "geoembeddings-pair-manifest/2.0"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_pair_manifest(broken)
    broken = json.loads(json.dumps(value)); broken["reference"]["manifest_sha256"] = ""
    with pytest.raises(ValueError, match="missing hashes"):
        validate_pair_manifest(broken)
    broken = json.loads(json.dumps(value)); broken["allowed_to_change_fields"] = ["users"]
    with pytest.raises(ValueError, match="overlap"):
        validate_pair_manifest(broken)
    broken = json.loads(json.dumps(value)); broken["matching_keys"]["regions"] = ["user_id"]
    with pytest.raises(ValueError, match="ambiguous"):
        validate_pair_manifest(broken)


def test_pair_manifest_identity_compatible_fixed_seed_runs(tmp_path: Path) -> None:
    reference, intervention = tmp_path / "reference", tmp_path / "intervention"
    _run(reference, 101); _run(intervention, 202)
    output = tmp_path / "pair" / "pair_manifest.json"
    result = create_pair_manifest(reference, intervention, output)
    assert result["intervention_type"] == "observation"
    assert result["intervention_parameters"]["changed_streams"] == ["observation"]
    assert set(result["invariant_entity_classes"]) == {"users", "regions", "pois", "episodes", "choices", "trajectories"}
    assert output.is_file()
    with pytest.raises(FileExistsError):
        create_pair_manifest(reference, intervention, output)
    create_pair_manifest(reference, intervention, output, overwrite=True)


def test_pair_manifest_rejects_incompatible_contract(tmp_path: Path) -> None:
    reference, intervention = tmp_path / "reference", tmp_path / "intervention"
    _run(reference, 101); _run(intervention, 202)
    path = intervention / "manifest.json"
    manifest = json.loads(path.read_text()); manifest["dataset_contract"]["version"] = "1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract"):
        create_pair_manifest(reference, intervention, tmp_path / "pair" / "pair_manifest.json")
