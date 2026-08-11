from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
import pytest

from geoembeddings import simulator
from geoembeddings.simulation_validation import validate
from geoembeddings.contract import validate_identity_manifest
from geoembeddings.layout import DatasetLayout


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _run(root: Path, *, seed: int = 20260811, observation_seed: int | None = None) -> dict:
    config = simulator.load_config(CONFIG)
    config["run"].update(users=50, days=7, seed=seed, output=str(root))
    if observation_seed is not None:
        config["run"]["random_streams"] = {"observation": observation_seed}
    simulator.activate_config(config)
    args = argparse.Namespace(
        output=str(root), overwrite=False, seed=seed, users=50, days=7,
        start_date=config["run"]["start_date"], scenario=config["run"]["scenario"],
        full_kanto=False, config=str(CONFIG),
    )
    return simulator.simulate(args)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_named_stream_derivation_is_repeatable_and_independent():
    first = simulator.make_random_streams(17)
    second = simulator.make_random_streams(17)
    assert first.seeds == second.seeds
    assert len(set(first.seeds.values())) == len(simulator.RANDOM_STREAM_NAMES)
    assert [first.generators[name].random() for name in simulator.RANDOM_STREAM_NAMES] == [
        second.generators[name].random() for name in simulator.RANDOM_STREAM_NAMES
    ]

    changed = simulator.make_random_streams(17, {"observation": 99})
    assert changed.seeds["observation"] == 99
    assert {name: changed.seeds[name] for name in simulator.RANDOM_STREAM_NAMES if name != "observation"} == {
        name: first.seeds[name] for name in simulator.RANDOM_STREAM_NAMES if name != "observation"
    }


def test_stable_identity_hash_is_order_independent_and_rejects_duplicates():
    ids = [simulator.stable_identifier("episode", "user-a", "2026-08-11"), simulator.stable_identifier("episode", "user-b", "2026-08-11")]
    assert simulator.identity_set_hash(ids) == simulator.identity_set_hash(list(reversed(ids)))
    assert ids[0] == simulator.stable_identifier("episode", "user-a", "2026-08-11")
    with pytest.raises(ValueError, match="unique"):
        simulator.identity_set_hash([ids[0], ids[0]])


def test_identity_manifest_schema_and_version_rejection():
    streams = simulator.make_random_streams(17)
    section = {
        "schema_version": "geoembeddings-simulation-identity/1.0",
        "identity_generation_version": simulator.IDENTITY_GENERATION_VERSION,
        "hash_algorithm": "sha256-canonical-sorted-identifiers/1.0",
        "random_streams": {"algorithm": simulator.RANDOM_STREAM_ALGORITHM, "root_seed": 17, "seeds": streams.seeds},
        "entities": {name: {"count": 0, "identity_sha256": hashlib.sha256(b"").hexdigest()} for name in ("users", "regions", "pois", "episodes", "choices", "trajectories")},
    }
    validate_identity_manifest(json.loads(json.dumps(section)), stream_names=simulator.RANDOM_STREAM_NAMES)
    section["schema_version"] = "geoembeddings-simulation-identity/2.0"
    with pytest.raises(ValueError, match="Unsupported simulation identity schema"):
        validate_identity_manifest(section, stream_names=simulator.RANDOM_STREAM_NAMES)


def test_fixed_seed_simulation_provenance_identity_and_observation_independence(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    changed = tmp_path / "changed-observation"
    manifest = _run(reference)
    _run(repeat)
    _run(changed, observation_seed=99)

    assert validate(reference)["status"] == "passed"
    assert manifest["random_streams"]["root_seed"] == 20260811
    assert set(manifest["random_streams"]["seeds"]) == set(simulator.RANDOM_STREAM_NAMES)
    assert manifest["identity"]["random_streams"] == manifest["random_streams"]
    assert manifest["identity"]["entities"]["users"]["count"] == 50
    DatasetLayout.from_path(reference).validate(require_truth=True)
    resolved = yaml.safe_load((reference / "config.resolved.yaml").read_text())
    assert resolved["run"]["random_streams"] == manifest["random_streams"]["seeds"]

    stable_truth = (
        "user_latents.csv.gz", "episodes_truth.csv.gz", "candidate_sets.csv.gz",
        "choices_truth.csv.gz", "trajectories_truth.csv.gz",
    )
    for filename in stable_truth:
        assert _digest(reference / "truth" / filename) == _digest(repeat / "truth" / filename)
        assert _digest(reference / "truth" / filename) == _digest(changed / "truth" / filename)

    assert _digest(reference / "observed" / "observed_events.csv.gz") == _digest(
        repeat / "observed" / "observed_events.csv.gz"
    )
    assert _digest(reference / "truth" / "observation_process.csv.gz") != _digest(
        changed / "truth" / "observation_process.csv.gz"
    )
    observed_names = {path.name for path in (reference / "observed").iterdir()}
    assert observed_names == {"users_observed.csv.gz", "observed_events.csv.gz"}
    assert "random_streams" not in json.loads((reference / "manifest.json").read_text())["validation"]

    reference_identity = manifest["identity"]["entities"]
    changed_identity = json.loads((changed / "manifest.json").read_text())["identity"]["entities"]
    for entity in ("users", "regions", "pois", "episodes", "choices", "trajectories"):
        assert reference_identity[entity] == changed_identity[entity]


def test_validation_rejects_duplicate_and_inconsistent_identity(tmp_path):
    root = tmp_path / "run"
    _run(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["entities"]["users"]["identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = validate(root)
    assert report["status"] == "failed"
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "Identity manifest consistency" in failed
