from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from geoembeddings import simulator
from geoembeddings.pair_integrity import (compare_rows,
                                          require_passing_pair_integrity,
                                          validate_pair)
from geoembeddings.pair_manifest import create_pair_manifest


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _simulate(root: Path, observation_seed: int) -> None:
    config = simulator.load_config(CONFIG)
    config["run"].update(users=10, days=2, seed=71, output=str(root),
                         random_streams={"observation": observation_seed})
    simulator.activate_config(config)
    simulator.simulate(argparse.Namespace(
        output=str(root), overwrite=False, seed=71, users=10, days=2,
        start_date=config["run"]["start_date"], scenario=config["run"]["scenario"],
        full_kanto=False, config=str(CONFIG),
    ))


def test_row_comparison_reports_schema_key_duplicate_and_field_mismatches() -> None:
    schema = ["id", "value"]
    clean = [{"id": "a", "value": "1"}]
    assert compare_rows("truth.x", schema, clean, schema, clean, ("id",), ())["passed"]

    schema_failure = compare_rows("truth.x", schema, clean, ["id", "other"],
                                  [{"id": "a", "other": "1"}], ("id",), ())
    assert not schema_failure["passed"] and not schema_failure["schema_match"]

    missing = compare_rows("truth.x", schema, clean, schema, [], ("id",), ())
    assert not missing["passed"] and missing["missing_keys"]["samples"] == [["a"]]

    duplicate = compare_rows("truth.x", schema, clean * 2, schema, clean, ("id",), ())
    assert not duplicate["passed"] and duplicate["duplicate_keys"]["reference_count"] == 1

    mismatch = compare_rows("truth.x", schema, clean, schema,
                            [{"id": "a", "value": "2"}], ("id",), ())
    assert not mismatch["passed"]
    assert mismatch["field_mismatches"]["samples"][0] == {
        "key": ["a"], "field": "truth.x.value", "reference": "1",
        "intervention": "2", "allowed_by": None,
    }
    allowed = compare_rows("truth.x", schema, clean, schema,
                           [{"id": "a", "value": "2"}], ("id",), ("truth.x.value",))
    assert allowed["passed"] and allowed["allowed_changes"] == {"truth.x.value": 1}


def test_validate_pair_end_to_end_and_evaluation_gate(tmp_path: Path) -> None:
    reference, intervention = tmp_path / "reference", tmp_path / "intervention"
    _simulate(reference, 101)
    _simulate(intervention, 202)
    pair_manifest = tmp_path / "pair" / "pair_manifest.json"
    create_pair_manifest(reference, intervention, pair_manifest)

    # No representation evaluator may start before the prerequisite exists.
    with pytest.raises(FileNotFoundError, match="integrity report"):
        require_passing_pair_integrity(pair_manifest)

    report = validate_pair(pair_manifest)
    assert report["status"] == "passed"
    assert report["table_results"]["truth.user_latents"]["field_mismatches"]["disallowed_count"] == 0
    assert report["table_results"]["observed.events"]["allowed_changes"]
    require_passing_pair_integrity(pair_manifest)

    report_path = pair_manifest.parent / "pair_integrity.json"
    failing = json.loads(report_path.read_text())
    failing["status"] = "failed"
    report_path.write_text(json.dumps(failing))
    with pytest.raises(ValueError, match="passing supported"):
        require_passing_pair_integrity(pair_manifest)

    report_path.write_text(json.dumps(report))
    manifest = json.loads(pair_manifest.read_text())
    manifest["creation_provenance"]["created_at"] += "-stale"
    pair_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale"):
        require_passing_pair_integrity(pair_manifest)


def test_validate_pair_rejects_stale_input_before_passing_report(tmp_path: Path) -> None:
    reference, intervention = tmp_path / "reference", tmp_path / "intervention"
    _simulate(reference, 303)
    _simulate(intervention, 404)
    pair_manifest = tmp_path / "pair" / "pair_manifest.json"
    create_pair_manifest(reference, intervention, pair_manifest)
    with (intervention / "observed" / "users_observed.csv.gz").open("ab") as handle:
        handle.write(b"stale")
    with pytest.raises(ValueError, match="source hash is stale"):
        validate_pair(pair_manifest)
    assert not (pair_manifest.parent / "pair_integrity.json").exists()
