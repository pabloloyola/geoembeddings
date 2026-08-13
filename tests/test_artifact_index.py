from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from geoembeddings.artifact_index import (build_artifact_index, inspect_evidence_indexes,
                                          normalize_identifier, stable_values_hash)
from geoembeddings.io import sha256_file


def test_stable_values_hash_is_order_and_duplicate_independent() -> None:
    assert stable_values_hash(["user-b", "user-a", "user-a"]) == stable_values_hash(
        ["user-a", "user-b"]
    )
    assert stable_values_hash(["user-a"]) != stable_values_hash(["user-b"])


def test_normalize_identifier_handles_local_and_external_paths(tmp_path: Path) -> None:
    assert normalize_identifier(tmp_path / "a" / ".." / "result.json", base=tmp_path) == "result.json"
    assert normalize_identifier("HTTPS://Example.COM//archive/run-1/") == "https://example.com/archive/run-1"


def test_index_rejects_mismatched_preparation_source_metadata(tmp_path: Path) -> None:
    run = tmp_path / "run"
    experiment = tmp_path / "experiment"
    observed = run / "observed"
    prepared = experiment / "prepared"
    observed.mkdir(parents=True)
    prepared.mkdir(parents=True)
    for name, content in (("users_observed.csv.gz", b"users"), ("observed_events.csv.gz", b"events")):
        with gzip.GzipFile(filename=str(observed / name), mode="wb", mtime=0) as handle:
            handle.write(content)
    _write_json(run / "manifest.json", {"dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"}})
    metadata = {
        "dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"},
        "source_files": {
            "users_observed.csv.gz": sha256_file(observed / "users_observed.csv.gz"),
            "observed_events.csv.gz": sha256_file(observed / "observed_events.csv.gz"),
        },
        "rows": {"users": 2},
        "train_end": "2026-01-01T00:00:00+00:00",
        "validation_end": "2026-01-02T00:00:00+00:00",
        "categorical_fields": ["service_id", "action_type"],
        "continuous_fields": ["latitude"],
    }
    _write_json(prepared / "prepared_metadata.json", metadata)
    for name in ("statistical_baseline.npz", "embeddings.npz"):
        np.savez_compressed(
            experiment / name,
            user_id=np.asarray(["u1", "u2"]),
            cutoff=np.asarray(["test", "test"]),
            embedding=np.ones((2, 2)),
        )
    for name in ("dense_statistical_baseline.npz", "dense_embeddings.npz"):
        np.savez_compressed(
            experiment / name,
            user_id=np.asarray(["u1", "u2"]),
            timestamp=np.asarray(["2026-01-01", "2026-01-01"]),
            embedding=np.ones((2, 2)),
        )
    _write_json(
        experiment / "baseline_episode_response.json",
        {
            "metric_contract": {
                "prepared_metadata_sha256": "0" * 64,
                "source_hashes": metadata["source_files"],
            }
        },
    )

    with pytest.raises(ValueError, match="baseline episode preparation identity mismatch"):
        build_artifact_index(run, experiment, tmp_path / "index.json", repository_root=tmp_path)


def test_inspection_reports_present_absent_external_and_lost_without_writes(tmp_path: Path) -> None:
    artifacts = tmp_path / "docs" / "artifacts"
    artifacts.mkdir(parents=True)
    local = tmp_path / "result.bin"
    local.write_bytes(b"evidence")
    index_path = artifacts / "current.json"
    _write_json(index_path, {
        "schema_version": "geoembeddings-evidence-index/1.0", "task_id": "T-new",
        "commands": ["uv run geoembed pipeline --run-dir runs/new --experiment-dir experiments/new"],
        "required_artifacts": {"all": [
            {"id": "local", "identifier": "result.bin", "bytes": 8, "sha256": sha256_file(local)},
            {"id": "absent", "identifier": "missing.bin", "bytes": 1, "sha256": "0" * 64},
            {"id": "remote", "identifier": "https://example.test/archive.bin", "bytes": 1, "sha256": "1" * 64},
        ]}})
    lost_path = artifacts / "lost.json"
    _write_json(lost_path, {"task_id": "T-old", "evidence_status": "evidence lost/unverifiable",
                            "required_artifacts": {"all": [{"identifier": "old.bin", "sha256": "2" * 64}]}})
    before = {path: path.read_bytes() for path in (index_path, lost_path)}

    report = inspect_evidence_indexes(artifacts, repository_root=tmp_path)

    assert report["ci_status"] == "ok"
    assert report["summary"] == {"present_local": 1, "locally_absent": 1,
        "intentionally_external": 1, "historically_lost": 1, "total": 4,
        "content_verified": 1, "content_mismatch": 0}
    current = report["indexes"][0]
    assert current["artifacts"][0]["byte_count_matches"] is True
    assert current["artifacts"][0]["sha256_matches"] is True
    assert current["index_alone_sufficient_for_documentation_claims"] is False
    assert current["rerun_commands_for_new_lineage"]
    assert {path: path.read_bytes() for path in before} == before


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
