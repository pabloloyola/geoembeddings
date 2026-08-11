from __future__ import annotations

import json

import numpy as np
import pytest

from geoembeddings.io import sha256_file
from geoembeddings.reliability import (calibration_bins, coverage_risk_curve,
    evaluate_reliability, resampling_statistics, validate_reliability_report)


def _inputs(tmp_path):
    observed = tmp_path / "run" / "observed"; prepared = tmp_path / "experiment" / "prepared"
    observed.mkdir(parents=True); prepared.mkdir(parents=True)
    for name in ("users_observed.csv.gz", "observed_events.csv.gz"):
        (observed / name).write_bytes(name.encode())
    metadata = {"source_files": {name: sha256_file(observed/name) for name in ("users_observed.csv.gz", "observed_events.csv.gz")},
        "train_end": "2026-01-01T00:00:00Z", "validation_end": "2026-01-02T00:00:00Z",
        "categorical_fields": ["service_id"], "continuous_fields": ["latitude"]}
    (prepared/"prepared_metadata.json").write_text(json.dumps(metadata))
    export = tmp_path/"experiment"/"embeddings.npz"
    users = np.repeat(["u1", "u2", "u3", "u4"], 3); cutoffs = np.tile(["train", "validation", "test"], 4)
    values = np.arange(24, dtype=float).reshape(12, 2) + 1
    np.savez_compressed(export, user_id=users, cutoff=cutoffs, embedding=values)
    return observed, prepared, export


def test_seeded_resampling_is_reproducible_and_seed_sensitive():
    values = np.asarray([[1., 0.], [0., 1.], [1., 1.]])
    first = resampling_statistics(values, seed=7, resamples=50)
    assert first == resampling_statistics(values, seed=7, resamples=50)
    assert first != resampling_statistics(values, seed=8, resamples=50)
    assert first["embedding_variance"] >= 0 and np.isfinite(list(first.values())).all()


def test_calibration_bins_preserve_empty_and_sparse_bins():
    bins = calibration_bins(np.array([.1, .2]), np.array([.3, .4]), bins=4, minimum_count=2)
    assert len(bins) == 4 and sum(x["count"] for x in bins) == 2
    assert any(x["count"] == 0 and x["status"] == "insufficient_coverage" for x in bins)
    assert all(x["status"] == "insufficient_coverage" for x in bins)


def test_coverage_risk_reports_insufficient_counts_and_values():
    curve = coverage_risk_curve(np.array([.3, .1, .2]), np.array([.9, .1, .4]), [.25, 1], minimum_count=2)
    assert curve[0]["status"] == "insufficient_coverage" and curve[0]["risk"] is None
    assert curve[1]["status"] == "ok" and curve[1]["risk"] == pytest.approx((.9+.1+.4)/3)


def test_report_schema_finiteness_and_overwrite(tmp_path):
    observed, prepared, export = _inputs(tmp_path); output = tmp_path/"experiment"/"reliability.json"
    config = {"seed": 1, "evaluation": {"reliability": {"seed": 9, "resamples": 20,
        "calibration_bins": 2, "minimum_bin_count": 2, "coverage_levels": [.5, 1]}}}
    report = evaluate_reliability(observed, prepared, export, output, config, kind="learned")
    validate_reliability_report(report); assert report["seed"] == 9 and report["coverage"]["evaluated_users"] == 4
    with pytest.raises(FileExistsError): evaluate_reliability(observed, prepared, export, output, config, kind="learned")
    report["metrics"]["bad"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"): validate_reliability_report(report)


def test_source_mismatch_is_rejected(tmp_path):
    observed, prepared, export = _inputs(tmp_path); (observed/"observed_events.csv.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source hashes"):
        evaluate_reliability(observed, prepared, export, tmp_path/"out.json", {}, kind="baseline")


def test_preparation_config_mismatch_is_rejected(tmp_path):
    observed, prepared, export = _inputs(tmp_path)
    config = {"data": {"categorical_fields": ["wrong"], "continuous_fields": ["latitude"]}}
    with pytest.raises(ValueError, match="preparation field identity"):
        evaluate_reliability(observed, prepared, export, tmp_path/"out.json", config, kind="baseline")
