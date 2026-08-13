import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from geoembeddings.calibration import (
    SCHEMA,
    bootstrap_user,
    calibrate_reliability,
    fit_affine,
    frozen_user_split,
)
from geoembeddings.io import sha256_file


def test_window_bootstrap_is_deterministic_and_sparse_safe():
    values = np.arange(20, dtype=float).reshape(5, 4)
    assert bootstrap_user(
        values, seed=7, replicates=20, minimum_windows=2
    ) == bootstrap_user(values, seed=7, replicates=20, minimum_windows=2)
    assert bootstrap_user(values[:2], seed=7, replicates=20, minimum_windows=2) is None


def test_affine_calibration_is_nonnegative():
    fitted = fit_affine(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert fitted["slope"] == pytest.approx(2.0)
    assert fitted["intercept"] >= 0


def _fixture(tmp_path: Path):
    run = tmp_path / "run"
    observed = run / "observed"
    observed.mkdir(parents=True)
    (observed / "events.csv").write_text("public\n1\n")
    manifest = {"dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    # Legacy validation expects these canonical observed filenames.
    from geoembeddings.contract import LEGACY_OBSERVED_FILES

    for filename in LEGACY_OBSERVED_FILES.values():
        (observed / filename).write_text("x\n")
    source = next(iter(LEGACY_OBSERVED_FILES.values()))
    roots = {}
    users = np.array([f"u{i}" for i in range(12) for _ in range(4)])
    timestamps = np.array([f"2026-01-0{j + 1}" for _ in range(12) for j in range(4)])
    for name in ("statistical_baseline", "capacity_matched_single", "factorized_pc"):
        root = tmp_path / name
        prepared = root / "prepared"
        prepared.mkdir(parents=True)
        metadata = {
            "source_files": {source: sha256_file(observed / source)},
            "categorical_fields": [],
            "continuous_fields": [],
            "train_end": "2026-01-02",
            "validation_end": "2026-01-03",
        }
        path = prepared / "prepared_metadata.json"
        path.write_text(json.dumps(metadata))
        dense = root / (
            "dense_statistical_baseline.npz"
            if name == "statistical_baseline"
            else "dense_embeddings.npz"
        )
        values = np.arange(len(users) * 3, dtype=float).reshape(len(users), 3) / 100
        np.savez(
            dense,
            user_id=users,
            timestamp=timestamps,
            embedding=values,
            preparation_hash=np.asarray(sha256_file(path)),
            source_file_names=np.asarray([source]),
            source_hashes=np.asarray([metadata["source_files"][source]]),
        )
        roots[name] = root
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": SCHEMA,
                "split": {"seed": 9, "calibration_fraction": 0.5},
                "bootstrap": {"seed": 10, "replicates": 20, "minimum_windows": 2},
                "reporting": {
                    "bins": 3,
                    "minimum_bin_count": 1,
                    "coverage_levels": [0.5, 1.0],
                },
            }
        )
    )
    return run, roots, config


def test_cross_stage_calibration_authenticates_disjoint_shared_users(tmp_path):
    run, roots, config = _fixture(tmp_path)
    report = calibrate_reliability(run, roots, config, tmp_path / "audit")
    assert report["split"]["overlap_count"] == 0
    assert report["selection"]["aggregate_winner"] is None
    assert (
        report["selection"]["selected_candidate_conclusion"]["status"] == "unavailable"
    )
    assert all(
        value["role"] == "diagnostic_control" for value in report["controls"].values()
    )
    assert report["controls"]["factorized_pc"]["raw"]["coverage_risk"]
    with pytest.raises(FileExistsError):
        calibrate_reliability(run, roots, config, tmp_path / "audit")
    changed = yaml.safe_load(config.read_text())
    changed["split"]["seed"] += 1
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="immutable calibration identity drift"):
        calibrate_reliability(run, roots, config, tmp_path / "audit", overwrite=True)


def test_cross_stage_rejects_source_and_user_drift(tmp_path):
    run, roots, config = _fixture(tmp_path)
    metadata = json.loads(
        (
            roots["statistical_baseline"] / "prepared" / "prepared_metadata.json"
        ).read_text()
    )
    source = run / "observed" / next(iter(metadata["source_files"]))
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="source authentication"):
        calibrate_reliability(run, roots, config, tmp_path / "audit")


def test_frozen_split_has_no_overlap():
    calibration, test = frozen_user_split(
        [f"u{i}" for i in range(20)], seed=3, calibration_fraction=0.5
    )
    assert set(calibration).isdisjoint(test)
