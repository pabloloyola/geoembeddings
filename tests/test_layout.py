from __future__ import annotations

import json

import pytest

from geoembeddings.contract import DATASET_CONTRACT_NAME, DATASET_CONTRACT_VERSION, OBSERVED_FILES
from geoembeddings.layout import DatasetLayout, ExperimentLayout, PairLayout


def test_layout_resolves_all_paths(tmp_path) -> None:
    run = DatasetLayout.from_path(tmp_path / "run")
    experiment = ExperimentLayout.from_path(tmp_path / "experiment")
    assert run.observed == run.root / "observed"
    assert run.truth == run.root / "truth"
    assert experiment.prepared == experiment.root / "prepared"
    assert experiment.checkpoint == experiment.root / "model" / "best_model.pt"
    assert experiment.dense_embeddings == experiment.root / "dense_embeddings.npz"
    assert experiment.episode_response == experiment.root / "episode_response.json"
    assert experiment.baseline_episode_response == experiment.root / "baseline_episode_response.json"
    assert experiment.temporal_routine_evaluation("baseline") == experiment.root / "baseline_temporal_routine.json"
    assert experiment.temporal_routine_evaluation("learned") == experiment.root / "learned_temporal_routine.json"
    assert experiment.reliability_evaluation("baseline") == experiment.root / "baseline_reliability.json"
    assert experiment.reliability_evaluation("learned") == experiment.root / "reliability.json"
    assert experiment.offline_benchmark == experiment.root / "benchmarks" / "offline.json"
    pair = PairLayout.from_path(tmp_path / "pair")
    assert pair.manifest == pair.root / "pair_manifest.json"
    assert pair.integrity_report == pair.root / "pair_integrity.json"


def test_layout_validates_contract(tmp_path) -> None:
    run = DatasetLayout.from_path(tmp_path / "run")
    run.observed.mkdir(parents=True)
    for name in OBSERVED_FILES.values():
        (run.observed / name).write_bytes(b"placeholder")
    run.manifest_path.write_text(
        json.dumps(
            {
                "dataset_contract": {
                    "name": DATASET_CONTRACT_NAME,
                    "version": DATASET_CONTRACT_VERSION,
                }
            }
        ),
        encoding="utf-8",
    )
    run.validate()


def test_layout_explicitly_supports_legacy_event_contract(tmp_path) -> None:
    run = DatasetLayout.from_path(tmp_path / "legacy")
    run.observed.mkdir(parents=True)
    for name in ("users_observed.csv.gz", "observed_events.csv.gz"):
        (run.observed / name).write_bytes(b"legacy")
    run.manifest_path.write_text(json.dumps({"dataset_contract": {"name": DATASET_CONTRACT_NAME, "version": "1.0"}}))
    assert run.validate()["dataset_contract"]["version"] == "1.0"


def test_dataset_root_rejects_internal_directory(tmp_path) -> None:
    with pytest.raises(ValueError):
        DatasetLayout.from_path(tmp_path / "observed")
