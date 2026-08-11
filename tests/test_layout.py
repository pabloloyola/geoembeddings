from __future__ import annotations

import json

import pytest

from geoembeddings.contract import DATASET_CONTRACT_NAME, DATASET_CONTRACT_VERSION
from geoembeddings.layout import DatasetLayout, ExperimentLayout


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


def test_layout_validates_contract(tmp_path) -> None:
    run = DatasetLayout.from_path(tmp_path / "run")
    run.observed.mkdir(parents=True)
    for name in ("users_observed.csv.gz", "observed_events.csv.gz"):
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


def test_dataset_root_rejects_internal_directory(tmp_path) -> None:
    with pytest.raises(ValueError):
        DatasetLayout.from_path(tmp_path / "observed")
