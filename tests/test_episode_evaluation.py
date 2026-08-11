from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoembeddings.evaluation import assign_episode_intervals, load_episode_evaluation_inputs
import inspect
from geoembeddings import baseline, export, evaluation


def _write(tmp_path: Path, *, embedding=None, timestamps=None, episodes=None):
    truth = tmp_path / "run" / "truth"; truth.mkdir(parents=True)
    timestamps = timestamps or ["2026-01-01T09:00:00Z", "2026-01-01T10:00:00Z"]
    embedding = np.asarray(embedding if embedding is not None else [[1, 0], [0, 1]], dtype=float)
    dense = tmp_path / "dense.npz"
    np.savez(dense, user_id=np.asarray(["u1"] * len(timestamps)), timestamp=np.asarray(timestamps),
             cutoff_kind=np.asarray(["observed_event"] * len(timestamps)), embedding=embedding,
             history_event_count=np.arange(1, len(timestamps) + 1))
    pd.DataFrame(episodes or [{"user_id":"u1", "episode_id":"e1", "start_time":"2026-01-01T09:00:00Z",
        "end_time":"2026-01-01T10:00:00Z", "primary_intent":"work"}]).to_csv(truth / "episodes_truth.csv.gz", index=False)
    return dense, truth


def test_half_open_interval_assigns_start_but_not_end(tmp_path):
    dense_path, truth = _write(tmp_path)
    dense, _, episodes = load_episode_evaluation_inputs(dense_path, truth)
    assigned = assign_episode_intervals(dense, episodes)
    assert assigned["episode_id"].tolist() == ["e1", None]


@pytest.mark.parametrize("episodes,match", [
    ([{"user_id":"u1","episode_id":"e1","start_time":"2026-01-01T10:00:00Z","end_time":"2026-01-01T09:00:00Z","primary_intent":"x"}], "end_time"),
    ([{"user_id":"u1","episode_id":"e1","start_time":"2026-01-01T08:00:00Z","end_time":"2026-01-01T10:00:00Z","primary_intent":"x"},
      {"user_id":"u1","episode_id":"e2","start_time":"2026-01-01T09:00:00Z","end_time":"2026-01-01T11:00:00Z","primary_intent":"y"}], "overlap"),
])
def test_rejects_malformed_or_overlapping_intervals(tmp_path, episodes, match):
    dense, truth = _write(tmp_path, episodes=episodes)
    with pytest.raises(ValueError, match=match): load_episode_evaluation_inputs(dense, truth)


def test_sparse_export_and_missing_truth_users_are_allowed(tmp_path):
    dense, truth = _write(tmp_path, timestamps=["2026-01-01T09:30:00Z"], embedding=[[1,2]])
    frame, matrix, episodes = load_episode_evaluation_inputs(dense, truth)
    assert len(frame) == matrix.shape[0] == 1 and len(episodes) == 1


@pytest.mark.parametrize("embedding,match", [([[1, np.nan], [0, 1]], "non-finite"), ([1, 2], "2-D")])
def test_rejects_invalid_embedding_values_or_dimensions(tmp_path, embedding, match):
    dense, truth = _write(tmp_path, embedding=embedding)
    with pytest.raises(ValueError, match=match): load_episode_evaluation_inputs(dense, truth)


def test_rejects_duplicate_or_nonmonotonic_dense_rows(tmp_path):
    dense, truth = _write(tmp_path, timestamps=["2026-01-01T10:00:00Z", "2026-01-01T09:00:00Z"])
    with pytest.raises(ValueError, match="monotonic"): load_episode_evaluation_inputs(dense, truth)
    dense, truth = _write(tmp_path / "duplicate", timestamps=["2026-01-01T09:00:00Z"] * 2)
    with pytest.raises(ValueError, match="duplicate"): load_episode_evaluation_inputs(dense, truth)


def test_requires_direct_truth_path(tmp_path):
    dense, truth = _write(tmp_path)
    with pytest.raises(ValueError, match="directly"): load_episode_evaluation_inputs(dense, truth.parent)


def test_only_evaluator_names_protected_episode_file():
    assert "episodes_truth.csv.gz" in inspect.getsource(evaluation)
    assert "episodes_truth.csv.gz" not in inspect.getsource(export)
    assert "episodes_truth.csv.gz" not in inspect.getsource(baseline)
