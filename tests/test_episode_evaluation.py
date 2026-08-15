from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from geoembeddings.evaluation import assign_episode_intervals, load_episode_evaluation_inputs
from geoembeddings.representation_schema import EXPORT_SCHEMA_VERSION
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


def test_intent_probe_supports_more_features_than_labeled_rows():
    users = [f"u{i}" for i in range(20)]
    rows = pd.DataFrame({
        "user_id": users,
        "primary_intent": ["routine", "travel"] * 10,
        "embedding_index": np.arange(20),
    })
    embeddings = np.random.default_rng(7).normal(size=(20, 200))

    report = evaluation._intent_probe(rows, embeddings, fraction=0.8, alpha=10.0)

    assert report["status"] == "ok"
    assert np.isfinite(report["accuracy"])


def _episode_config():
    return {
        "seed": 7,
        "evaluation": {
            "probe_train_fraction": 0.8,
            "ridge_alpha": 10.0,
            "episode_response": {"boundary_bin_edges_hours": [-2, 0, 2]},
        },
    }


def _prepared(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "prepared_metadata.json").write_text(json.dumps({
        "source_files": {"observed_events.csv.gz": "events-hash"},
    }))
    return prepared


def test_episode_report_evaluates_named_components_and_preserves_combined_alias(tmp_path):
    dense, truth = _write(tmp_path)
    with np.load(dense, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    persistent = np.asarray([[1.0, 0.0], [0.9, 0.1]])
    context = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    combined = persistent + context
    np.savez(
        dense,
        **{name: value for name, value in arrays.items() if name != "embedding"},
        embedding=combined,
        schema_version=np.asarray(EXPORT_SCHEMA_VERSION),
        model_variant=np.asarray("two_timescale_pc"),
        categorical_fields=np.asarray(["service_id"]),
        continuous_fields=np.asarray(["latitude"]),
        preparation_hash=np.asarray("prepared-hash"),
        source_file_names=np.asarray(["observed_events.csv.gz"]),
        source_hashes=np.asarray(["events-hash"]),
        train_end=np.asarray("2026-01-01T08:00:00Z"),
        validation_end=np.asarray("2026-01-01T08:30:00Z"),
        export_cutoffs=np.asarray(["observed_event"]),
        compatibility=np.asarray("embedding aliases component_combined"),
        component_names=np.asarray(["persistent", "context", "combined"]),
        component_dimensions=np.asarray([2, 2, 2]),
        component_persistent=persistent,
        component_context=context,
        component_combined=combined,
    )

    report = evaluation.evaluate_episode_response(
        truth, _prepared(tmp_path), dense, tmp_path / "episode.json",
        _episode_config(), kind="learned",
    )

    assert report["metric_contract"]["version"] == "episode-response/2.0"
    assert report["metric_contract"]["component_schema"] == {
        "persistent": 2, "context": 2, "combined": 2,
    }
    assert set(report["component_evaluations"]) == {"persistent", "context", "combined"}
    assert report["component_evaluations"]["context"]["applicability"] == "applicable"
    assert report["R4_episode_coherence"] == report["component_evaluations"]["combined"]["R4_episode_coherence"]
    assert report["intent_probe"] == report["component_evaluations"]["combined"]["intent_probe"]


def test_episode_report_marks_legacy_zero_context_not_applicable(tmp_path):
    dense, truth = _write(tmp_path)
    report = evaluation.evaluate_episode_response(
        truth, _prepared(tmp_path), dense, tmp_path / "episode.json",
        _episode_config(), kind="learned",
    )

    assert report["component_evaluations"]["context"]["applicability"] == (
        "not_applicable_structural_zero_adapter"
    )
    assert report["R1_single_vector_diagnostics"]["temporary_episode_drift"] == (
        report["component_evaluations"]["combined"]["R1_temporal_response"]["temporary_episode_drift"]
    )
