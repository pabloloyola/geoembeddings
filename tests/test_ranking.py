from __future__ import annotations

import argparse
import builtins
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from geoembeddings import simulator
from geoembeddings.cli import main
from geoembeddings.io import sha256_file
from geoembeddings.ranking import (RankingCandidate, RankingPrediction, RankingRequest,
    FrozenEmbeddingRows, RankingTrainingBatch, compute_ranking_metrics, rank_candidates,
    select_causal_embeddings, train_frozen_head, _candidate_matrix)
from geoembeddings.representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION


def request(request_id: str = "r", user_id: str = "u") -> RankingRequest:
    return RankingRequest(request_id, user_id, datetime.fromisoformat("2026-01-02T10:00:00"), "hakone")


def candidate(poi_id: str, *, available: bool = True, category: str = "cafe", travel: float = 5) -> RankingCandidate:
    return RankingCandidate("r", poi_id, category, travel, available)


def test_availability_ties_empty_candidates_and_nearest() -> None:
    rows = [candidate("b", travel=2), candidate("a", travel=2), candidate("hidden", available=False, travel=1)]
    ranked = rank_candidates("nearest", [request()], rows)
    assert [(row.poi_id, row.rank) for row in ranked] == [("a", 1), ("b", 2)]
    assert rank_candidates("popularity", [request("empty")], rows) == []


def test_temporal_leakage_missing_interactions_and_unknown_categories() -> None:
    interactions = [
        {"poi_id": "a", "interaction_timestamp": "2026-01-02T09:59:59"},
        {"poi_id": "b", "interaction_timestamp": "2026-01-02T10:00:00"},
        {"poi_id": "b", "interaction_timestamp": "2026-01-03T10:00:00"},
    ]
    ranked = rank_candidates("popularity", [request()], [candidate("a"), candidate("b")], interactions=interactions)
    assert [row.poi_id for row in ranked] == ["a", "b"]
    assert [row.poi_id for row in rank_candidates("popularity", [request()], [candidate("b"), candidate("a")])] == ["a", "b"]
    events = [{"user_id": "u", "timestamp": "2026-01-01T10:00:00", "object_category": "cafe"},
              {"user_id": "u", "timestamp": "2026-01-03T10:00:00", "object_category": "unknown"}]
    ranked = rank_candidates("category_preference", [request()],
        [candidate("x", category="unknown"), candidate("y", category="cafe")], history_events=events)
    assert [row.poi_id for row in ranked] == ["y", "x"]


def test_metric_edge_cases() -> None:
    empty = compute_ranking_metrics(["r"], [], {}, [1, 2])
    assert empty.evaluated_requests == 0 and empty.mrr == 0
    predictions = [RankingPrediction("r", "x", 1, 1), RankingPrediction("r", "p", 2, 0)]
    metrics = compute_ranking_metrics(["r"], predictions, {"r": {"p"}}, [1, 2])
    assert metrics.recall_at_k == {"1": 0.0, "2": 1.0}
    assert metrics.mrr == .5
    with pytest.raises(ValueError, match="positive"):
        compute_ranking_metrics([], [], {}, [0])


def test_frozen_embedding_causal_selection_rejections_and_empty_history() -> None:
    times = np.asarray([datetime.fromisoformat("2026-01-01T00:00:00"),
                        datetime.fromisoformat("2026-01-02T10:00:00"),
                        datetime.fromisoformat("2026-01-03T00:00:00")], dtype=object)
    rows = FrozenEmbeddingRows(np.asarray(["u", "u", "u"]), np.asarray(["event"] * 3), times,
                               np.asarray([[1., 2.], [5., 6.], [9., 9.]]), "combined")
    requests = [RankingRequest("before", "u", datetime.fromisoformat("2025-12-31T23:00:00"), "hakone"),
                request("exact"),
                RankingRequest("after", "u", datetime.fromisoformat("2026-01-02T12:00:00"), "hakone"),
                request("missing", "other")]
    selected = select_causal_embeddings(rows, requests)
    assert "before" not in selected
    assert np.array_equal(selected["exact"], [5., 6.])
    assert np.array_equal(selected["after"], [5., 6.])  # the future [9, 9] vector is never selected
    assert "missing" not in selected  # empty history is explicit, never substituted
    future = FrozenEmbeddingRows(np.asarray(["u"]), np.asarray(["test"]), times[2:],
                                 np.asarray([[9., 9.]]), "combined")
    assert select_causal_embeddings(future, [request()]) == {}
    with pytest.raises(ValueError, match="non-finite"):
        select_causal_embeddings(FrozenEmbeddingRows(np.asarray(["u"]), np.asarray(["train"]), times[:1],
            np.asarray([[np.nan, 1.]]), "combined"), [request()])
    with pytest.raises(ValueError, match="duplicate"):
        select_causal_embeddings(FrozenEmbeddingRows(np.asarray(["u", "u"]), np.asarray(["train", "train"]),
            np.asarray([times[0], times[0]], dtype=object), np.ones((2, 2)), "combined"), [request()])
    with pytest.raises(ValueError, match="dimensionally"):
        select_causal_embeddings(FrozenEmbeddingRows(np.asarray(["u"]), np.asarray(["train"]), times,
            np.ones((2, 2)), "combined"), [request()])


def test_candidate_preprocessing_uses_only_scorable_training_candidates() -> None:
    requests = [request("ordinary-1"), request("ordinary-2"), request("unscorable")]
    request_rows = {rid: {"latitude": 1, "longitude": 2, "context_source": "arrival"}
                    for rid in ("ordinary-1", "ordinary-2", "unscorable")}
    catalog = {
        "p1": {"latitude": 1, "longitude": 2, "price_level": 1, "family_suitability": 1,
               "local_popularity": 10, "category": "cafe", "environment": "indoor"},
        "p2": {"latitude": 1, "longitude": 2, "price_level": 3, "family_suitability": 0,
               "local_popularity": 30, "category": "museum", "environment": "indoor"},
        "future": {"latitude": 99, "longitude": 99, "price_level": 99, "family_suitability": 99,
                   "local_popularity": 999, "category": "future-only", "environment": "outdoor"},
    }
    candidates = [RankingCandidate("ordinary-1", "p1", "cafe", 10, True),
                  RankingCandidate("ordinary-2", "p2", "museum", 30, True),
                  RankingCandidate("unscorable", "future", "future-only", 999, True)]
    embeddings = {"ordinary-1": np.ones(2), "ordinary-2": np.ones(2)}
    batch, state = _candidate_matrix(requests, candidates, request_rows, catalog, embeddings,
        fit_ids={r.request_id for r in requests}, fit=True)
    assert set(batch.request_ids) == {"ordinary-1", "ordinary-2"}
    assert state["mean"][0] == 20
    assert state["vocabularies"]["category"] == ["cafe", "museum"]
    with pytest.raises(ValueError, match="empty scorable training split"):
        _candidate_matrix(requests, candidates, request_rows, catalog, {},
            fit_ids={r.request_id for r in requests}, fit=True)


def test_frozen_head_is_deterministic_and_preserves_typed_identity_order() -> None:
    batch = RankingTrainingBatch(("r", "r"), ("poi-b", "poi-a"), np.asarray([[1., 0.], [1., 1.]]),
                                 np.asarray([0., 1.]))
    first = train_frozen_head(batch, seed=7)
    second = train_frozen_head(batch, seed=7)
    assert np.array_equal(first, second)
    assert batch.poi_ids == ("poi-b", "poi-a")
    assert float(batch.features[1] @ first) > float(batch.features[0] @ first)


def _simulate(root: Path) -> None:
    config_path = Path("configs/simulation/kanto_v1.yaml")
    config = simulator.load_config(config_path)
    config["run"].update(users=10, days=2, seed=20260812, output=str(root))
    simulator.activate_config(config)
    simulator.simulate(argparse.Namespace(output=str(root), overwrite=False, seed=20260812, users=10, days=2,
        start_date=config["run"]["start_date"], scenario=config["run"]["scenario"], full_kanto=False, config=str(config_path)))


def test_rankers_share_sets_reject_v1_and_never_open_truth(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    experiment = tmp_path / "experiment"
    _simulate(run)
    real_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        if "truth" in Path(file).parts:
            raise AssertionError("rank opened truth")
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    reports = []
    for model in ("popularity", "nearest", "category_preference"):
        monkeypatch.setattr("sys.argv", ["geoembed", "rank", "--run-dir", str(run),
            "--experiment-dir", str(experiment), "--model", model])
        main()
        reports.append(json.loads((experiment / "ranking" / f"{model}.json").read_text()))
        artifact = np.load(experiment / "ranking" / f"{model}.npz", allow_pickle=False)
        assert len(artifact["request_id"]) == reports[-1]["coverage"]["available_candidates"]
    assert len({report["request_hash"] for report in reports}) == 1
    assert len({report["candidate_hash"] for report in reports}) == 1

    # Complete the observed-only cross-stage path with the canonical timestamped dense schema.
    request_rows = __import__("pandas").read_csv(run / "observed" / "recommendation_requests.csv.gz")
    timestamps = sorted(request_rows["request_timestamp"].astype(str))
    train_end = timestamps[max(0, len(timestamps) // 2)]
    prepared = experiment / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "prepared_metadata.json").write_text(json.dumps({"train_end": train_end,
        "validation_end": train_end, "timestamp_max": timestamps[-1]}))
    dense_rows = request_rows[["user_id", "request_timestamp"]].astype(str).drop_duplicates()
    users = dense_rows["user_id"].to_numpy()
    dense_timestamps = dense_rows["request_timestamp"].to_numpy()
    matrix = np.ones((len(users), 2))
    prepared_hash = sha256_file(prepared / "prepared_metadata.json")
    np.savez_compressed(experiment / "dense_embeddings.npz", user_id=users,
        timestamp=dense_timestamps, cutoff_kind=np.asarray(["request_time"] * len(users)),
        embedding=matrix, component_persistent=matrix, component_context=np.zeros_like(matrix),
        component_combined=matrix, component_names=np.asarray(COMPONENT_NAMES),
        component_dimensions=np.asarray([2, 2, 2]), schema_version=np.asarray(EXPORT_SCHEMA_VERSION),
        model_variant=np.asarray("single_vector"), categorical_fields=np.asarray([], dtype=str),
        continuous_fields=np.asarray([], dtype=str), preparation_hash=np.asarray(prepared_hash),
        source_file_names=np.asarray(["observed_events.csv.gz"]),
        source_hashes=np.asarray([sha256_file(run / "observed" / "observed_events.csv.gz")]),
        train_end=np.asarray(train_end), validation_end=np.asarray(train_end),
        export_cutoffs=dense_timestamps, compatibility=np.asarray("embedding aliases component_combined"))
    monkeypatch.setattr("sys.argv", ["geoembed", "rank", "--run-dir", str(run),
        "--experiment-dir", str(experiment), "--model", "frozen_embedding"])
    main()
    frozen_report = json.loads((experiment / "ranking" / "frozen_embedding.json").read_text())
    assert frozen_report["request_hash"] == reports[0]["request_hash"]
    assert frozen_report["candidate_hash"] == reports[0]["candidate_hash"]
    assert set(frozen_report["baseline_comparisons"]) == {"popularity", "nearest", "category_preference"}
    assert frozen_report["split_counts"]["training"]["scorable"] > 1
    assert set(frozen_report["split_counts"]) == {"training", "validation", "test"}
    assert (experiment / "ranking" / "frozen_embedding_checkpoint.npz").is_file()

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_contract"]["version"] = "1.0"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["geoembed", "rank", "--run-dir", str(run),
        "--experiment-dir", str(tmp_path / "legacy"), "--model", "nearest"])
    with pytest.raises((FileNotFoundError, ValueError), match="recommendation|Incomplete"):
        main()
