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
from geoembeddings.ranking import (RankingCandidate, RankingPrediction, RankingRequest,
    compute_ranking_metrics, rank_candidates)


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

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_contract"]["version"] = "1.0"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["geoembed", "rank", "--run-dir", str(run),
        "--experiment-dir", str(tmp_path / "legacy"), "--model", "nearest"])
    with pytest.raises((FileNotFoundError, ValueError), match="recommendation|Incomplete"):
        main()
