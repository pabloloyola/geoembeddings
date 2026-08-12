"""Observable-only naive ranking baselines for dataset contract 2.0."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contract import DATASET_CONTRACT_NAME, DATASET_CONTRACT_VERSION, OBSERVED_FILES
from .io import sha256_file, write_json
from .recommendation import validate_recommendation_tables

RANKING_PREDICTION_SCHEMA = "geoembeddings-ranking-predictions/1.0"
RANKING_REPORT_SCHEMA = "geoembeddings-ranking-report/1.0"
RANKING_MODELS = ("popularity", "nearest", "category_preference")


@dataclass(frozen=True)
class RankingRequest:
    request_id: str
    user_id: str
    timestamp: datetime
    region_id: str


@dataclass(frozen=True)
class RankingCandidate:
    request_id: str
    poi_id: str
    category: str
    travel_time_minutes: float
    is_available: bool


@dataclass(frozen=True)
class RankingPrediction:
    request_id: str
    poi_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RankingMetrics:
    evaluated_requests: int
    recall_at_k: Mapping[str, float]
    ndcg_at_k: Mapping[str, float]
    mrr: float


def _canonical_hash(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    payload = [[str(row[field]) for field in fields] for row in rows]
    payload.sort()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def rank_candidates(
    model: str,
    requests: Sequence[RankingRequest],
    candidates: Sequence[RankingCandidate],
    *,
    interactions: Sequence[Mapping[str, Any]] = (),
    history_events: Sequence[Mapping[str, Any]] = (),
) -> list[RankingPrediction]:
    """Rank available candidates with rolling, strictly-prior observable statistics."""
    if model not in RANKING_MODELS:
        raise ValueError(f"Unsupported ranking model: {model}")
    by_request: dict[str, list[RankingCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.is_available:
            by_request[candidate.request_id].append(candidate)
    parsed_interactions = sorted(
        ((_timestamp(row["interaction_timestamp"]), str(row["poi_id"])) for row in interactions),
        key=lambda item: (item[0], item[1]),
    )
    parsed_events = sorted(
        ((_timestamp(row["timestamp"]), str(row["user_id"]), str(row["object_category"])) for row in history_events),
        key=lambda item: (item[0], item[1], item[2]),
    )
    output: list[RankingPrediction] = []
    for request in sorted(requests, key=lambda item: (item.timestamp, item.request_id)):
        # Strict inequality is the causal boundary: equal/future observations cannot score this request.
        popularity = Counter(poi for timestamp, poi in parsed_interactions if timestamp < request.timestamp)
        preferences = Counter(category for timestamp, user, category in parsed_events
                              if user == request.user_id and timestamp < request.timestamp)
        rows = by_request.get(request.request_id, [])
        def score(candidate: RankingCandidate) -> float:
            if model == "popularity":
                return float(popularity[candidate.poi_id])
            if model == "nearest":
                return -candidate.travel_time_minutes
            return float(preferences[candidate.category])
        # Stable public POI identity is the final, ascending tie-break key.
        ranked = sorted(rows, key=lambda candidate: (-score(candidate), candidate.poi_id))
        output.extend(RankingPrediction(request.request_id, row.poi_id, index, score(row))
                      for index, row in enumerate(ranked, 1))
    return output


def compute_ranking_metrics(
    request_ids: Sequence[str], predictions: Sequence[RankingPrediction],
    positives: Mapping[str, set[str]], ks: Sequence[int],
) -> RankingMetrics:
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ranking K values must be positive")
    ranked: dict[str, list[str]] = defaultdict(list)
    for prediction in sorted(predictions, key=lambda row: (row.request_id, row.rank, row.poi_id)):
        ranked[prediction.request_id].append(prediction.poi_id)
    recall = {str(k): [] for k in sorted(set(ks))}
    ndcg = {str(k): [] for k in sorted(set(ks))}
    reciprocal: list[float] = []
    evaluated = 0
    for request_id in request_ids:
        relevant = positives.get(request_id, set())
        if not relevant:
            continue
        evaluated += 1
        ordered = ranked.get(request_id, [])
        relevant_ranks = [index for index, poi_id in enumerate(ordered, 1) if poi_id in relevant]
        reciprocal.append(0.0 if not relevant_ranks else 1.0 / relevant_ranks[0])
        for key in recall:
            k = int(key)
            hits = sum(poi_id in relevant for poi_id in ordered[:k])
            recall[key].append(hits / len(relevant))
            dcg = sum(1.0 / math.log2(index + 1) for index, poi_id in enumerate(ordered[:k], 1)
                      if poi_id in relevant)
            ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(k, len(relevant)) + 1))
            ndcg[key].append(dcg / ideal if ideal else 0.0)
    mean = lambda values: float(sum(values) / len(values)) if values else 0.0
    return RankingMetrics(evaluated, {key: mean(value) for key, value in recall.items()},
                          {key: mean(value) for key, value in ndcg.items()}, mean(reciprocal))


def run_ranking(
    observed_dir: Path, manifest: Mapping[str, Any], prediction_path: Path, report_path: Path,
    *, model: str, ks: Sequence[int] = (1, 5, 10), overwrite: bool = False,
) -> dict[str, Any]:
    if manifest.get("dataset_contract") != {"name": DATASET_CONTRACT_NAME, "version": DATASET_CONTRACT_VERSION}:
        raise ValueError("rank requires dataset contract 2.0 recommendation tables")
    if prediction_path.exists() or report_path.exists():
        if not overwrite:
            raise FileExistsError("ranking artifacts already exist; use --overwrite to replace both")
    frames = {name: pd.read_csv(observed_dir / OBSERVED_FILES[name], keep_default_na=False)
              for name in ("poi_catalog", "recommendation_requests", "impressions", "interactions")}
    events = pd.read_csv(observed_dir / OBSERVED_FILES["events"], keep_default_na=False)
    tables = {name: frame.to_dict("records") for name, frame in frames.items()}
    validate_recommendation_tables(tables)
    catalog = {str(row["poi_id"]): row for row in tables["poi_catalog"]}
    requests = [RankingRequest(str(row["request_id"]), str(row["user_id"]), _timestamp(row["request_timestamp"]),
                               str(row["region_id"])) for row in tables["recommendation_requests"]]
    candidates = [RankingCandidate(str(row["request_id"]), str(row["poi_id"]),
                    str(catalog[str(row["poi_id"])]["category"]), float(row["travel_time_minutes"]),
                    bool(int(row["is_available"]))) for row in tables["impressions"]]
    if any(not math.isfinite(row.travel_time_minutes) or row.travel_time_minutes < 0 for row in candidates):
        raise ValueError("candidate travel_time_minutes must be finite and non-negative")
    predictions = rank_candidates(model, requests, candidates, interactions=tables["interactions"],
                                  history_events=events.to_dict("records"))
    positives: dict[str, set[str]] = defaultdict(set)
    for row in tables["interactions"]:
        positives[str(row["request_id"])].add(str(row["poi_id"]))
    metrics = compute_ranking_metrics([row.request_id for row in requests], predictions, positives, ks)
    available = [row for row in candidates if row.is_available]
    request_hash = _canonical_hash(tables["recommendation_requests"], ("request_id", "user_id", "request_timestamp"))
    candidate_hash = _canonical_hash((asdict(row) for row in available),
                                     ("request_id", "poi_id", "category", "travel_time_minutes", "is_available"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, schema_version=np.array(RANKING_PREDICTION_SCHEMA), model=np.array(model),
        request_id=np.array([row.request_id for row in predictions]), poi_id=np.array([row.poi_id for row in predictions]),
        rank=np.array([row.rank for row in predictions], dtype=np.int64),
        score=np.array([row.score for row in predictions], dtype=np.float64),
        request_hash=np.array(request_hash), candidate_hash=np.array(candidate_hash))
    source_hashes = {name: sha256_file(observed_dir / OBSERVED_FILES[name])
                     for name in ("events", "poi_catalog", "recommendation_requests", "impressions", "interactions")}
    report = {
        "schema_version": RANKING_REPORT_SCHEMA, "prediction_schema_version": RANKING_PREDICTION_SCHEMA,
        "model": model, "request_hash": request_hash, "candidate_hash": candidate_hash,
        "source_manifest_identity": {"dataset_contract": manifest["dataset_contract"],
            "manifest_content_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(), "identity": manifest.get("identity")},
        "source_hashes": source_hashes,
        "cutoff_definitions": {
            "statistics": "strictly interaction_timestamp/event timestamp < each request_timestamp",
            "labels": "all observable interactions associated with the request",
            "availability": "is_available == 1 with availability_observed_at <= request_timestamp",
        },
        "scorer_configuration": {"model": model, "tie_break": "poi_id ascending", "k": sorted(set(ks)),
            "popularity": "rolling prior interaction count", "category_preference": "rolling per-user prior observed-event category count",
            "nearest": "negative request-time travel_time_minutes"},
        "coverage": {"requests_total": len(requests), "requests_scored": len({row.request_id for row in predictions}),
            "request_coverage": len({row.request_id for row in predictions}) / len(requests) if requests else 0.0,
            "available_candidates": len(available), "requests_with_interactions": metrics.evaluated_requests},
        "metrics": asdict(metrics), "predictions": prediction_path.name,
    }
    write_json(report, report_path)
    return report
