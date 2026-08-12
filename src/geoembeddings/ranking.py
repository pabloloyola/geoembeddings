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
RANKING_MODELS = ("popularity", "nearest", "category_preference", "frozen_embedding")
FROZEN_CHECKPOINT_SCHEMA = "geoembeddings-frozen-ranking-checkpoint/1.0"
FROZEN_FEATURES = ("travel_time_minutes", "request_latitude", "request_longitude", "poi_latitude",
                   "poi_longitude", "price_level", "family_suitability", "local_popularity",
                   "category", "environment", "context_source")


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


@dataclass(frozen=True)
class FrozenEmbeddingRows:
    """Validated frozen representation rows at named, timestamped cutoffs."""
    user_ids: np.ndarray
    cutoff_names: np.ndarray
    cutoff_times: np.ndarray
    values: np.ndarray
    component: str


@dataclass(frozen=True)
class RankingTrainingBatch:
    """Explicit observed-only boundary consumed by the scoring-head trainer."""
    request_ids: tuple[str, ...]
    poi_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray


def _canonical_hash(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    payload = [[str(row[field]) for field in fields] for row in rows]
    payload.sort()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def select_causal_embeddings(rows: FrozenEmbeddingRows, requests: Sequence[RankingRequest]) -> dict[str, np.ndarray]:
    if rows.values.ndim != 2 or len(rows.user_ids) != len(rows.cutoff_names) or len(rows.user_ids) != len(rows.values):
        raise ValueError("dimensionally inconsistent embedding export")
    if not np.isfinite(rows.values).all():
        raise ValueError("non-finite embedding export")
    keys = list(zip(rows.user_ids.astype(str), rows.cutoff_names.astype(str), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate user/cutoff embedding identity")
    output: dict[str, np.ndarray] = {}
    for request in requests:
        eligible = [(rows.cutoff_times[i], rows.values[i]) for i, user in enumerate(rows.user_ids.astype(str))
                    if user == request.user_id and rows.cutoff_times[i] <= request.timestamp]
        if eligible:
            output[request.request_id] = max(eligible, key=lambda item: item[0])[1]
    return output


def _load_frozen_embeddings(path: Path, prepared: Mapping[str, Any], events_hash: str) -> FrozenEmbeddingRows:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen embedding export: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"user_id", "cutoff", "source_file_names", "source_hashes"}
        if not required.issubset(data.files):
            raise ValueError("embedding export lacks required lineage")
        source = dict(zip(data["source_file_names"].astype(str), data["source_hashes"].astype(str), strict=True))
        event_name = OBSERVED_FILES["events"]
        if source.get(event_name) != events_hash:
            raise ValueError("embedding source hashes mismatch observed events")
        component = "combined"
        key = "component_combined" if "component_combined" in data.files else "embedding"
        values = np.asarray(data[key], dtype=np.float64)
        names = data["cutoff"].astype(str)
        cutoff_map = {"train": _timestamp(prepared["train_end"]),
                      "validation": _timestamp(prepared["validation_end"]),
                      "test": _timestamp(prepared["timestamp_max"])}
        unknown = sorted(set(names) - set(cutoff_map))
        if unknown:
            raise ValueError(f"unknown cutoff identities: {unknown}")
        return FrozenEmbeddingRows(data["user_id"].astype(str), names,
            np.asarray([cutoff_map[name] for name in names], dtype=object), values, component)


def _candidate_matrix(requests: Sequence[RankingRequest], candidates: Sequence[RankingCandidate],
                      request_rows: Mapping[str, Mapping[str, Any]], catalog: Mapping[str, Mapping[str, Any]],
                      embeddings: Mapping[str, np.ndarray], *, fit_ids: set[str], fit: bool,
                      state: Mapping[str, Any] | None = None) -> tuple[RankingTrainingBatch, dict[str, Any]]:
    categorical = ("category", "environment", "context_source")
    numeric = FROZEN_FEATURES[:8]
    if fit:
        vocab = {field: sorted({str((catalog[c.poi_id] if field != "context_source" else request_rows[c.request_id])[field])
                                for c in candidates if c.request_id in fit_ids}) for field in categorical}
    else:
        assert state is not None
        vocab = state["vocabularies"]
    order = list(numeric) + [f"{field}={value}" for field in categorical for value in vocab[field]]
    raw: list[list[float]] = []; ids: list[str] = []; pois: list[str] = []
    for c in candidates:
        if not c.is_available or c.request_id not in embeddings:
            continue
        rr, poi = request_rows[c.request_id], catalog[c.poi_id]
        vals = [c.travel_time_minutes, float(rr["latitude"]), float(rr["longitude"]), float(poi["latitude"]),
                float(poi["longitude"]), float(poi["price_level"]), float(poi["family_suitability"]),
                float(poi["local_popularity"])]
        vals += [float(str((poi if field != "context_source" else rr)[field]) == value)
                 for field in categorical for value in vocab[field]]
        raw.append(vals); ids.append(c.request_id); pois.append(c.poi_id)
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError("candidate features must be finite")
    if fit:
        train_mask = np.asarray([rid in fit_ids for rid in ids]); mean = x[train_mask].mean(0); scale = x[train_mask].std(0); scale[scale == 0] = 1
        state = {"feature_order": order, "vocabularies": vocab, "mean": mean, "scale": scale}
    assert state is not None
    if order != list(state["feature_order"]): raise ValueError("candidate feature field order mismatch")
    z = (x - state["mean"]) / state["scale"]
    design = np.asarray([np.concatenate(([1.0], z[i], embeddings[rid], np.outer(embeddings[rid], z[i]).ravel()))
                         for i, rid in enumerate(ids)])
    return RankingTrainingBatch(tuple(ids), tuple(pois), design, np.zeros(len(ids))), dict(state)


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


def train_frozen_head(batch: RankingTrainingBatch, *, seed: int = 20260812,
                      iterations: int = 300, learning_rate: float = .05) -> np.ndarray:
    """Fit only a deterministic logistic candidate head; embedding values are immutable inputs."""
    if not len(batch.features): raise ValueError("no training candidates")
    if batch.features.ndim != 2 or batch.labels.shape != (len(batch.features),):
        raise ValueError("invalid typed ranking training batch")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1e-4, batch.features.shape[1])
    for _ in range(iterations):
        logits = np.clip(batch.features @ weights, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        weights -= learning_rate * ((batch.features.T @ (probabilities - batch.labels)) / len(batch.labels)
                                    + 1e-4 * weights)
    return weights


def run_ranking(
    observed_dir: Path, manifest: Mapping[str, Any], prediction_path: Path, report_path: Path,
    *, model: str, ks: Sequence[int] = (1, 5, 10), overwrite: bool = False,
    embedding_path: Path | None = None, checkpoint_path: Path | None = None,
    baseline_report_paths: Mapping[str, Path] | None = None,
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
    frozen_lineage: dict[str, Any] | None = None
    unscorable: dict[str, str] = {}
    if model == "frozen_embedding":
        if embedding_path is None or checkpoint_path is None:
            raise ValueError("frozen_embedding requires canonical embedding and checkpoint paths")
        prepared_path = embedding_path.parent / "prepared" / "prepared_metadata.json"
        if not prepared_path.is_file(): raise FileNotFoundError("missing prepared metadata for frozen export")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        events_hash = sha256_file(observed_dir / OBSERVED_FILES["events"])
        frozen = _load_frozen_embeddings(embedding_path, prepared, events_hash)
        selected = select_causal_embeddings(frozen, requests)
        for request in requests:
            if request.request_id not in selected:
                unscorable[request.request_id] = "no_embedding_at_or_before_request_cutoff"
        request_rows = {str(row["request_id"]): row for row in tables["recommendation_requests"]}
        train_ids = {r.request_id for r in requests if r.timestamp <= _timestamp(prepared["train_end"])}
        if not train_ids: raise ValueError("no training requests at or before train_end")
        all_batch, preprocessing = _candidate_matrix(requests, candidates, request_rows, catalog, selected,
                                                       fit_ids=train_ids, fit=True)
        positive_pairs = {(str(row["request_id"]), str(row["poi_id"])) for row in tables["interactions"]
                          if str(row["request_id"]) in train_ids}
        train_indices = [i for i, rid in enumerate(all_batch.request_ids) if rid in train_ids]
        labels = np.asarray([float((all_batch.request_ids[i], all_batch.poi_ids[i]) in positive_pairs)
                             for i in train_indices])
        training = RankingTrainingBatch(tuple(all_batch.request_ids[i] for i in train_indices),
            tuple(all_batch.poi_ids[i] for i in train_indices), all_batch.features[train_indices], labels)
        weights = train_frozen_head(training)
        scores = all_batch.features @ weights
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for rid, poi, score in zip(all_batch.request_ids, all_batch.poi_ids, scores, strict=True):
            grouped[rid].append((poi, float(score)))
        predictions = []
        for request in sorted(requests, key=lambda row: (row.timestamp, row.request_id)):
            ranked = sorted(grouped.get(request.request_id, []), key=lambda row: (-row[1], row[0]))
            predictions.extend(RankingPrediction(request.request_id, poi, rank, score)
                               for rank, (poi, score) in enumerate(ranked, 1))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists() and not overwrite: raise FileExistsError("frozen ranking checkpoint exists")
        np.savez_compressed(checkpoint_path, schema_version=np.asarray(FROZEN_CHECKPOINT_SCHEMA), weights=weights,
            feature_order=np.asarray(preprocessing["feature_order"]), mean=preprocessing["mean"],
            scale=preprocessing["scale"], embedding_component=np.asarray(frozen.component),
            embedding_dimension=np.asarray(frozen.values.shape[1]), seed=np.asarray(20260812))
        checkpoint_hash = sha256_file(checkpoint_path)
        frozen_lineage = {"checkpoint": checkpoint_path.name, "checkpoint_sha256": checkpoint_hash,
            "embedding_export": embedding_path.name, "embedding_export_sha256": sha256_file(embedding_path),
            "embedding_component": frozen.component, "embedding_dimension": int(frozen.values.shape[1]),
            "feature_order": preprocessing["feature_order"], "cutoffs": sorted(set(frozen.cutoff_names.astype(str))),
            "split_definition": {"training": "request_timestamp <= prepared train_end", "train_end": prepared["train_end"]},
            "seed": 20260812, "training_requests": len(train_ids), "training_candidates": len(training.labels)}
    else:
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
            "available_candidates": len(available), "requests_with_interactions": metrics.evaluated_requests,
            "users_total": len({r.user_id for r in requests}),
            "users_scored": len({r.user_id for r in requests if r.request_id in {p.request_id for p in predictions}}),
            "unscorable_requests": unscorable},
        "metrics": asdict(metrics), "predictions": prediction_path.name,
    }
    if frozen_lineage is not None:
        report["frozen_embedding_lineage"] = frozen_lineage
        comparisons = {}
        for name, path in (baseline_report_paths or {}).items():
            if path.is_file():
                baseline = json.loads(path.read_text(encoding="utf-8"))
                if baseline.get("request_hash") != request_hash or baseline.get("candidate_hash") != candidate_hash:
                    raise ValueError(f"{name} baseline request/candidate sets mismatch")
                comparisons[name] = {"recall_at_k_delta": {k: metrics.recall_at_k[k] - baseline["metrics"]["recall_at_k"][k] for k in metrics.recall_at_k},
                    "ndcg_at_k_delta": {k: metrics.ndcg_at_k[k] - baseline["metrics"]["ndcg_at_k"][k] for k in metrics.ndcg_at_k},
                    "mrr_delta": metrics.mrr - baseline["metrics"]["mrr"]}
            else: comparisons[name] = {"status": "missing_baseline_report"}
        report["baseline_comparisons"] = comparisons
    write_json(report, report_path)
    return report
