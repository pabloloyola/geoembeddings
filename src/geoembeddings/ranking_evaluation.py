"""Frozen, observed-identity transfer slices for ranking predictions (T3.7)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contract import OBSERVED_FILES
from .io import sha256_file, write_json
from .ranking import (RANKING_PREDICTION_SCHEMA, RANKING_REPORT_SCHEMA, RankingPrediction,
                      _canonical_hash, compute_ranking_metrics)
from .recommendation import validate_recommendation_tables

TRANSFER_SPLIT_SCHEMA = "geoembeddings-ranking-transfer-splits/1.0"
TRANSFER_REPORT_SCHEMA = "geoembeddings-ranking-transfer-evaluation/1.0"
DEFAULT_MODELS = ("popularity", "nearest", "category_preference", "frozen_embedding")


def _time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def classify_transfer_slices(
    requests: Sequence[Mapping[str, Any]], impressions: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]], catalog: Sequence[Mapping[str, Any]], *, train_end: datetime,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Fit identity sets at ``timestamp <= train_end`` and apply them without refitting."""
    catalog_by_poi = {str(row["poi_id"]): row for row in catalog}
    if len(catalog_by_poi) != len(catalog):
        raise ValueError("duplicate POI identity in catalog")
    request_by_id = {str(row["request_id"]): row for row in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("duplicate request identity")
    observed_rows = [*impressions, *interactions]
    unknown = sorted({str(row["poi_id"]) for row in observed_rows} - set(catalog_by_poi))
    if unknown:
        raise ValueError(f"unknown POIs in observable ranking tables: {unknown[:5]}")
    unknown_requests = sorted({str(row["request_id"]) for row in observed_rows} - set(request_by_id))
    if unknown_requests:
        raise ValueError(f"unknown requests in observable ranking tables: {unknown_requests[:5]}")

    training_requests = {rid for rid, row in request_by_id.items()
                         if _time(row["request_timestamp"]) <= train_end}
    # Impressions and interactions have no independent eligibility clock: their request identity
    # places them in training. Catalog metadata is used only to map an observed POI to its region.
    seen_pois = {str(row["poi_id"]) for row in observed_rows
                 if str(row["request_id"]) in training_requests}
    seen_regions = {str(request_by_id[rid]["region_id"]) for rid in training_requests}
    seen_regions |= {str(catalog_by_poi[poi]["region_id"]) for poi in seen_pois}

    evaluation_ids = {rid for rid, row in request_by_id.items() if _time(row["request_timestamp"]) > train_end}
    candidate_pois = defaultdict(set)
    for row in impressions:
        if int(row["is_available"]) == 1:
            candidate_pois[str(row["request_id"])].add(str(row["poi_id"]))
    first_time: dict[tuple[str, str], datetime] = {}
    for rid in evaluation_ids:
        row = request_by_id[rid]
        key = (str(row["user_id"]), str(row["region_id"]))
        timestamp = _time(row["request_timestamp"])
        first_time[key] = min(timestamp, first_time.get(key, timestamp))

    flags: dict[str, set[str]] = defaultdict(set)
    for rid in evaluation_ids:
        row = request_by_id[rid]
        region_state = "seen_region" if str(row["region_id"]) in seen_regions else "unseen_region"
        pois = candidate_pois[rid]
        # Request-level POI slices overlap when a candidate set mixes identities. This preserves
        # the request and measures the relevant candidate subset rather than assigning it arbitrarily.
        poi_states = []
        if pois & seen_pois:
            poi_states.append("seen_poi")
        if pois - seen_pois:
            poi_states.append("unseen_poi")
        key = (str(row["user_id"]), str(row["region_id"]))
        stage = "early_trip" if _time(row["request_timestamp"]) == first_time[key] else "late_trip"
        flags[region_state].add(rid); flags[stage].add(rid)
        for poi_state in poi_states:
            flags[poi_state].add(rid)
            flags[f"{region_state}_{poi_state}"].add(rid)
            flags[f"{region_state}_{poi_state}_{stage}"].add(rid)
    definitions = {
        "schema_version": TRANSFER_SPLIT_SCHEMA,
        "training_boundary": "request_timestamp <= train_end (cutoff equality is training)",
        "evaluation_boundary": "request_timestamp > train_end",
        "seen_poi": "POI identity occurs in an impression or interaction whose request is in training",
        "seen_region": "region occurs on a training request or is catalog metadata for a seen POI",
        "catalog_policy": "catalog metadata maps already-observed POI identities to regions; catalog presence alone never makes an identity seen",
        "poi_slice_policy": "request belongs to each POI state represented by an available candidate; mixed sets therefore overlap",
        "trip_stage": "within each user and request-region after train_end, all requests at the earliest observable timestamp are early; strictly later timestamps are late",
        "train_end": train_end.isoformat(),
        "training_request_ids_sha256": _canonical_hash(({"request_id": rid} for rid in training_requests), ("request_id",)),
        "seen_region_ids_sha256": _canonical_hash(({"region_id": value} for value in seen_regions), ("region_id",)),
        "seen_poi_ids_sha256": _canonical_hash(({"poi_id": value} for value in seen_pois), ("poi_id",)),
        "counts": {"training_requests": len(training_requests), "evaluation_requests": len(evaluation_ids),
                   "seen_regions": len(seen_regions), "seen_pois": len(seen_pois)},
    }
    return dict(flags), definitions


def _load_predictions(path: Path, report: Mapping[str, Any]) -> list[RankingPrediction]:
    with np.load(path, allow_pickle=False) as arrays:
        if str(np.asarray(arrays["schema_version"]).item()) != RANKING_PREDICTION_SCHEMA:
            raise ValueError(f"unsupported prediction schema: {path}")
        if str(np.asarray(arrays["request_hash"]).item()) != report.get("request_hash") or \
                str(np.asarray(arrays["candidate_hash"]).item()) != report.get("candidate_hash"):
            raise ValueError(f"prediction request/candidate hashes do not match source ranking report: {path.name}")
        rows = [RankingPrediction(str(rid), str(poi), int(rank), float(score)) for rid, poi, rank, score in
                zip(arrays["request_id"], arrays["poi_id"], arrays["rank"], arrays["score"], strict=True)]
    identities = [(row.request_id, row.poi_id) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate predictions in {path.name}")
    by_request: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if not np.isfinite(row.score) or row.rank < 1:
            raise ValueError(f"invalid prediction value in {path.name}")
        by_request[row.request_id].append(row.rank)
    if any(sorted(ranks) != list(range(1, len(ranks) + 1)) for ranks in by_request.values()):
        raise ValueError(f"prediction ranks are not contiguous in {path.name}")
    return rows


def evaluate_ranking_transfer(
    observed_dir: Path, ranking_dir: Path, output_path: Path, *, models: Sequence[str] = DEFAULT_MODELS,
    ks: Sequence[int] = (1, 5, 10), overwrite: bool = False,
) -> dict[str, Any]:
    if set(models) != set(DEFAULT_MODELS) or len(models) != len(DEFAULT_MODELS):
        raise ValueError("evaluate-ranking requires all T3.4 controls and the corrected T3.5 frozen ranker")
    if output_path.exists() and not overwrite:
        raise FileExistsError("ranking transfer artifact exists; use --overwrite to replace it")
    frames = {name: pd.read_csv(observed_dir / OBSERVED_FILES[name], keep_default_na=False)
              for name in ("poi_catalog", "recommendation_requests", "impressions", "interactions")}
    tables = {name: frame.to_dict("records") for name, frame in frames.items()}
    validate_recommendation_tables(tables)
    reports: dict[str, Any] = {}
    predictions: dict[str, list[RankingPrediction]] = {}
    for model in models:
        report_path, prediction_path = ranking_dir / f"{model}.json", ranking_dir / f"{model}.npz"
        if not report_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"missing ranking report/predictions for {model}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != RANKING_REPORT_SCHEMA or report.get("model") != model:
            raise ValueError(f"invalid source ranking report for {model}")
        reports[model] = report
        predictions[model] = _load_predictions(prediction_path, report)
    request_hashes = {report["request_hash"] for report in reports.values()}
    candidate_hashes = {report["candidate_hash"] for report in reports.values()}
    if len(request_hashes) != 1 or len(candidate_hashes) != 1:
        raise ValueError("source ranking reports use mismatched request/candidate hashes")
    source_hashes = {name: sha256_file(observed_dir / OBSERVED_FILES[name]) for name in
                     ("users", "events", "poi_catalog", "recommendation_requests", "impressions", "interactions")}
    for model, report in reports.items():
        if report.get("source_hashes") != source_hashes:
            raise ValueError(f"{model} source hashes do not match current observed tables")
    frozen = reports.get("frozen_embedding", {}).get("frozen_embedding_lineage", {})
    split = frozen.get("split_definition")
    if not split or "train_end" not in split:
        raise ValueError("frozen_embedding report lacks the frozen training split contract")
    flags, definitions = classify_transfer_slices(tables["recommendation_requests"], tables["impressions"],
        tables["interactions"], tables["poi_catalog"], train_end=_time(split["train_end"]))
    request_rows = {str(row["request_id"]): row for row in tables["recommendation_requests"]}
    available = {(str(row["request_id"]), str(row["poi_id"])) for row in tables["impressions"]
                 if int(row["is_available"]) == 1}
    positives = defaultdict(set)
    for row in tables["interactions"]:
        positives[str(row["request_id"])].add(str(row["poi_id"]))
    slices: dict[str, Any] = {}
    all_names = ("seen_region", "unseen_region", "seen_poi", "unseen_poi", "early_trip", "late_trip",
                 "seen_region_seen_poi", "seen_region_unseen_poi", "unseen_region_seen_poi", "unseen_region_unseen_poi")
    all_names += tuple(sorted(name for name in flags if name not in all_names))
    for name in all_names:
        ids = flags.get(name, set())
        expected = {(rid, poi) for rid, poi in available if rid in ids}
        total_positive = sum(len(positives[rid]) for rid in ids)
        model_results = {}
        for model, rows in predictions.items():
            selected = [row for row in rows if row.request_id in ids and (row.request_id, row.poi_id) in expected]
            scored_ids = {row.request_id for row in selected}
            predicted_pairs = {(row.request_id, row.poi_id) for row in selected}
            scored_users = {str(request_rows[rid]["user_id"]) for rid in scored_ids}
            users = {str(request_rows[rid]["user_id"]) for rid in ids}
            metrics = compute_ranking_metrics(sorted(ids), selected, positives, ks)
            model_results[model] = {"metrics": asdict(metrics), "coverage": {
                "requests": {"total": len(ids), "covered": len(scored_ids), "fraction": len(scored_ids) / len(ids) if ids else 0.0},
                "users": {"total": len(users), "covered": len(scored_users), "fraction": len(scored_users) / len(users) if users else 0.0},
                "positive_labels": {"total": total_positive, "covered": sum((rid, poi) in predicted_pairs for rid in ids for poi in positives[rid]),
                                    "fraction": (sum((rid, poi) in predicted_pairs for rid in ids for poi in positives[rid]) / total_positive) if total_positive else 0.0},
                "candidates": {"total": len(expected), "covered": len(predicted_pairs), "fraction": len(predicted_pairs) / len(expected) if expected else 0.0},
            }}
        slices[name] = {"request_count": len(ids), "models": model_results,
                        "empty": not ids, "exclusions": {"outside_slice": len(request_rows) - len(ids)}}
    output = {"schema_version": TRANSFER_REPORT_SCHEMA, "split_contract": definitions,
        "source_hashes": source_hashes, "source_ranking_reports": {model: {
            "report": f"{model}.json", "report_sha256": sha256_file(ranking_dir / f"{model}.json"),
            "predictions": f"{model}.npz", "predictions_sha256": sha256_file(ranking_dir / f"{model}.npz")}
            for model in models}, "request_hash": next(iter(request_hashes)), "candidate_hash": next(iter(candidate_hashes)),
        "metrics": slices, "exclusions": {"training_requests": definitions["counts"]["training_requests"]},
        "utility_regret": {"status": "unavailable", "reason": "observable-only evaluator received no protected utility truth"},
        "limitations": ["Synthetic seen/unseen transfer is not external geographic validity.",
                        "Observable interactions are implicit labels, not utility or counterfactual relevance.",
                        "Mixed seen/unseen candidate sets make POI request slices overlap."],
    }
    write_json(output, output_path)
    return output
