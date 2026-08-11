"""Versioned, observable-only recommendation table contract and diagnostics."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable

RECOMMENDATION_SCHEMA_VERSION = "geoembeddings-recommendation-observed/1.0"
POI_FIELDS = ("poi_id", "region_id", "prefecture", "category", "latitude", "longitude", "price_level",
              "family_suitability", "environment", "local_popularity", "opens_at", "closes_at", "catalog_valid_from")
REQUEST_FIELDS = ("request_id", "user_id", "request_timestamp", "region_id", "latitude", "longitude", "context_source")
IMPRESSION_FIELDS = ("request_id", "poi_id", "candidate_position", "is_available", "is_shown", "shown_rank",
                     "travel_time_minutes", "availability_observed_at")
INTERACTION_FIELDS = ("interaction_id", "request_id", "poi_id", "interaction_timestamp", "interaction_type")
FORBIDDEN_PARTS = ("utility", "probability", "latent", "episode", "counterfactual", "inaccessible", "is_chosen", "true_")


def validate_recommendation_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    specs = {"poi_catalog": POI_FIELDS, "recommendation_requests": REQUEST_FIELDS,
             "impressions": IMPRESSION_FIELDS, "interactions": INTERACTION_FIELDS}
    for name, fields in specs.items():
        rows = tables[name]
        if rows and tuple(rows[0]) != fields:
            raise ValueError(f"{name} field order must be {fields}")
        for row in rows:
            if tuple(row) != fields:
                raise ValueError(f"{name} has inconsistent field order")
            leaked = [field for field in row if any(part in field.lower() for part in FORBIDDEN_PARTS)]
            if leaked:
                raise ValueError(f"protected fields in observed {name}: {leaked}")
    pois = {str(row["poi_id"]) for row in tables["poi_catalog"]}
    requests = {str(row["request_id"]): row for row in tables["recommendation_requests"]}
    if len(pois) != len(tables["poi_catalog"]) or len(requests) != len(tables["recommendation_requests"]):
        raise ValueError("recommendation primary IDs must be unique")
    pairs: set[tuple[str, str]] = set()
    for row in tables["impressions"]:
        key = (str(row["request_id"]), str(row["poi_id"]))
        if key in pairs or key[0] not in requests or key[1] not in pois:
            raise ValueError("impression references must resolve uniquely")
        pairs.add(key)
        observed = datetime.fromisoformat(str(row["availability_observed_at"]))
        requested = datetime.fromisoformat(str(requests[key[0]]["request_timestamp"]))
        if observed > requested or int(row["is_shown"]) > int(row["is_available"]):
            raise ValueError("availability must be observed by request time and shown POIs must be available")
    for row in tables["interactions"]:
        key = (str(row["request_id"]), str(row["poi_id"]))
        if key not in pairs or datetime.fromisoformat(str(row["interaction_timestamp"])) < datetime.fromisoformat(str(requests[key[0]]["request_timestamp"])):
            raise ValueError("interactions must reference candidates and occur after the request")
    return {"schema_version": RECOMMENDATION_SCHEMA_VERSION, "requests": len(requests), "candidates": len(pairs),
            "available_candidates": sum(int(r["is_available"]) for r in tables["impressions"])}


def naive_ranker_diagnostics(tables: dict[str, list[dict[str, Any]]], events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Execute three controls using public fields only; this is a readiness gate, not T3.4 evaluation."""
    catalog = {str(row["poi_id"]): row for row in tables["poi_catalog"]}
    preferences: dict[str, Counter[str]] = {}
    for event in events:
        preferences.setdefault(str(event["user_id"]), Counter())[str(event["object_category"])] += 1
    users = {str(r["request_id"]): str(r["user_id"]) for r in tables["recommendation_requests"]}
    positives = {(str(r["request_id"]), str(r["poi_id"])) for r in tables["interactions"]}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in tables["impressions"]:
        if int(row["is_available"]): candidates.setdefault(str(row["request_id"]), []).append(row)
    hits = {name: 0 for name in ("popularity", "nearest_poi", "category_preference")}
    for request_id, rows in candidates.items():
        ranked = {
            "popularity": max(rows, key=lambda r: (float(catalog[str(r["poi_id"])]["local_popularity"]), str(r["poi_id"]))),
            "nearest_poi": min(rows, key=lambda r: (float(r["travel_time_minutes"]), str(r["poi_id"]))),
            "category_preference": max(rows, key=lambda r: (preferences.get(users[request_id], Counter())[str(catalog[str(r["poi_id"])]["category"])], str(r["poi_id"]))),
        }
        for name, row in ranked.items(): hits[name] += int((request_id, str(row["poi_id"])) in positives)
    return {name: {"requests_scored": len(candidates), "top1_interactions": hit} for name, hit in hits.items()}
