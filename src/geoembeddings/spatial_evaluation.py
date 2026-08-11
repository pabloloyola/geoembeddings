"""Versioned, leakage-safe spatial and geographic-transfer evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .schema import load_observed


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """Return great-circle distance in kilometres, with broadcast semantics."""
    values = [np.asarray(value, dtype=np.float64) for value in (lat1, lon1, lat2, lon2)]
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Distance coordinates must be finite")
    a, b, c, d = map(np.radians, values)
    delta_lat, delta_lon = c - a, d - b
    term = np.sin(delta_lat / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(term, 0.0, 1.0)))


def fit_spatial_contract(train_events: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    """Fit labels and the relevance radius from training rows only."""
    fields = list(settings.get("geohash_fields", []))
    required = {"latitude", "longitude", "region_id", *fields}
    missing = required.difference(train_events.columns)
    if missing:
        raise ValueError(f"Training events lack spatial fields: {sorted(missing)}")
    if train_events.empty:
        raise ValueError("Spatial fitting requires at least one training event")
    coordinates = train_events[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Training coordinates must be finite")
    # Consecutive within-user distances provide an opportunity-aware scale without test access.
    ordered = train_events.sort_values(["user_id", "timestamp"])
    distances: list[float] = []
    for _, group in ordered.groupby("user_id", sort=False):
        if len(group) > 1:
            distances.extend(haversine_km(group.latitude.iloc[:-1], group.longitude.iloc[:-1],
                                          group.latitude.iloc[1:], group.longitude.iloc[1:]).tolist())
    quantile = float(settings.get("distance_relevance_quantile", 0.5))
    if not 0 <= quantile <= 1:
        raise ValueError("distance_relevance_quantile must be in [0, 1]")
    radius = float(np.quantile(distances, quantile)) if distances else 0.0
    return {"known_labels": {field: sorted(train_events[field].dropna().astype(str).unique()) for field in fields},
            "known_regions": sorted(train_events["region_id"].dropna().astype(str).unique()),
            "distance_relevance_radius_km": radius, "fitting_rows": int(len(train_events)),
            "fitting_max_timestamp": str(pd.to_datetime(train_events["timestamp"], utc=True).max()),
            "distance_sample_count": len(distances)}


def validate_train_only_geography(vocabularies: dict[str, Any], fitted: dict[str, Any]) -> None:
    """Reject a preparation/evaluation handoff whose vocabulary used held-out rows."""
    for field, labels in fitted["known_labels"].items():
        vocab_known = sorted(set(vocabularies.get(field, {})) - {"<PAD>", "<UNK>"})
        if vocab_known != labels:
            raise ValueError(f"Geographic split leakage/stale preparation for {field}: vocabulary is not train-only")


def _load_embeddings(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"user_id", "cutoff", "embedding"}
        if not required.issubset(payload.files):
            raise ValueError(f"Embedding export lacks arrays: {sorted(required - set(payload.files))}")
        vectors = np.asarray(payload["embedding"], dtype=np.float64)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be finite and two-dimensional")
        result: dict[tuple[str, str], np.ndarray] = {}
        for user, cutoff, vector in zip(payload["user_id"].astype(str), payload["cutoff"].astype(str), vectors):
            if (user, cutoff) in result:
                raise ValueError(f"Duplicate embedding key: {(user, cutoff)}")
            result[(user, cutoff)] = vector
    return result


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_spatial_transfer(observed_dir: str | Path, prepared_dir: str | Path,
                              embeddings_path: str | Path, output_path: str | Path,
                              config: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Evaluate frozen exports using public geography; this API cannot receive truth/."""
    observed_dir = Path(observed_dir).resolve()
    if observed_dir.name != "observed":
        raise ValueError("Spatial transfer evaluation accepts only the canonical observed/ directory")
    settings = config.get("evaluation", {}).get("transfer", {})
    if settings.get("schema_version") != "spatial-transfer-slices/1.0":
        raise ValueError("Unsupported or missing evaluation.transfer.schema_version")
    metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    metadata = read_json(metadata_path)
    _, events = load_observed(observed_dir)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    train_end, validation_end = pd.Timestamp(metadata["train_end"]), pd.Timestamp(metadata["validation_end"])
    train = events[events.timestamp <= train_end]
    fitted = fit_spatial_contract(train, settings)
    # Reject stale or geographically leaked preparation vocabularies.
    vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
    validate_train_only_geography(vocabularies, fitted)
    target = (events[events.timestamp > validation_end].sort_values(["user_id", "timestamp"])
              .groupby("user_id", as_index=False).first())
    gallery = (train.sort_values(["user_id", "timestamp"]).groupby("user_id", as_index=False).last())
    embedding_map = _load_embeddings(embeddings_path)
    target = target[target.user_id.astype(str).map(lambda u: (u, "test") in embedding_map)]
    gallery = gallery[gallery.user_id.astype(str).map(lambda u: (u, "train") in embedding_map)]
    gallery = gallery.set_index("user_id", drop=False)
    query_rows, retrieval_distances, boundary_cosines = [], [], []
    recalls = {int(k): [] for k in settings.get("retrieval_k", [1, 5])}
    boundary_limit = float(settings.get("boundary_max_distance_km", 2.0))
    if gallery.empty:
        gallery_vectors = np.empty((0, 0))
    else:
        gallery_vectors = np.stack([embedding_map[(str(u), "train")] for u in gallery.user_id])
        gallery_vectors /= np.maximum(np.linalg.norm(gallery_vectors, axis=1, keepdims=True), 1e-12)
    for row in target.itertuples(index=False):
        query = embedding_map[(str(row.user_id), "test")]
        query = query / max(np.linalg.norm(query), 1e-12)
        if len(gallery_vectors):
            order = np.argsort(-(gallery_vectors @ query))
            distances = haversine_km(row.latitude, row.longitude,
                gallery.latitude.to_numpy()[order], gallery.longitude.to_numpy()[order])
            retrieval_distances.append(float(distances[0]))
            for k in recalls:
                recalls[k].append(float(np.any(distances[:min(k, len(distances))] <= fitted["distance_relevance_radius_km"])))
            # A close pair on opposite geohash sides measures boundary brittleness.
            field = settings["geohash_fields"][-1]
            labels = gallery[field].astype(str).to_numpy()[order]
            candidates = np.where((distances <= boundary_limit) & (labels != str(getattr(row, field))))[0]
            if len(candidates):
                boundary_cosines.append(float(gallery_vectors[order[candidates[0]]] @ query))
        query_rows.append(row)
    held = set(map(str, settings.get("held_out_regions", [])))
    slices: dict[str, Any] = {}
    masks = {"later_time": np.ones(len(target), dtype=bool),
             "held_out_region": target.region_id.astype(str).isin(held).to_numpy()}
    for field, labels in fitted["known_labels"].items():
        known = target[field].astype(str).isin(labels).to_numpy()
        masks[f"seen_{field}"] = known
        masks[f"unseen_{field}"] = ~known
    for name, mask in masks.items():
        subset = target.iloc[np.flatnonzero(mask)]
        slices[name] = {"rows": int(mask.sum()), "users": int(subset.user_id.nunique()),
            "coverage_of_later_time": float(mask.mean()) if len(mask) else 0.0,
            "known_label_coverage": {field: float(subset[field].astype(str).isin(labels).mean()) if len(subset) else 0.0
                for field, labels in fitted["known_labels"].items()}}
    definition = {key: settings[key] for key in sorted(settings)}
    definition_hash = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {"metric_contract": {"version": "spatial-transfer-metrics/1.0", "kind": kind,
        "slice_definition": definition, "slice_definition_sha256": definition_hash,
        "source_hashes": metadata["source_files"], "train_end": metadata["train_end"],
        "validation_end": metadata["validation_end"], "prepared_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "users": sorted(target.user_id.astype(str).unique()), "cutoffs": ["train", "test"]},
        "train_only_fit": fitted, "coverage": {"eligible_later_time_rows": len(target), "gallery_users": len(gallery),
            "known_labels": fitted["known_labels"]}, "slices": slices,
        "distance_retrieval": {"metric": "haversine_km", "queries": len(retrieval_distances),
            "mean_top1_distance_km": _mean(retrieval_distances),
            "recall_within_train_fitted_radius": {f"at_{k}": _mean(values) for k, values in recalls.items()}},
        "geohash_boundary_pairs": {"max_distance_km": boundary_limit, "pairs": len(boundary_cosines),
            "mean_cross_boundary_cosine": _mean(boundary_cosines)},
        "axes_are_not_composited": True,
        "limitations": ["Observed geohashes and coordinates include simulator observation noise.",
            "Unseen POI transfer is not measurable under the current observed contract."],
        "information_boundary": "This evaluator uses observed/ and prepared artifacts only; it has no truth_dir parameter."}
    write_json(report, output_path)
    return report
