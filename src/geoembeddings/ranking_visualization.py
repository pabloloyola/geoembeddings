"""Authenticated, observed-only R9 rendering for ranking controls."""

from __future__ import annotations

import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contract import OBSERVED_FILES
from .io import sha256_file, write_json
from .ranking import RANKING_PREDICTION_SCHEMA, RANKING_REPORT_SCHEMA
from .recommendation import validate_recommendation_tables

RANKING_VISUALIZATION_SCHEMA = "geoembeddings-ranking-visualization/1.0"
DEFAULT_MODELS = ("popularity", "nearest", "category_preference", "frozen_embedding")
LIMITATIONS = (
    "Rank-order differences are descriptive comparisons, not causal explanations.",
    "Scores are model-specific and are not comparable in magnitude across controls.",
    "No feature attribution is shown because these artifacts do not expose an authenticated score decomposition.",
    "Interactions are observed outcomes subject to platform exposure and missingness, not protected utility.",
    "This observed-only renderer does not read synthetic truth or support protected-utility claims.",
)


def _reject_truth_path(path: Path) -> None:
    if "truth" in path.resolve().parts or path.name != "observed":
        raise ValueError("ranking visualization accepts only the canonical observed directory")


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> str:
    if name not in arrays or np.asarray(arrays[name]).shape != ():
        raise ValueError(f"ranking prediction lacks scalar {name}")
    return str(np.asarray(arrays[name]).item())


def _load_model(ranking_dir: Path, model: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, str]]:
    report_path, prediction_path = ranking_dir / f"{model}.json", ranking_dir / f"{model}.npz"
    if not report_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing ranking artifact pair for {model}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != RANKING_REPORT_SCHEMA or report.get("model") != model:
        raise ValueError(f"{model} ranking report schema/model identity mismatch")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if _scalar(arrays, "schema_version") != RANKING_PREDICTION_SCHEMA or _scalar(arrays, "model") != model:
        raise ValueError(f"{model} prediction schema/model identity mismatch")
    for identity in ("request_hash", "candidate_hash"):
        if _scalar(arrays, identity) != str(report.get(identity)):
            raise ValueError(f"{model} prediction/report {identity} mismatch")
    required = ("request_id", "poi_id", "rank", "score")
    if any(name not in arrays for name in required) or len({len(arrays[name]) for name in required}) != 1:
        raise ValueError(f"{model} prediction columns are missing or row-misaligned")
    if not np.isfinite(arrays["score"]).all():
        raise ValueError(f"{model} predictions contain non-finite scores")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for request_id, poi_id, rank, score in zip(*(arrays[name] for name in required), strict=True):
        key = (str(request_id), str(poi_id))
        if key in identities or int(rank) < 1:
            raise ValueError(f"{model} predictions contain duplicate identities or invalid ranks")
        identities.add(key)
        grouped[key[0]].append({"poi_id": key[1], "rank": int(rank), "score": float(score)})
    for request_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["rank"])
        if [row["rank"] for row in ordered] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{model} ranks are not contiguous for request {request_id}")
        expected = sorted(rows, key=lambda row: (-row["score"], row["poi_id"]))
        if [row["poi_id"] for row in ordered] != [row["poi_id"] for row in expected]:
            raise ValueError(f"{model} rank order violates deterministic score/POI tie ordering")
        grouped[request_id] = ordered
    return report, dict(grouped), {
        "report_sha256": sha256_file(report_path), "predictions_sha256": sha256_file(prediction_path)
    }


def _table(model: str, rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ("rank", "poi_id", "category", "travel_time_minutes", "availability", "price_level",
               "family_suitability", "environment", "local_popularity", "score", "impression_status",
               "observed_interaction")
    head = "".join(f"<th>{html.escape(column.replace('_', ' '))}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns) + "</tr>"
                   for row in rows)
    return f"<section><h2>{html.escape(model)}</h2><p>Score decomposition: unavailable (no authenticated attribution).</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"


def render_ranking_explanation(observed_dir: Path, ranking_dir: Path, output_dir: Path, *,
                               models: Sequence[str] = DEFAULT_MODELS, overwrite: bool = False) -> dict[str, Any]:
    """Authenticate four ranking surfaces and render one deterministic common request."""
    _reject_truth_path(observed_dir)
    output_html, output_metadata = output_dir / "ranking_explanation.html", output_dir / "metadata.json"
    if output_html.exists() or output_metadata.exists():
        if not overwrite:
            raise FileExistsError("ranking visualization already exists; use --overwrite")
    if tuple(models) != DEFAULT_MODELS:
        raise ValueError(f"R9 renderer requires exactly these controls in order: {DEFAULT_MODELS}")
    names = ("poi_catalog", "recommendation_requests", "impressions", "interactions")
    frames = {name: pd.read_csv(observed_dir / OBSERVED_FILES[name], keep_default_na=False) for name in names}
    tables = {name: frame.to_dict("records") for name, frame in frames.items()}
    validate_recommendation_tables(tables)
    source_hashes = {name: sha256_file(observed_dir / OBSERVED_FILES[name]) for name in names}
    reports, predictions, model_files = {}, {}, {}
    for model in models:
        reports[model], predictions[model], model_files[model] = _load_model(ranking_dir, model)
        for source, digest in source_hashes.items():
            if reports[model].get("source_hashes", {}).get(source) != digest:
                raise ValueError(f"{model} observed source hash mismatch: {source}")
    request_hashes = {str(report["request_hash"]) for report in reports.values()}
    candidate_hashes = {str(report["candidate_hash"]) for report in reports.values()}
    if len(request_hashes) != 1 or len(candidate_hashes) != 1:
        raise ValueError("compared models must share identical request and available-candidate hashes")
    common = set.intersection(*(set(predictions[model]) for model in models))
    if not common:
        raise ValueError("no request is scored by every compared model")
    request_id = min(common, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    request = next(row for row in tables["recommendation_requests"] if str(row["request_id"]) == request_id)
    catalog = {str(row["poi_id"]): row for row in tables["poi_catalog"]}
    impressions = {str(row["poi_id"]): row for row in tables["impressions"] if str(row["request_id"]) == request_id}
    interactions: dict[str, list[str]] = defaultdict(list)
    for row in tables["interactions"]:
        if str(row["request_id"]) == request_id:
            interactions[str(row["poi_id"])].append(str(row["interaction_type"]))
    available_ids = {poi for poi, row in impressions.items() if int(row["is_available"]) == 1}
    cards, rank_orders = {}, {}
    for model in models:
        predicted_ids = {row["poi_id"] for row in predictions[model][request_id]}
        if predicted_ids != available_ids:
            raise ValueError(f"{model} candidate set for selected request differs from observed available candidates")
        by_poi = {row["poi_id"]: row for row in predictions[model][request_id]}
        rank_orders[model] = [row["poi_id"] for row in predictions[model][request_id]]
        cards[model] = []
        for poi_id, impression in sorted(impressions.items()):
            poi, prediction = catalog[poi_id], by_poi.get(poi_id)
            cards[model].append({"rank": prediction["rank"] if prediction else "excluded",
                "poi_id": poi_id, "category": poi["category"],
                "travel_time_minutes": impression["travel_time_minutes"],
                "availability": "available" if int(impression["is_available"]) else "unavailable (excluded)",
                "price_level": poi["price_level"], "family_suitability": poi["family_suitability"],
                "environment": poi["environment"], "local_popularity": poi["local_popularity"],
                "score": prediction["score"] if prediction else "not scored",
                "impression_status": (f"shown at {impression['shown_rank']}" if int(impression["is_shown"]) else "not shown"),
                "observed_interaction": ", ".join(sorted(interactions.get(poi_id, []))) or "none recorded"})
    changes = []
    anchor = models[0]
    for model in models[1:]:
        moved = [{"poi_id": poi, anchor: rank_orders[anchor].index(poi) + 1,
                  model: rank_orders[model].index(poi) + 1} for poi in rank_orders[anchor]
                 if rank_orders[anchor].index(poi) != rank_orders[model].index(poi)]
        changes.append({"comparison": f"{anchor} vs {model}", "moved_candidates": moved})
    metadata = {"schema_version": RANKING_VISUALIZATION_SCHEMA, "mode": "observed_only",
        "request_selection": "minimum (SHA-256(request_id), request_id) among requests scored by all models",
        "request_identity": {key: request[key] for key in request},
        "candidate_identity": {"available_candidate_hash": next(iter(candidate_hashes)),
            "available_poi_ids": sorted(available_ids), "unavailable_poi_ids": sorted(set(impressions) - available_ids)},
        "request_hash": next(iter(request_hashes)), "models": {
            model: {"model_identity": reports[model]["model"], **model_files[model]} for model in models},
        "source_hashes": source_hashes, "exclusions": {"unavailable_candidates": sorted(set(impressions) - available_ids),
            "unscored_requests": {model: reports[model]["coverage"].get("unscorable_requests", {}) for model in models}},
        "rank_order_differences": changes, "interpretation_limitations": list(LIMITATIONS),
        "protected_utility": {"status": "unavailable", "reason": "observed-only mode; no truth path is accepted"}}
    output_dir.mkdir(parents=True, exist_ok=True)
    request_summary = " · ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in request.items())
    changed_html = "".join(
        f"<li>{html.escape(item['comparison'])}: " +
        (", ".join(f"{html.escape(move['poi_id'])} ({move[anchor]}→{move[item['comparison'].split(' vs ')[1]]})"
                   for move in item["moved_candidates"]) or "identical order") + "</li>"
        for item in changes)
    limitations_html = "".join(f"<li>{html.escape(value)}</li>" for value in LIMITATIONS)
    document = ("<!doctype html><html><head><meta charset='utf-8'><title>R9 ranking explanation</title>"
        "<style>body{font-family:system-ui;margin:2rem}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}"
        "section{overflow:auto;border:1px solid #ccc;padding:1rem}table{border-collapse:collapse;font-size:.82rem}th,td{border:1px solid #ddd;padding:.35rem}"
        "th{background:#f3f3f3}</style></head><body><h1>Observed-only R9 ranking explanation</h1>"
        f"<p>{request_summary}</p><main>{''.join(_table(model, cards[model]) for model in models)}</main>"
        f"<h2>What changed?</h2><p>Descriptive rank-order differences only; not causal explanations.</p><ul>{changed_html}</ul>"
        f"<h2>Interpretation limitations</h2><ul>{limitations_html}</ul></body></html>")
    output_html.write_text(document, encoding="utf-8")
    metadata["rendered_artifact"] = {"path": output_html.name, "sha256": sha256_file(output_html)}
    write_json(metadata, output_metadata)
    return metadata
