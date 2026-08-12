"""Authenticated protected ranking evaluation for matched simulator pairs (T3.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contract import OBSERVED_FILES, PairManifest
from .io import read_json, sha256_file, write_json
from .layout import PairLayout
from .pair_integrity import require_passing_pair_integrity
from .ranking import RANKING_PREDICTION_SCHEMA, RANKING_REPORT_SCHEMA

RANKING_PAIR_SCHEMA = "geoembeddings-ranking-counterfactual/1.0"
TRUTH_NAME = "recommendation_counterfactuals.csv.gz"


def _load_authenticated_predictions(prediction: Path, report_path: Path, side: Any) -> tuple[dict[str, list[str]], dict[str, Any]]:
    report = read_json(report_path)
    if report.get("schema_version") != RANKING_REPORT_SCHEMA or report.get("prediction_schema_version") != RANKING_PREDICTION_SCHEMA:
        raise ValueError("unsupported ranking prediction/report schema")
    reverse = {filename: key for key, filename in OBSERVED_FILES.items()}
    expected_sources = {reverse[Path(name).name]: digest for name, digest in side.source_hashes.items()
                        if name.startswith("observed/") and Path(name).name in reverse}
    if report.get("source_hashes") != expected_sources:
        raise ValueError("ranking prediction source hashes do not match pair identity")
    with np.load(prediction, allow_pickle=False) as data:
        if str(data["schema_version"].item()) != RANKING_PREDICTION_SCHEMA:
            raise ValueError("unsupported ranking prediction schema")
        for field in ("request_id", "poi_id", "rank", "score"):
            if field not in data:
                raise ValueError(f"ranking prediction field missing: {field}")
        if str(data["request_hash"].item()) != report.get("request_hash") or str(data["candidate_hash"].item()) != report.get("candidate_hash"):
            raise ValueError("ranking prediction identities do not match its authenticated report")
        rows = list(zip(data["request_id"].astype(str), data["poi_id"].astype(str), data["rank"].astype(int), strict=True))
    if len({(r, p) for r, p, _ in rows}) != len(rows):
        raise ValueError("duplicate ranking request/candidate identity")
    grouped: dict[str, list[tuple[int, str]]] = {}
    for request, poi, rank in rows:
        grouped.setdefault(request, []).append((int(rank), poi))
    ordered = {request: [poi for rank, poi in sorted(values)] for request, values in grouped.items()}
    return ordered, report


def protected_ranking_metrics(predictions: Mapping[str, Sequence[str]], truth: pd.DataFrame) -> dict[str, Any]:
    required = {"request_id", "poi_id", "utility", "choice_probability"}
    if set(truth.columns) != required:
        raise ValueError(f"protected ranking truth schema must be exactly {sorted(required)}")
    if truth.duplicated(["request_id", "poi_id"]).any():
        raise ValueError("duplicate protected request/candidate identity")
    expected = {(str(r.request_id), str(r.poi_id)) for r in truth.itertuples()}
    actual = {(request, poi) for request, pois in predictions.items() for poi in pois}
    if actual != expected:
        raise ValueError("ranking predictions and protected request/candidate identities mismatch")
    regrets, recoveries = [], []
    for request, group in truth.groupby("request_id", sort=True):
        order = list(predictions[str(request)])
        indexed = group.set_index("poi_id")
        utilities = indexed["utility"].astype(float)
        probabilities = indexed["choice_probability"].astype(float)
        if not np.isfinite(utilities).all() or not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 1e-6:
            raise ValueError("protected utilities/probabilities must be finite and probabilities sum to one")
        regrets.append(float(utilities.max() - utilities.loc[order[0]]))
        # Compare normalized softmax prediction scores represented by reciprocal rank.
        predicted = np.asarray([1 / (order.index(poi) + 1) for poi in indexed.index], dtype=float)
        predicted /= predicted.sum()
        recoveries.append(float(np.mean(np.square(predicted - probabilities.to_numpy()))))
    return {"requests": len(regrets), "mean_utility_regret_at_1": float(np.mean(regrets)) if regrets else 0.0,
            "choice_probability_brier": float(np.mean(recoveries)) if recoveries else 0.0}


def evaluate_ranking_pair(pair_manifest_path: str | Path, prediction_paths: Sequence[Path],
                          report_paths: Sequence[Path], output: Path, *, overwrite: bool = False,
                          sensitivity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read protected truth only after the current passing pair-integrity gate."""
    integrity = require_passing_pair_integrity(pair_manifest_path)
    if len(prediction_paths) != 2 or len(report_paths) != 2:
        raise ValueError("reference and intervention ranking artifacts are required")
    if output.exists() and not overwrite:
        raise FileExistsError("protected ranking report exists; use --overwrite")
    layout = PairLayout.from_manifest_path(pair_manifest_path)
    pair = PairManifest.from_dict(read_json(layout.manifest))
    results, reports = {}, []
    for label, side, prediction, ranking_report in zip(("reference", "intervention"),
            (pair.reference, pair.intervention), prediction_paths, report_paths, strict=True):
        ranked, authenticated = _load_authenticated_predictions(prediction, ranking_report, side)
        truth_path = Path(side.run_dir) / "truth" / TRUTH_NAME
        if not truth_path.is_file():
            raise FileNotFoundError(f"missing protected recommendation truth: {truth_path}")
        results[label] = protected_ranking_metrics(ranked, pd.read_csv(truth_path, keep_default_na=False))
        reports.append(authenticated)
    if reports[0].get("model") != reports[1].get("model"):
        raise ValueError("paired ranking prediction models mismatch")
    report = {"schema_version": RANKING_PAIR_SCHEMA,
        "integrity": {"status": integrity["status"], "pair_manifest_sha256": sha256_file(layout.manifest),
                      "pair_integrity_sha256": sha256_file(layout.integrity_report)},
        "model": reports[0]["model"], "observed_ranking_metrics": {
            side: reports[i]["metrics"] for i, side in enumerate(("reference", "intervention"))},
        "protected_counterfactual_metrics": results,
        "coverage": {side: {"observed": reports[i]["coverage"], "protected_requests": results[side]["requests"]}
                     for i, side in enumerate(("reference", "intervention"))},
        "sensitivity_diagnostics": dict(sensitivity or {}),
        "controls_required": ["popularity", "nearest", "category_preference", "frozen_embedding"],
        "limitations": ["Metrics identify behavior only under the authenticated synthetic simulator and logging-policy assumptions.",
            "Position-based propensity adjustment is not real-world causal evidence and may remain confounded by unobserved exposure assignment."],
        "information_boundary": "Training artifacts authenticate observed files only; protected utility and choice probability are opened after validate-pair succeeds."}
    write_json(report, output)
    return report
