"""Seeded representation reliability diagnostics (R10)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .contract import RELIABILITY_REPORT_SCHEMA
from .io import sha256_file, write_json
from .runtime_metadata import collect_runtime_metadata


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(1.0 - np.dot(a, b) / denominator)


def load_reliability_inputs(observed_dir: str | Path, prepared_dir: str | Path,
                            embeddings_path: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Validate observed/preparation lineage and the frozen export."""
    observed_dir, prepared_dir = Path(observed_dir).resolve(), Path(prepared_dir).resolve()
    metadata_path = prepared_dir / "prepared_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = {name: sha256_file(observed_dir / name) for name in metadata["source_files"]}
    if actual != metadata["source_files"]:
        raise ValueError("Observed source hashes do not match prepared metadata")
    with np.load(embeddings_path, allow_pickle=False) as payload:
        required = {"user_id", "cutoff", "embedding"}
        if not required.issubset(payload.files):
            raise ValueError(f"Embedding export is missing arrays: {sorted(required - set(payload.files))}")
        users, cutoffs = payload["user_id"].astype(str), payload["cutoff"].astype(str)
        values = np.asarray(payload["embedding"], dtype=np.float64)
    if values.ndim != 2 or len(users) != len(cutoffs) or len(users) != len(values) or values.shape[1] < 1:
        raise ValueError("Embedding arrays must be row-aligned with a non-empty 2-D embedding")
    if len(users) == 0 or not np.isfinite(values).all():
        raise ValueError("Embedding export must be non-empty and finite")
    keys = np.char.add(np.char.add(users, "\0"), cutoffs)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Embedding export contains duplicate user/cutoff rows")
    return metadata, users, cutoffs, values


def resampling_statistics(values: np.ndarray, *, seed: int, resamples: int) -> dict[str, float]:
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    if len(values) < 2:
        raise ValueError("at least two observations are required for resampling")
    rng = np.random.default_rng(seed)
    centroids = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    variance = float(np.mean(np.sum((centroids - centroids.mean(axis=0)) ** 2, axis=1)))
    error = float(np.mean([_cosine_distance(row, values.mean(axis=0)) for row in values]))
    return {"embedding_variance": variance, "realized_error": error}


def validate_preparation_config(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    data = config.get("data")
    if data is None:
        return
    categorical = list(data.get("categorical_fields", []))
    if bool(data.get("include_object_id", False)): categorical.append("object_id")
    if categorical != metadata["categorical_fields"] or list(data.get("continuous_fields", [])) != metadata["continuous_fields"]:
        raise ValueError("Resolved configuration does not match preparation field identity")


def calibration_bins(uncertainty: np.ndarray, error: np.ndarray, *, bins: int,
                     minimum_count: int) -> list[dict[str, Any]]:
    if bins < 1 or minimum_count < 1:
        raise ValueError("bins and minimum_count must be positive")
    if len(uncertainty) != len(error) or not np.isfinite(uncertainty).all() or not np.isfinite(error).all():
        raise ValueError("calibration inputs must be aligned and finite")
    edges = np.linspace(0, len(uncertainty), bins + 1, dtype=int)
    order = np.argsort(uncertainty, kind="stable")
    result = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = order[left:right]
        count = int(len(selected)); sufficient = count >= minimum_count
        result.append({"bin": index, "count": count, "status": "ok" if sufficient else "insufficient_coverage",
                       "mean_uncertainty": float(np.mean(uncertainty[selected])) if count else None,
                       "mean_error": float(np.mean(error[selected])) if count else None})
    return result


def coverage_risk_curve(uncertainty: np.ndarray, error: np.ndarray,
                        coverages: list[float], *, minimum_count: int) -> list[dict[str, Any]]:
    order = np.argsort(uncertainty, kind="stable")
    result = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage levels must be in (0, 1]")
        count = min(len(order), int(np.ceil(len(order) * coverage)))
        selected = order[:count]; sufficient = count >= minimum_count
        result.append({"requested_coverage": float(coverage), "count": count,
                       "realized_coverage": float(count / len(order)) if len(order) else 0.0,
                       "status": "ok" if sufficient else "insufficient_coverage",
                       "risk": float(np.mean(error[selected])) if sufficient else None})
    return result


def validate_reliability_report(report: dict[str, Any]) -> None:
    required = {"schema_version", "runtime_metadata", "seed", "representation_kind", "source_hashes",
                "preparation_identity", "cutoff_identity", "resampling", "coverage", "metrics"}
    if report.get("schema_version") != RELIABILITY_REPORT_SCHEMA or not required.issubset(report):
        raise ValueError("Invalid reliability report schema")
    def walk(value: Any) -> None:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Reliability report contains a non-finite metric")
        if isinstance(value, dict):
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(report)


def evaluate_reliability(observed_dir: str | Path, prepared_dir: str | Path,
                         embeddings_path: str | Path, output_path: str | Path,
                         config: dict[str, Any], *, kind: str, overwrite: bool = False) -> dict[str, Any]:
    started = time.perf_counter(); output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing reliability report: {output_path}")
    settings = config.get("evaluation", {}).get("reliability", {})
    seed = int(settings.get("seed", config.get("seed", 0)))
    resamples = int(settings.get("resamples", 200)); bins = int(settings.get("calibration_bins", 5))
    minimum = int(settings.get("minimum_bin_count", 5))
    coverages = [float(x) for x in settings.get("coverage_levels", [0.25, 0.5, 0.75, 1.0])]
    metadata, users, cutoffs, values = load_reliability_inputs(observed_dir, prepared_dir, embeddings_path)
    validate_preparation_config(metadata, config)
    rows = []
    for user_index, user in enumerate(sorted(set(users))):
        selected = values[users == user]
        if len(selected) >= 2:
            stats = resampling_statistics(selected, seed=seed + user_index, resamples=resamples)
            rows.append({"user_id": user, "sample_count": len(selected), **stats})
    uncertainty = np.asarray([r["embedding_variance"] for r in rows]); error = np.asarray([r["realized_error"] for r in rows])
    metric_status = "ok" if len(rows) >= minimum else "insufficient_coverage"
    report = {"schema_version": RELIABILITY_REPORT_SCHEMA,
        "runtime_metadata": collect_runtime_metadata(duration_seconds=time.perf_counter()-started, seed=seed, device=None).to_dict(),
        "seed": seed, "representation_kind": kind, "source_hashes": metadata["source_files"],
        "preparation_identity": {"prepared_metadata_sha256": sha256_file(Path(prepared_dir)/"prepared_metadata.json"),
            "categorical_fields": metadata["categorical_fields"], "continuous_fields": metadata["continuous_fields"]},
        "cutoff_identity": {"train_end": metadata["train_end"], "validation_end": metadata["validation_end"],
            "cutoffs": sorted(set(cutoffs)), "user_cutoff_sha256": hashlib.sha256("\n".join(sorted(np.char.add(np.char.add(users, "\0"), cutoffs))).encode()).hexdigest()},
        "resampling": {"method": "seeded cutoff bootstrap", "resamples": resamples, "seed_derivation": "seed + sorted-user-index"},
        "coverage": {"export_rows": len(users), "export_users": len(set(users)), "evaluated_users": len(rows),
            "insufficient_users": sorted(set(users) - {r["user_id"] for r in rows}), "minimum_bin_count": minimum},
        "metrics": {"status": metric_status, "user_statistics": rows,
            "reliability_error_bins": calibration_bins(uncertainty, error, bins=bins, minimum_count=minimum),
            "coverage_risk": coverage_risk_curve(uncertainty, error, coverages, minimum_count=minimum)}}
    validate_reliability_report(report); write_json(report, output_path); return report
