"""Evaluator-only R3/R4 temporal and recurring-routine diagnostics."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import _cosine, _intent_probe, _summary, assign_episode_intervals, load_episode_evaluation_inputs
from .io import write_json
from .runtime_metadata import collect_runtime_metadata


def cyclic_bin(value: float, edges: list[float], period: float) -> int:
    """Return a half-open cyclic bin, mapping the period boundary to zero."""
    values = np.asarray(edges, dtype=float)
    if len(values) < 2 or values[0] != 0 or values[-1] != period or not np.all(np.diff(values) > 0):
        raise ValueError("cyclic edges must be strictly increasing from zero through the period")
    wrapped = float(value) % period
    return min(int(np.searchsorted(values, wrapped, side="right") - 1), len(values) - 2)


def deterministic_user_split(user_ids: list[str] | np.ndarray, train_fraction: float, seed: int) -> np.ndarray:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}\0{u}".encode()).digest()[:8], "big") / 2**64 < train_fraction
        for u in map(str, user_ids)
    ])


def episode_duration_hours(episodes: pd.DataFrame) -> np.ndarray:
    durations = (episodes["end_time"] - episodes["start_time"]).dt.total_seconds().to_numpy() / 3600
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("episode durations must be finite and positive")
    return durations


def select_repeated_and_one_off(episodes: pd.DataFrame, repeated_min: int) -> pd.DataFrame:
    """Select recurrent routine episodes and user-local singleton non-routine episodes."""
    if repeated_min < 2:
        raise ValueError("repeated_min_occurrences must be at least two")
    result = episodes.copy()
    counts = result.groupby(["user_id", "primary_intent"])["episode_id"].transform("count")
    result["routine_class"] = np.where(
        (result["primary_intent"] == "routine") & (counts >= repeated_min), "repeated_routine",
        np.where((result["primary_intent"] != "routine") & (counts == 1), "one_off_episode", None),
    )
    return result[result["routine_class"].notna()].copy()


def periodic_retrieval(rows: pd.DataFrame, embeddings: np.ndarray) -> dict[str, Any]:
    """Nearest-neighbour user and periodic-state retrieval, excluding the query row."""
    if len(rows) < 2:
        return {"status": "insufficient_rows", "queries": 0, "user_top1": None, "state_top1": None}
    indices = rows["embedding_index"].to_numpy(dtype=int)
    matrix = embeddings[indices]
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    users = rows["user_id"].astype(str).to_numpy()
    states = rows["temporal_bin"].astype(str).to_numpy()
    return {"status": "ok", "queries": len(rows),
            "user_top1": float(np.mean(users[nearest] == users)),
            "state_top1": float(np.mean(states[nearest] == states))}


def _ridge_regression(rows: pd.DataFrame, embeddings: np.ndarray, target: str,
                      fraction: float, alpha: float, seed: int) -> dict[str, Any]:
    if rows.empty:
        return {"status": "empty", "rows": 0, "r2": None, "mae_hours": None}
    train = deterministic_user_split(rows["user_id"].tolist(), fraction, seed)
    if train.sum() < 2 or (~train).sum() < 2:
        return {"status": "insufficient_held_out_users", "rows": len(rows), "r2": None, "mae_hours": None}
    x = embeddings[rows["embedding_index"].to_numpy(dtype=int)]
    y = rows[target].to_numpy(dtype=float)
    mean, std = x[train].mean(0), x[train].std(0); std[std < 1e-8] = 1
    x = (x - mean) / std
    xt = x[train]
    weights = xt.T @ np.linalg.solve(xt @ xt.T + alpha * np.eye(len(xt)), y[train])
    prediction = x[~train] @ weights
    truth = y[~train]
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {"status": "ok", "rows": len(rows), "train_users": int(rows.loc[train, "user_id"].nunique()),
            "test_users": int(rows.loc[~train, "user_id"].nunique()),
            "r2": (1 - float(np.sum((truth-prediction)**2)) / denominator) if denominator else None,
            "mae_hours": float(np.mean(np.abs(truth-prediction)))}


def _geometry(rows: pd.DataFrame, embeddings: np.ndarray) -> dict[str, Any]:
    matrix = embeddings[rows["embedding_index"].to_numpy(dtype=int)] if len(rows) else embeddings[:0]
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    users = rows["user_id"].astype(str).to_numpy()
    pairs = [_cosine(normalized[i], normalized[j]) for i in range(len(rows)) for j in range(i + 1, len(rows)) if users[i] != users[j]]
    centered = normalized - normalized.mean(0, keepdims=True) if len(normalized) else normalized
    singular = np.linalg.svd(centered, compute_uv=False) if len(centered) else np.asarray([])
    energy = singular ** 2
    rank = float(np.exp(-np.sum((energy/energy.sum()) * np.log(np.maximum(energy/energy.sum(), 1e-15))))) if energy.sum() else 0.0
    return {"different_user_cosine": _summary(pairs), "effective_rank": rank,
            "effective_rank_ratio": rank / min(matrix.shape) if matrix.size else 0.0}


def evaluate_temporal_routine(truth_dir: str | Path, prepared_dir: str | Path,
                              dense_path: str | Path, output_path: str | Path,
                              config: dict[str, Any], *, kind: str) -> dict[str, Any]:
    started = time.perf_counter()
    dense, embeddings, episodes = load_episode_evaluation_inputs(dense_path, truth_dir)
    settings = config.get("evaluation", {}).get("temporal_routine", {})
    hour_edges = settings.get("hour_bin_edges", [])
    day_edges = settings.get("day_bin_edges", [])
    seed = int(settings.get("split_seed", 0)); fraction = float(settings.get("probe_train_fraction", config["evaluation"]["probe_train_fraction"]))
    assigned = assign_episode_intervals(dense, episodes)
    assigned["embedding_index"] = np.arange(len(assigned))
    assigned["hour_bin"] = [cyclic_bin(x.hour + x.minute / 60, hour_edges, 24) for x in assigned["timestamp"]]
    assigned["day_bin"] = [cyclic_bin(x.weekday(), day_edges, 7) for x in assigned["timestamp"]]
    assigned["temporal_bin"] = assigned["day_bin"].astype(str) + ":" + assigned["hour_bin"].astype(str)
    episode_lookup = episodes.set_index(["user_id", "episode_id"])
    labeled = assigned[assigned["episode_id"].notna()].copy()
    if not labeled.empty:
        keys = list(zip(labeled["user_id"], labeled["episode_id"]))
        starts = pd.DatetimeIndex([episode_lookup.loc[k, "start_time"] for k in keys])
        ends = pd.DatetimeIndex([episode_lookup.loc[k, "end_time"] for k in keys])
        labeled["episode_duration_hours"] = (ends-starts).total_seconds()/3600
        labeled["elapsed_episode_hours"] = (pd.DatetimeIndex(labeled["timestamp"])-starts).total_seconds()/3600
        labeled["remaining_episode_hours"] = (ends-pd.DatetimeIndex(labeled["timestamp"])).total_seconds()/3600
    selected_episodes = select_repeated_and_one_off(episodes, int(settings.get("repeated_min_occurrences", 2)))
    selected = labeled.merge(selected_episodes[["user_id", "episode_id", "routine_class"]], on=["user_id", "episode_id"], how="inner")
    class_rows = selected.copy()
    class_rows["primary_intent"] = class_rows["routine_class"]
    class_probe = _intent_probe(class_rows, embeddings, fraction, float(config["evaluation"]["ridge_alpha"])) if len(class_rows) else {"status": "empty_classes", "rows": 0}
    metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    coverage_rows = [{"user_id": str(r.user_id), "label": str(r.primary_intent) if r.primary_intent is not None else None,
                      "class": getattr(r, "routine_class", None), "temporal_bin": str(r.temporal_bin),
                      "history_event_count": int(r.history_event_count)} for r in selected.itertuples()]
    report = {
        "runtime_metadata": collect_runtime_metadata(duration_seconds=time.perf_counter() - started,
            seed=seed, device=None).to_dict(),
        "metric_contract": {"version": "temporal-routine/1.0", "kind": kind, "source_hashes": metadata["source_files"],
            "prepared_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "dense_users": sorted(dense.user_id.unique()), "dense_keys_sha256": hashlib.sha256("\n".join(f"{u}\0{t.isoformat()}" for u,t in zip(dense.user_id,dense.timestamp)).encode()).hexdigest(),
            "hour_bin_edges": hour_edges, "day_bin_edges": day_edges, "split_seed": seed, "probe_train_fraction": fraction},
        "coverage": {"dense_rows": len(dense), "dense_users": int(dense.user_id.nunique()), "labeled_rows": len(labeled),
            "label_coverage": len(labeled)/len(dense) if len(dense) else 0.0, "zero_history_rows": int((dense.history_event_count <= 0).sum()),
            "class_counts": selected["routine_class"].value_counts().sort_index().to_dict() if len(selected) else {}, "rows": coverage_rows},
        "cyclic_probes": {"hour": _intent_probe(assigned.assign(primary_intent=assigned.hour_bin.astype(str)), embeddings, fraction, float(config["evaluation"]["ridge_alpha"])),
            "day": _intent_probe(assigned.assign(primary_intent=assigned.day_bin.astype(str)), embeddings, fraction, float(config["evaluation"]["ridge_alpha"]))},
        "duration_tasks": {name: _ridge_regression(labeled, embeddings, name, fraction, float(config["evaluation"]["ridge_alpha"]), seed) for name in ("episode_duration_hours", "elapsed_episode_hours", "remaining_episode_hours")},
        "periodic_retrieval": periodic_retrieval(assigned, embeddings),
        "repeated_routine_vs_one_off": {"selection": "routine intent repeated per user versus singleton non-routine intent", "probe": class_probe},
        "collapse_diagnostics": _geometry(assigned, embeddings),
        "schedule_shift": {"status": "blocked", "reason": "Simulator audit found no controlled matched schedule-shift intervention; observational calendar labels are not a substitute."},
        "information_boundary": "Truth episode, intent, schedule, and duration labels are joined only inside evaluator-only code.",
    }
    write_json(report, output_path)
    return report
