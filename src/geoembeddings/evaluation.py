from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import write_json


LATENT_TRAITS = [
    "price_sensitivity",
    "distance_sensitivity",
    "novelty_seeking",
    "family_orientation",
    "travel_propensity",
    "time_flexibility",
    "transit_preference",
    "digital_engagement",
]


def load_episode_evaluation_inputs(
    dense_path: str | Path, truth_dir: str | Path
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load public dense rows and protected episodes at the evaluator boundary."""
    truth_dir = Path(truth_dir).resolve()
    if truth_dir.name != "truth":
        raise ValueError("truth_dir must point directly to the simulator's truth/ directory")
    episode_path = truth_dir / "episodes_truth.csv.gz"
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing evaluator-only file: {episode_path}")
    with np.load(dense_path, allow_pickle=False) as payload:
        required = {"user_id", "timestamp", "cutoff_kind", "embedding", "history_event_count"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Dense export is missing required arrays: {sorted(missing)}")
        n = len(payload["user_id"])
        for key in required - {"embedding"}:
            if len(payload[key]) != n:
                raise ValueError(f"Dense array {key!r} is not row-aligned")
        embeddings = np.asarray(payload["embedding"], dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[0] != n or embeddings.shape[1] < 1:
            raise ValueError("Dense embedding must be a row-aligned, non-empty 2-D array")
        if not np.isfinite(embeddings).all():
            raise ValueError("Dense embeddings contain non-finite values")
        dense = pd.DataFrame({"user_id": payload["user_id"].astype(str),
                              "timestamp": pd.to_datetime(payload["timestamp"].astype(str), utc=True),
                              "history_event_count": payload["history_event_count"].astype(np.int64)})
    if dense[["user_id", "timestamp"]].duplicated().any():
        raise ValueError("Dense export contains duplicate user/timestamp records")
    for user_id, group in dense.groupby("user_id", sort=False):
        if not group["timestamp"].is_monotonic_increasing:
            raise ValueError(f"Dense timestamps are not monotonic for user {user_id}")
    episodes = pd.read_csv(episode_path)
    required_episode = {"user_id", "episode_id", "start_time", "end_time", "primary_intent"}
    missing_episode = required_episode.difference(episodes.columns)
    if missing_episode:
        raise ValueError(f"Episode truth is missing columns: {sorted(missing_episode)}")
    episodes = episodes.copy()
    episodes["user_id"] = episodes["user_id"].astype(str)
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], utc=True)
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], utc=True)
    if episodes[["user_id", "episode_id"]].duplicated().any():
        raise ValueError("Episode truth contains duplicate user/episode records")
    if (episodes["end_time"] <= episodes["start_time"]).any():
        raise ValueError("Episode intervals must have end_time after start_time")
    for user_id, group in episodes.sort_values(["user_id", "start_time"]).groupby("user_id"):
        if (group["start_time"].iloc[1:].to_numpy() < group["end_time"].iloc[:-1].to_numpy()).any():
            raise ValueError(f"Episode intervals overlap for user {user_id}")
    return dense, embeddings, episodes


def assign_episode_intervals(dense: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Assign timestamps to non-overlapping half-open [start, end) intervals."""
    result = dense.copy()
    result["episode_id"] = None
    result["primary_intent"] = None
    for user_id, positions in result.groupby("user_id").groups.items():
        candidates = episodes[episodes["user_id"] == user_id]
        for episode in candidates.itertuples(index=False):
            mask = (result.index.isin(positions) & (result["timestamp"] >= episode.start_time)
                    & (result["timestamp"] < episode.end_time))
            result.loc[mask, "episode_id"] = episode.episode_id
            result.loc[mask, "primary_intent"] = episode.primary_intent
    return result


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def evaluate_episode_response(
    truth_dir: str | Path, prepared_dir: str | Path, dense_path: str | Path,
    output_path: str | Path, config: dict[str, Any], *, kind: str,
) -> dict[str, Any]:
    dense, embeddings, episodes = load_episode_evaluation_inputs(dense_path, truth_dir)
    settings = config.get("evaluation", {}).get("episode_response", {})
    edges = np.asarray(settings.get("boundary_bin_edges_hours", []), dtype=float)
    if edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("evaluation.episode_response.boundary_bin_edges_hours must be finite and strictly increasing")
    assigned = assign_episode_intervals(dense, episodes)
    assigned["embedding_index"] = np.arange(len(assigned))
    same, adjacent, changes = [], [], []
    for _, group in assigned.groupby("user_id", sort=False):
        rows = list(group.itertuples())
        for left, right in zip(rows, rows[1:]):
            sim = _cosine(embeddings[left.embedding_index], embeddings[right.embedding_index])
            if left.episode_id is not None and left.episode_id == right.episode_id:
                same.append(sim)
            elif left.episode_id is not None and right.episode_id is not None:
                adjacent.append(sim)
                changes.append(1.0 - sim)
    curves = [[] for _ in range(len(edges) - 1)]
    boundary_count = 0
    drift, recovery = [], []
    for episode in episodes.itertuples(index=False):
        user = assigned[assigned["user_id"] == episode.user_id]
        if user.empty:
            continue
        boundary_count += 1
        delta = (user["timestamp"] - episode.start_time).dt.total_seconds().to_numpy() / 3600
        indices = user["embedding_index"].to_numpy(dtype=int)
        before = np.where(delta < 0)[0]
        if not len(before):
            continue
        anchor = embeddings[indices[before[-1]]]
        for pos, hours in enumerate(delta):
            bin_index = int(np.searchsorted(edges, hours, side="right") - 1)
            if 0 <= bin_index < len(curves):
                curves[bin_index].append(1.0 - _cosine(anchor, embeddings[indices[pos]]))
        inside = np.where((user["timestamp"] >= episode.start_time).to_numpy() &
                          (user["timestamp"] < episode.end_time).to_numpy())[0]
        after = np.where((user["timestamp"] >= episode.end_time).to_numpy())[0]
        if len(inside):
            drift.append(1.0 - _cosine(anchor, embeddings[indices[inside[0]]]))
        if len(after):
            recovery.append(_cosine(anchor, embeddings[indices[after[0]]]))
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    centered = normalized - normalized.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular ** 2
    effective_rank = float(np.exp(-np.sum((eigen/eigen.sum()) * np.log(np.maximum(eigen/eigen.sum(), 1e-15))))) if eigen.sum() else 0.0
    user_means = {u: normalized[np.asarray(list(ix), dtype=int)].mean(0) for u, ix in dense.groupby("user_id").groups.items()}
    users = sorted(user_means)
    separation = [_cosine(user_means[a], user_means[b]) for i, a in enumerate(users) for b in users[i+1:]]
    metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    source_hashes = json.loads(metadata_path.read_text())["source_files"]
    bins = [{"left_hours": float(edges[i]), "right_hours": float(edges[i+1]), "points": len(v),
             "mean_cosine_drift_from_pre_start": float(np.mean(v)) if v else None} for i, v in enumerate(curves)]
    report = {
        "metric_contract": {"version": "episode-response/1.0", "kind": kind,
            "interval_semantics": "start_time <= timestamp < end_time", "boundary_bin_edges_hours": edges.tolist(),
            "prepared_metadata_sha256": metadata_hash, "source_hashes": source_hashes,
            "dense_users": sorted(dense["user_id"].unique()), "dense_timestamps_sha256": hashlib.sha256("\n".join(dense["timestamp"].astype(str)).encode()).hexdigest(),
            "embedding_dim": int(embeddings.shape[1])},
        "coverage": {"dense_users": int(dense["user_id"].nunique()), "truth_users": int(episodes["user_id"].nunique()),
            "missing_dense_users": sorted(set(episodes["user_id"]) - set(dense["user_id"])), "episodes": len(episodes),
            "episodes_with_dense_rows": int(assigned["episode_id"].nunique()), "start_boundaries_with_user_history": boundary_count,
            "populated_time_bins": sum(bool(v) for v in curves), "total_time_bins": len(curves)},
        "R4_episode_coherence": {"within_episode_consecutive_cosine": _summary(same),
            "adjacent_episode_consecutive_cosine": _summary(adjacent), "boundary_change_magnitude": _summary(changes),
            "start_response_curve": bins},
        "R1_single_vector_diagnostics": {"temporary_episode_drift": _summary(drift), "post_episode_recovery_cosine": _summary(recovery),
            "limitation": "Single-vector drift and recovery do not establish persistent/context disentanglement."},
        "collapse_diagnostics": {"different_user_mean_cosine": float(np.mean(separation)) if separation else None,
            "different_user_pairs": len(separation), "effective_rank": effective_rank},
        "intent_probe": _intent_probe(assigned, embeddings, float(config["evaluation"]["probe_train_fraction"]), float(config["evaluation"]["ridge_alpha"])),
        "information_boundary": "episode_id, primary_intent, and all episode truth remain evaluator-only and occur only in this report",
    }
    write_json(report, output_path)
    return report


def _summary(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None}


def _intent_probe(rows: pd.DataFrame, embeddings: np.ndarray, fraction: float, alpha: float) -> dict[str, Any]:
    labeled = rows[rows["primary_intent"].notna()]
    if labeled.empty:
        return {"status": "insufficient_rows", "rows": 0}
    users = labeled["user_id"].astype(str).to_numpy()
    train = np.asarray([_stable_fraction(u) < fraction for u in users])
    labels = labeled["primary_intent"].astype(str).to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    if train.sum() < 2 or (~train).sum() < 2 or len(classes) < 2:
        return {"status": "insufficient_held_out_users", "rows": len(labeled), "class_counts": {str(c): int(n) for c, n in zip(classes, counts)}}
    x = embeddings[labeled["embedding_index"].to_numpy(dtype=int)]
    x_mean, x_std = x[train].mean(0), x[train].std(0); x_std[x_std < 1e-8] = 1
    x = (x - x_mean) / x_std
    y = np.stack([(labels == c).astype(float) for c in classes], 1)
    x_train = x[train]
    if x_train.shape[0] < x_train.shape[1]:
        # Statistical histograms can have many more columns than labeled rows.
        # The dual ridge form is equivalent, but avoids constructing and solving
        # a potentially multi-gigabyte feature-by-feature system.
        dual = np.linalg.solve(
            x_train @ x_train.T + alpha * np.eye(x_train.shape[0]), y[train]
        )
        weights = x_train.T @ dual
    else:
        weights = np.linalg.solve(
            x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
            x_train.T @ y[train],
        )
    prediction = classes[np.argmax(x[~train] @ weights, axis=1)]
    truth = labels[~train]
    majority = classes[np.argmax([(labels[train] == c).sum() for c in classes])]
    recalls = [float(np.mean(prediction[truth == c] == c)) for c in classes if np.any(truth == c)]
    f1s = []
    for c in classes:
        tp = np.sum((prediction == c) & (truth == c)); fp = np.sum((prediction == c) & (truth != c)); fn = np.sum((prediction != c) & (truth == c))
        f1s.append(float(2*tp / max(2*tp+fp+fn, 1)))
    return {"status": "ok", "split": "stable held-out users", "train_users": len(set(users[train])), "test_users": len(set(users[~train])),
            "class_counts": {str(c): int(n) for c, n in zip(classes, counts)}, "accuracy": float(np.mean(prediction == truth)),
            "majority_baseline_accuracy": float(np.mean(truth == majority)), "macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(recalls))}


def evaluate_embeddings(
    observed_dir: str | Path,
    truth_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_path: str | Path | None,
    embeddings_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth_dir = Path(truth_dir)
    if truth_dir.name != "truth":
        raise ValueError("--truth-dir must point directly to the simulator's truth/ directory")
    latent_path = truth_dir / "user_latents.csv.gz"
    if not latent_path.is_file():
        raise FileNotFoundError(f"Missing evaluator-only file: {latent_path}")

    next_event: dict[str, Any]
    if checkpoint_path is None:
        next_event = {"status": "not_applicable_for_non_learned_baseline"}
    else:
        from .training import evaluate_next_event

        next_event = evaluate_next_event(
            observed_dir, prepared_dir, checkpoint_path, config
        )
    payload = np.load(embeddings_path, allow_pickle=False)
    user_ids = payload["user_id"].astype(str)
    cutoffs = payload["cutoff"].astype(str)
    embeddings = payload["embedding"].astype(np.float64)
    latent = pd.read_csv(latent_path)

    final_mask = cutoffs == "test"
    final_users = user_ids[final_mask]
    final_embeddings = embeddings[final_mask]
    probe_report = _latent_probe(
        final_users,
        final_embeddings,
        latent,
        float(config["evaluation"]["probe_train_fraction"]),
        float(config["evaluation"]["ridge_alpha"]),
    )
    stability = _cutoff_stability(user_ids, cutoffs, embeddings)

    learned_model = checkpoint_path is not None
    report = {
        "report_scope": {
            "name": "base_three_cutoff_evaluation",
            "description": "Persistent probes, cross-cutoff stability, and learned next-event metrics only.",
            "supplemental_reports": {
                "R4_episode_coherence": {
                    "command": "evaluate --episodes",
                    "artifacts": ["episode_response.json", "baseline_episode_response.json"],
                },
                "R7_noise_and_sparsity_robustness": {
                    "command": "robustness",
                    "artifacts": [
                        "robustness/learned_robustness.json",
                        "robustness/baseline_robustness.json",
                    ],
                },
            },
        },
        "information_boundary": {
            "train": "observed/ only",
            "evaluation": "truth/ opened only by this evaluator command",
        },
        "next_event": next_event,
        "persistent_trait_probes": probe_report,
        "cross_cutoff_stability": stability,
        "requirements": {
            "R1_persistent_context_separation": {
                "status": "partial",
                "evidence": ["persistent_trait_probes", "cross_cutoff_stability"],
                "missing": "episode-conditioned contextual embeddings",
            },
            "R2_multiscale_spatial_fidelity": {
                "status": "partial" if learned_model else "pending",
                "evidence": (
                    ["next_event.next_geohash_5_accuracy", "next_event.next_geohash_7_accuracy"]
                    if learned_model
                    else []
                ),
                "missing": "distance-aware and boundary-robust retrieval tests",
            },
            "R3_multiscale_temporal_fidelity": {
                "status": "partial" if learned_model else "pending",
                "evidence": ["next_event"] if learned_model else [],
                "missing": "routine and periodicity probes",
            },
            "R4_episode_coherence": {
                "status": "not_evaluated_in_base_report",
                "supplemental_status": "partial",
                "evidence": "evaluate --episodes writes the episode-response supplemental report",
                "missing": "factorized persistent/context separation and matched change scenarios",
            },
            "R5_preference_opportunity_separation": {"status": "pending"},
            "R6_cross_service_alignment": {"status": "pending"},
            "R7_noise_and_sparsity_robustness": {
                "status": "not_evaluated_in_base_report",
                "supplemental_status": "partial",
                "evidence": "robustness writes deterministic GPS, timestamp, service-removal, and truncation views",
                "missing": "real-noise calibration and causal invariance tests",
            },
            "R8_geographic_temporal_generalization": {"status": "pending"},
            "R9_new_context_recommendation": {
                "status": "blocked_by_data_contract",
                "missing": "observable requests, impressions, availability, and candidate metadata",
            },
        },
    }
    write_json(report, output_path)
    return report


def evaluate_event_removal(
    truth_dir: str | Path, original_embeddings_path: str | Path,
    export_manifest: dict[str, Any], output_path: str | Path, config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate matched clean/corrupted rows; truth opens only here for frozen probes."""
    latent_path = Path(truth_dir) / "user_latents.csv.gz"
    if Path(truth_dir).name != "truth" or not latent_path.is_file():
        raise ValueError("robustness evaluation requires the canonical truth/ boundary")
    original = np.load(original_embeddings_path, allow_pickle=False)
    original_map = {(str(u), str(c)): e.astype(np.float64) for u, c, e in
                    zip(original["user_id"], original["cutoff"], original["embedding"])}
    latent = pd.read_csv(latent_path)
    original_test = [(key, value) for key, value in original_map.items() if key[1] == "test"]
    original_probe = _latent_probe(np.asarray([k[0] for k, _ in original_test]),
        np.stack([v for _, v in original_test]), latent,
        float(config["evaluation"]["probe_train_fraction"]), float(config["evaluation"]["ridge_alpha"]))
    rates = []
    for artifact in export_manifest["artifacts"]:
        if artifact["path"] is None:
            rates.append({**artifact, "matched_rows": 0, "coverage": 0.0,
                          "cosine_drift": _summary([]), "probe": {"status": "unencodable"},
                          "probe_mean_r2_degradation": None})
            continue
        payload = np.load(artifact["path"], allow_pickle=False)
        thinned = {(str(u), str(c)): e.astype(np.float64) for u, c, e in
                   zip(payload["user_id"], payload["cutoff"], payload["embedding"])}
        keys = sorted(set(original_map) & set(thinned))
        drifts = [1.0 - float(np.dot(original_map[k], thinned[k]) /
                  max(np.linalg.norm(original_map[k]) * np.linalg.norm(thinned[k]), 1e-12)) for k in keys]
        tests = [k for k in keys if k[1] == "test"]
        matched_original_probe = (_latent_probe(np.asarray([k[0] for k in tests]),
                    np.stack([original_map[k] for k in tests]), latent,
                    float(config["evaluation"]["probe_train_fraction"]),
                    float(config["evaluation"]["ridge_alpha"])) if tests else {"status": "unencodable"})
        probe = (_latent_probe(np.asarray([k[0] for k in tests]), np.stack([thinned[k] for k in tests]),
                    latent, float(config["evaluation"]["probe_train_fraction"]),
                    float(config["evaluation"]["ridge_alpha"])) if tests else {"status": "unencodable"})
        base_r2, thin_r2 = matched_original_probe.get("mean_r2"), probe.get("mean_r2")
        rates.append({**artifact, "matched_rows": len(keys),
            "coverage": len(keys) / max(len(original_map), 1), "cosine_drift": _summary(drifts),
            "matched_unmodified_probe": matched_original_probe, "probe": probe, "probe_mean_r2_degradation":
                (base_r2 - thin_r2 if base_r2 is not None and thin_r2 is not None else None)})
    report = {"metric_contract": {"version": "robustness-metrics/2.0",
        "source_hashes": export_manifest["source_hashes"], "algorithm": export_manifest["algorithm"],
        "seed": export_manifest["seed"], "kind": export_manifest["kind"],
        "field_order": export_manifest["field_order"],
        "specification_hash": export_manifest.get("specification_hash"),
        "requested_views": export_manifest.get("requested_views"),
        "view_ids": [a.get("view_id") for a in export_manifest["artifacts"]],
        "mask_hashes": [a.get("mask_hash") for a in export_manifest["artifacts"]]},
        "unmodified_rows": len(original_map), "unmodified_probe": original_probe, "views": rates,
        "R6_axes": ["leave-one-service-out_cosine_drift", "frozen_probe_degradation", "coverage"],
        "R7_axes": ["gps_and_timestamp_drift", "recent_truncation", "coverage", "frozen_probe_degradation"],
        "limitations": "Deterministic corruptions are sensitivity analyses, not evidence of real-world noise or causal invariance.",
        "information_boundary": "truth/ is opened only by this evaluator; masks and encoders are observed-only"}
    write_json(report, output_path)
    return report


def _latent_probe(
    embedding_users: np.ndarray,
    embeddings: np.ndarray,
    latent: pd.DataFrame,
    train_fraction: float,
    ridge_alpha: float,
) -> dict[str, Any]:
    lookup = {str(user_id): index for index, user_id in enumerate(embedding_users)}
    shared_users = [str(user_id) for user_id in latent["user_id"] if str(user_id) in lookup]
    if len(shared_users) < 10:
        return {"status": "insufficient_users", "users": len(shared_users)}
    x = np.stack([embeddings[lookup[user_id]] for user_id in shared_users])
    y_frame = latent.set_index("user_id").loc[shared_users]
    train_mask = np.asarray([_stable_fraction(user_id) < train_fraction for user_id in shared_users])
    if train_mask.sum() < 2 or (~train_mask).sum() < 2:
        raise ValueError("Probe split produced fewer than two users in train or test")
    x_train, x_test = x[train_mask], x[~train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    scores: dict[str, float] = {}
    for trait in LATENT_TRAITS:
        if trait not in y_frame.columns:
            continue
        y = y_frame[trait].to_numpy(dtype=np.float64)
        y_train, y_test = y[train_mask], y[~train_mask]
        target_mean = float(y_train.mean())
        centered_target = y_train - target_mean
        if x_train.shape[1] > x_train.shape[0]:
            dual = np.linalg.solve(
                x_train @ x_train.T + ridge_alpha * np.eye(x_train.shape[0]),
                centered_target,
            )
            prediction = target_mean + x_test @ x_train.T @ dual
        else:
            weights = np.linalg.solve(
                x_train.T @ x_train + ridge_alpha * np.eye(x_train.shape[1]),
                x_train.T @ centered_target,
            )
            prediction = target_mean + x_test @ weights
        denominator = float(np.square(y_test - y_test.mean()).sum())
        r2 = 1.0 - float(np.square(y_test - prediction).sum()) / max(denominator, 1e-12)
        scores[trait] = r2
    return {
        "status": "ok",
        "train_users": int(train_mask.sum()),
        "test_users": int((~train_mask).sum()),
        "metric": "held-out-user ridge-probe R2",
        "r2": scores,
        "mean_r2": float(np.mean(list(scores.values()))) if scores else None,
    }


def _cutoff_stability(
    user_ids: np.ndarray,
    cutoffs: np.ndarray,
    embeddings: np.ndarray,
) -> dict[str, Any]:
    by_key = {(user_id, cutoff): embeddings[index] for index, (user_id, cutoff) in enumerate(zip(user_ids, cutoffs))}
    users = sorted(set(user_ids))
    comparisons: dict[str, list[float]] = {"train_to_validation": [], "validation_to_test": [], "train_to_test": []}
    pairs = {
        "train_to_validation": ("train", "validation"),
        "validation_to_test": ("validation", "test"),
        "train_to_test": ("train", "test"),
    }
    for user_id in users:
        for name, (left, right) in pairs.items():
            if (user_id, left) not in by_key or (user_id, right) not in by_key:
                continue
            a, b = by_key[(user_id, left)], by_key[(user_id, right)]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            comparisons[name].append(float(np.dot(a, b) / max(denominator, 1e-12)))
    return {
        name: {
            "users": len(values),
            "mean_cosine_similarity": float(np.mean(values)) if values else None,
            "p10": float(np.quantile(values, 0.10)) if values else None,
            "p90": float(np.quantile(values, 0.90)) if values else None,
        }
        for name, values in comparisons.items()
    }


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)
