from __future__ import annotations

import hashlib
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
            "R4_episode_coherence": {"status": "pending"},
            "R5_preference_opportunity_separation": {"status": "pending"},
            "R6_cross_service_alignment": {"status": "pending"},
            "R7_noise_and_sparsity_robustness": {"status": "pending"},
            "R8_geographic_temporal_generalization": {"status": "pending"},
            "R9_new_context_recommendation": {
                "status": "blocked_by_data_contract",
                "missing": "observable requests, impressions, availability, and candidate metadata",
            },
        },
    }
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
