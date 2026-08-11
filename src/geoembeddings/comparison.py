from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .schema import load_observed


PERSISTENT_TRAITS = [
    "price_sensitivity",
    "distance_sensitivity",
    "novelty_seeking",
    "family_orientation",
    "travel_propensity",
    "time_flexibility",
    "transit_preference",
    "digital_engagement",
]
PREFERENCE_TRAITS = [
    "pref_grocery",
    "pref_restaurant",
    "pref_cafe",
    "pref_mall",
    "pref_park",
    "pref_onsen",
]
TARGET_FIELDS = {
    "next_service": "service_id",
    "next_action": "action_type",
    "next_category": "object_category",
    "next_region": "region_id",
    "next_geohash_5": "geohash_5",
    "next_geohash_7": "geohash_7",
}


def compare_embeddings(
    observed_dir: str | Path,
    truth_dir: str | Path,
    baseline_prepared_dir: str | Path,
    learned_prepared_dir: str | Path,
    baseline_embeddings_path: str | Path,
    learned_embeddings_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare two frozen representations with identical users, splits, and probes."""
    truth_dir = Path(truth_dir)
    if truth_dir.name != "truth":
        raise ValueError("truth_dir must point directly to the simulator's truth/ directory")
    latent_path = truth_dir / "user_latents.csv.gz"
    if not latent_path.is_file():
        raise FileNotFoundError(f"Missing evaluator-only file: {latent_path}")

    baseline_metadata = read_json(Path(baseline_prepared_dir) / "prepared_metadata.json")
    metadata = read_json(Path(learned_prepared_dir) / "prepared_metadata.json")
    _validate_prepared_match(baseline_metadata, metadata)

    baseline = _load_embeddings(baseline_embeddings_path, "baseline")
    learned = _load_embeddings(learned_embeddings_path, "learned")
    common_keys = sorted(set(baseline) & set(learned))
    if not common_keys:
        raise ValueError("Baseline and learned exports have no shared user/cutoff rows")
    common_users = sorted(
        user_id
        for user_id in {key[0] for key in common_keys}
        if all((user_id, cutoff) in baseline and (user_id, cutoff) in learned
               for cutoff in ("train", "validation", "test"))
    )
    if len(common_users) < 10:
        raise ValueError(
            "Comparison requires at least 10 users with train, validation, and test embeddings; "
            f"found {len(common_users)}"
        )

    probe_fraction = float(config["evaluation"]["probe_train_fraction"])
    ridge_alpha = float(config["evaluation"]["ridge_alpha"])
    train_mask = np.asarray([_stable_fraction(user_id) < probe_fraction for user_id in common_users])
    if train_mask.sum() < 2 or (~train_mask).sum() < 2:
        raise ValueError("Stable comparison split produced fewer than two train or test users")

    matrices = {
        name: {
            cutoff: np.stack([source[(user_id, cutoff)] for user_id in common_users])
            for cutoff in ("train", "validation", "test")
        }
        for name, source in (("baseline", baseline), ("learned", learned))
    }
    latent = pd.read_csv(latent_path).set_index("user_id").loc[common_users]
    _, events = load_observed(observed_dir)
    counts = _event_counts_at_cutoffs(events, common_users, metadata)

    persistent = _compare_target_group(
        matrices, latent, PERSISTENT_TRAITS, train_mask, ridge_alpha
    )
    preferences = _compare_target_group(
        matrices, latent, PREFERENCE_TRAITS, train_mask, ridge_alpha
    )
    nuisance = _nuisance_matrix(latent, counts["test"], train_mask)
    preference_beyond_confounders = _incremental_probe(
        matrices, latent, PREFERENCE_TRAITS, nuisance, train_mask, ridge_alpha
    )
    geometry = {
        name: _geometry_report(values, common_users)
        for name, values in matrices.items()
    }
    activity = {
        name: _activity_dependence(values["test"], counts["test"], train_mask, ridge_alpha)
        for name, values in matrices.items()
    }
    future = _future_event_probes(
        matrices, common_users, events, metadata, train_mask, ridge_alpha
    )

    report: dict[str, Any] = {
        "comparison_contract": {
            "principle": "same frozen probe, users, cutoffs, split, and ridge penalty",
            "baseline_embedding_dim": int(matrices["baseline"]["test"].shape[1]),
            "learned_embedding_dim": int(matrices["learned"]["test"].shape[1]),
            "shared_users": len(common_users),
            "probe_train_users": int(train_mask.sum()),
            "probe_test_users": int((~train_mask).sum()),
            "ridge_alpha": ridge_alpha,
            "information_boundary": "truth/ is opened only by this comparison evaluator",
        },
        "persistent_information": persistent,
        "preference_information": preferences,
        "preference_beyond_geography_and_activity": preference_beyond_confounders,
        "stability_and_distinctiveness": geometry,
        "activity_volume_dependence": {
            "interpretation": (
                "Diagnostic only: lower dependence is not automatically better because digital "
                "engagement can be behaviorally meaningful. Large dependence warns that history "
                "quantity may dominate representation geometry."
            ),
            **activity,
        },
        "common_future_event_probes": {
            "protocol": (
                "Fit a ridge classifier on train-cutoff embeddings from probe-train users; "
                "evaluate the first post-validation-cutoff event for held-out users. Accuracy "
                "excludes target labels unseen by the probe, with known-label coverage reported."
            ),
            **future,
        },
        "requirements": _requirement_status(),
    }
    baseline_episode = Path(baseline_embeddings_path).parent / "baseline_episode_response.json"
    learned_episode = Path(learned_embeddings_path).parent / "episode_response.json"
    if baseline_episode.is_file() or learned_episode.is_file():
        if not (baseline_episode.is_file() and learned_episode.is_file()):
            raise ValueError("Episode comparison requires both baseline and learned episode reports")
        left, right = read_json(baseline_episode), read_json(learned_episode)
        contract_fields = ("source_hashes", "dense_users", "dense_timestamps_sha256", "boundary_bin_edges_hours")
        mismatches = [key for key in contract_fields if left["metric_contract"].get(key) != right["metric_contract"].get(key)]
        if mismatches:
            raise ValueError(f"Episode reports are not matched; mismatched fields: {mismatches}")
        metrics = {
            "within_episode_cosine": (left["R4_episode_coherence"]["within_episode_consecutive_cosine"]["mean"], right["R4_episode_coherence"]["within_episode_consecutive_cosine"]["mean"]),
            "boundary_change_magnitude": (left["R4_episode_coherence"]["boundary_change_magnitude"]["mean"], right["R4_episode_coherence"]["boundary_change_magnitude"]["mean"]),
            "post_episode_recovery": (left["R1_single_vector_diagnostics"]["post_episode_recovery_cosine"]["mean"], right["R1_single_vector_diagnostics"]["post_episode_recovery_cosine"]["mean"]),
        }
        report["episode_response_comparison"] = {name: {"baseline": a, "learned": b,
            "learned_minus_baseline": (b-a) if a is not None and b is not None else None} for name, (a,b) in metrics.items()}
    baseline_robust = Path(baseline_embeddings_path).parent / "robustness" / "baseline_event_removal.json"
    learned_robust = Path(learned_embeddings_path).parent / "robustness" / "learned_event_removal.json"
    if baseline_robust.is_file() or learned_robust.is_file():
        if not (baseline_robust.is_file() and learned_robust.is_file()):
            raise ValueError("R7 comparison requires both baseline and learned robustness reports")
        left, right = read_json(baseline_robust), read_json(learned_robust)
        contract_fields = ("source_hashes", "algorithm", "seed", "field_order", "removal_rates")
        mismatch = [key for key in contract_fields if left["metric_contract"].get(key) != right["metric_contract"].get(key)]
        if mismatch:
            raise ValueError(f"R7 reports are not matched; mismatched fields: {mismatch}")
        compared = []
        for a, b in zip(left["rates"], right["rates"]):
            row_fields = ("rate", "removed_events", "realized_removal_rate", "encoded_keys")
            bad = [key for key in row_fields if a.get(key) != b.get(key)]
            if bad:
                raise ValueError(f"R7 rate reports are not matched; mismatched fields: {bad}")
            compared.append({"rate": a["rate"], "matched_rows": a["matched_rows"],
                "cosine_drift_mean": {"baseline": a["cosine_drift"]["mean"],
                    "learned": b["cosine_drift"]["mean"],
                    "learned_minus_baseline": _nullable_delta(b["cosine_drift"]["mean"], a["cosine_drift"]["mean"])},
                "coverage": {"baseline": a["coverage"], "learned": b["coverage"],
                    "learned_minus_baseline": b["coverage"] - a["coverage"]},
                "probe_mean_r2_degradation": {"baseline": a["probe_mean_r2_degradation"],
                    "learned": b["probe_mean_r2_degradation"],
                    "learned_minus_baseline": _nullable_delta(b["probe_mean_r2_degradation"], a["probe_mean_r2_degradation"])}})
        report["R7_event_removal_comparison"] = {"axes_are_not_composited": True, "rates": compared}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {
        "json": str((output_dir / "embedding_comparison.json").resolve()),
        "markdown": str((output_dir / "embedding_comparison.md").resolve()),
    }
    write_json(report, output_dir / "embedding_comparison.json")
    (output_dir / "embedding_comparison.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    return report


def _nullable_delta(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _validate_prepared_match(
    baseline: dict[str, Any], learned: dict[str, Any]
) -> None:
    fields = (
        "source_files",
        "train_end",
        "validation_end",
        "categorical_fields",
        "continuous_fields",
    )
    mismatches = [field for field in fields if baseline.get(field) != learned.get(field)]
    if mismatches:
        raise ValueError(
            "Baseline and learned representations were not prepared from the same dataset/split "
            f"contract; mismatched fields: {mismatches}. Export the baseline using the learned "
            "run and experiment preparation before comparing."
        )


def _load_embeddings(path: str | Path, label: str) -> dict[tuple[str, str], np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} embeddings: {path}")
    payload = np.load(path, allow_pickle=False)
    required = {"user_id", "cutoff", "embedding"}
    if not required.issubset(payload.files):
        raise ValueError(f"{label} export is missing arrays: {sorted(required - set(payload.files))}")
    user_ids = payload["user_id"].astype(str)
    cutoffs = payload["cutoff"].astype(str)
    embeddings = payload["embedding"].astype(np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(user_ids) or len(user_ids) != len(cutoffs):
        raise ValueError(f"Malformed {label} embedding export at {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{label} embeddings contain non-finite values")
    result: dict[tuple[str, str], np.ndarray] = {}
    for user_id, cutoff, vector in zip(user_ids, cutoffs, embeddings):
        key = (str(user_id), str(cutoff))
        if key in result:
            raise ValueError(f"Duplicate {label} embedding row: {key}")
        result[key] = vector
    return result


def _compare_target_group(
    matrices: dict[str, dict[str, np.ndarray]],
    latent: pd.DataFrame,
    targets: list[str],
    train_mask: np.ndarray,
    ridge_alpha: float,
) -> dict[str, Any]:
    available = [target for target in targets if target in latent.columns]
    y = latent[available].to_numpy(dtype=np.float64)
    results = {
        name: _ridge_regression_report(values["test"], y, available, train_mask, ridge_alpha)
        for name, values in matrices.items()
    }
    return {
        **results,
        "learned_minus_baseline_mean_r2": (
            results["learned"]["mean_r2"] - results["baseline"]["mean_r2"]
        ),
    }


def _incremental_probe(
    matrices: dict[str, dict[str, np.ndarray]],
    latent: pd.DataFrame,
    targets: list[str],
    nuisance: np.ndarray,
    train_mask: np.ndarray,
    ridge_alpha: float,
) -> dict[str, Any]:
    available = [target for target in targets if target in latent.columns]
    y = latent[available].to_numpy(dtype=np.float64)
    nuisance_report = _ridge_regression_report(
        nuisance, y, available, train_mask, ridge_alpha
    )
    result: dict[str, Any] = {
        "nuisance_only": nuisance_report,
        "nuisance_features": [
            "home_region_id",
            "work_region_id",
            "log1p_observed_event_count_at_test",
        ],
    }
    for name, values in matrices.items():
        combined = np.concatenate([values["test"], nuisance], axis=1)
        report = _ridge_regression_report(combined, y, available, train_mask, ridge_alpha)
        report["incremental_mean_r2_over_nuisance"] = (
            report["mean_r2"] - nuisance_report["mean_r2"]
        )
        result[name] = report
    result["learned_minus_baseline_incremental_r2"] = (
        result["learned"]["incremental_mean_r2_over_nuisance"]
        - result["baseline"]["incremental_mean_r2_over_nuisance"]
    )
    return result


def _ridge_regression_report(
    x: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    train_mask: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    x_train, x_test = _standardize_train_test(x[train_mask], x[~train_mask])
    prediction = _ridge_predict(x_train, y[train_mask], x_test, alpha)
    scores: dict[str, float] = {}
    for index, target in enumerate(target_names):
        actual = y[~train_mask, index]
        denominator = float(np.square(actual - actual.mean()).sum())
        score = 1.0 - float(np.square(actual - prediction[:, index]).sum()) / max(
            denominator, 1e-12
        )
        scores[target] = score
    return {
        "metric": "held-out-user ridge-probe R2",
        "r2": scores,
        "mean_r2": float(np.mean(list(scores.values()))) if scores else None,
    }


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    target_mean = y_train.mean(axis=0, keepdims=True)
    centered = y_train - target_mean
    if x_train.shape[1] > x_train.shape[0]:
        dual = np.linalg.solve(
            x_train @ x_train.T + alpha * np.eye(x_train.shape[0]), centered
        )
        return target_mean + x_test @ x_train.T @ dual
    weights = np.linalg.solve(
        x_train.T @ x_train + alpha * np.eye(x_train.shape[1]), x_train.T @ centered
    )
    return target_mean + x_test @ weights


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def _geometry_report(values: dict[str, np.ndarray], user_ids: list[str]) -> dict[str, Any]:
    train = _row_normalize(values["train"])
    validation = _row_normalize(values["validation"])
    test = _row_normalize(values["test"])
    train_test = train @ test.T
    diagonal = np.diag(train_test)
    off_diagonal = train_test[~np.eye(len(user_ids), dtype=bool)]
    order = np.argsort(-train_test, axis=1)
    ranks = np.empty(len(user_ids), dtype=np.int64)
    for index in range(len(user_ids)):
        ranks[index] = int(np.flatnonzero(order[index] == index)[0]) + 1
    test_similarity = test @ test.T
    between_test = test_similarity[~np.eye(len(user_ids), dtype=bool)]
    maximum_impostor = np.max(
        np.where(np.eye(len(user_ids), dtype=bool), -np.inf, train_test), axis=1
    )
    return {
        "same_user_cosine": {
            "train_to_validation": _distribution(np.sum(train * validation, axis=1)),
            "validation_to_test": _distribution(np.sum(validation * test, axis=1)),
            "train_to_test": _distribution(diagonal),
        },
        "different_user_cosine": {
            "train_to_test_mean": float(off_diagonal.mean()),
            "test_to_test_mean": float(between_test.mean()),
            "test_to_test_p90": float(np.quantile(between_test, 0.90)),
        },
        "same_minus_different_train_test_cosine": float(diagonal.mean() - off_diagonal.mean()),
        "same_minus_strongest_impostor_cosine": _distribution(diagonal - maximum_impostor),
        "temporal_user_retrieval": {
            "train_query_test_gallery_top1": float(np.mean(ranks == 1)),
            "top5": float(np.mean(ranks <= 5)),
            "mrr": float(np.mean(1.0 / ranks)),
        },
        "test_geometry": _effective_rank(values["test"]),
    }


def _effective_rank(x: np.ndarray) -> dict[str, float]:
    centered = x - x.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return {"effective_rank": 0.0, "effective_rank_ratio": 0.0}
    probabilities = eigenvalues[eigenvalues > 1e-12] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    maximum_rank = max(1, min(x.shape[0] - 1, x.shape[1]))
    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / maximum_rank,
    }


def _activity_dependence(
    embeddings: np.ndarray,
    counts: np.ndarray,
    train_mask: np.ndarray,
    ridge_alpha: float,
) -> dict[str, float]:
    target = np.log1p(counts.astype(np.float64))
    report = _ridge_regression_report(
        embeddings, target[:, None], ["log1p_event_count"], train_mask, ridge_alpha
    )
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "held_out_count_probe_r2": report["mean_r2"],
        "embedding_norm_count_pearson": _pearson(norms, target),
    }


def _future_event_probes(
    matrices: dict[str, dict[str, np.ndarray]],
    user_ids: list[str],
    events: pd.DataFrame,
    metadata: dict[str, Any],
    train_mask: np.ndarray,
    ridge_alpha: float,
) -> dict[str, Any]:
    train_end = pd.Timestamp(metadata["train_end"])
    validation_end = pd.Timestamp(metadata["validation_end"])
    after_train = _first_events_after(events, user_ids, train_end)
    after_validation = _first_events_after(events, user_ids, validation_end)
    reports: dict[str, Any] = {}
    for objective, field in TARGET_FIELDS.items():
        if field not in events.columns:
            continue
        train_indices = [
            index for index, user_id in enumerate(user_ids)
            if train_mask[index] and user_id in after_train
        ]
        test_indices = [
            index for index, user_id in enumerate(user_ids)
            if not train_mask[index] and user_id in after_validation
        ]
        train_labels = np.asarray(
            [str(after_train[user_ids[index]][field]) for index in train_indices], dtype=str
        )
        test_labels = np.asarray(
            [str(after_validation[user_ids[index]][field]) for index in test_indices], dtype=str
        )
        classes = sorted(set(train_labels))
        class_to_index = {value: index for index, value in enumerate(classes)}
        known = np.asarray([value in class_to_index for value in test_labels], dtype=bool)
        field_report: dict[str, Any] = {
            "train_examples": len(train_indices),
            "test_examples": len(test_indices),
            "train_classes": len(classes),
            "known_label_coverage": float(known.mean()) if len(known) else None,
        }
        if len(classes) < 2 or not known.any():
            field_report["status"] = "insufficient_known_labels"
            reports[objective] = field_report
            continue
        y_train = np.zeros((len(train_labels), len(classes)), dtype=np.float64)
        y_train[np.arange(len(train_labels)), [class_to_index[value] for value in train_labels]] = 1.0
        y_test = np.asarray([class_to_index[value] for value in test_labels[known]], dtype=np.int64)
        for name, values in matrices.items():
            x_train, x_test = _standardize_train_test(
                values["train"][train_indices], values["validation"][test_indices]
            )
            scores = _ridge_predict(x_train, y_train, x_test[known], ridge_alpha)
            ranking = np.argsort(-scores, axis=1)
            top1 = ranking[:, 0]
            top5 = ranking[:, : min(5, len(classes))]
            field_report[name] = {
                "accuracy": float(np.mean(top1 == y_test)),
                "top5": float(np.mean(np.any(top5 == y_test[:, None], axis=1))),
            }
        field_report["learned_minus_baseline_accuracy"] = (
            field_report["learned"]["accuracy"] - field_report["baseline"]["accuracy"]
        )
        reports[objective] = field_report
    return reports


def _first_events_after(
    events: pd.DataFrame, user_ids: list[str], cutoff: pd.Timestamp
) -> dict[str, pd.Series]:
    eligible = events[events["timestamp"] > cutoff]
    first = eligible.groupby("user_id", sort=False).head(1)
    wanted = set(user_ids)
    return {
        str(row["user_id"]): row
        for _, row in first.iterrows()
        if str(row["user_id"]) in wanted
    }


def _event_counts_at_cutoffs(
    events: pd.DataFrame, user_ids: list[str], metadata: dict[str, Any]
) -> dict[str, np.ndarray]:
    cutoffs = {
        "train": pd.Timestamp(metadata["train_end"]),
        "validation": pd.Timestamp(metadata["validation_end"]),
        "test": events["timestamp"].max(),
    }
    result: dict[str, np.ndarray] = {}
    for name, cutoff in cutoffs.items():
        counts = events.loc[events["timestamp"] <= cutoff].groupby("user_id").size()
        result[name] = np.asarray([int(counts.get(user_id, 0)) for user_id in user_ids])
    return result


def _nuisance_matrix(
    latent: pd.DataFrame, counts: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    columns: list[np.ndarray] = [np.log1p(counts.astype(np.float64))[:, None]]
    for field in ("home_region_id", "work_region_id"):
        if field not in latent.columns:
            continue
        values = latent[field].astype(str).to_numpy()
        classes = sorted(set(values[train_mask]))
        for value in classes:
            columns.append((values == value).astype(np.float64)[:, None])
    return np.concatenate(columns, axis=1)


def _row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _requirement_status() -> dict[str, Any]:
    return {
        "R1_persistent_context_separation": {
            "status": "partial",
            "evidence": [
                "persistent_information",
                "preference_information",
                "stability_and_distinctiveness",
            ],
            "missing": "episode-conditioned embeddings for direct persistent/context separation",
        },
        "R2_multiscale_spatial_fidelity": {
            "status": "partial",
            "evidence": [
                "common_future_event_probes.next_region",
                "common_future_event_probes.next_geohash_5",
                "common_future_event_probes.next_geohash_7",
            ],
            "missing": "metric-distance and geohash-boundary tests",
        },
        "R3_multiscale_temporal_fidelity": {
            "status": "partial",
            "evidence": ["common_future_event_probes", "stability_and_distinctiveness"],
            "missing": "hour, duration, routine, and periodicity probes",
        },
        "R4_episode_coherence": {
            "status": "not_measurable_from_three_global_cutoffs",
            "missing": "protected episode-boundary joins and response metrics over dense exports",
        },
        "R5_preference_opportunity_separation": {
            "status": "partial",
            "evidence": ["preference_beyond_geography_and_activity"],
            "missing": "matched counterfactual runs with changed candidate exposure",
        },
        "R6_cross_service_alignment": {
            "status": "partial",
            "evidence": ["common_future_event_probes.next_service"],
            "missing": "leave-one-service-out encoding and cross-service prediction",
        },
        "R7_noise_and_sparsity_robustness": {
            "status": "partial",
            "evidence": ["activity_volume_dependence"],
            "missing": "controlled event removal, GPS perturbation, and missing-service tests",
        },
        "R8_geographic_temporal_generalization": {
            "status": "partial",
            "evidence": ["common_future_event_probes"],
            "missing": "explicit held-out-region evaluation",
        },
        "R9_new_context_recommendation": {
            "status": "blocked_by_data_contract",
            "missing": "observable requests, impressions, availability, and candidate metadata",
        },
        "R10_representation_uncertainty": {
            "status": "pending",
            "missing": "window-resampling agreement or another per-user reliability estimate",
        },
        "R11_nonstationarity": {
            "status": "not_measurable_from_three_global_cutoffs",
            "missing": "matched temporary-trip and sustained-change trajectories",
        },
        "R12_privacy": {
            "status": "pending",
            "missing": "membership, attribute-inference, and memorization audits",
        },
        "R13_computational_efficiency": {
            "status": "pending",
            "missing": "training, update, export, latency, and memory measurements",
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    contract = report["comparison_contract"]
    persistent = report["persistent_information"]
    preferences = report["preference_information"]
    beyond = report["preference_beyond_geography_and_activity"]
    geometry = report["stability_and_distinctiveness"]
    activity = report["activity_volume_dependence"]

    rows = [
        ("Persistent-trait probe mean R2", persistent["baseline"]["mean_r2"], persistent["learned"]["mean_r2"], "higher"),
        ("Category-preference probe mean R2", preferences["baseline"]["mean_r2"], preferences["learned"]["mean_r2"], "higher"),
        ("Preference incremental R2 beyond geography/activity", beyond["baseline"]["incremental_mean_r2_over_nuisance"], beyond["learned"]["incremental_mean_r2_over_nuisance"], "higher"),
        ("Same-user train-to-test cosine", geometry["baseline"]["same_user_cosine"]["train_to_test"]["mean"], geometry["learned"]["same_user_cosine"]["train_to_test"]["mean"], "higher, unless collapsed"),
        ("Same-minus-different cosine", geometry["baseline"]["same_minus_different_train_test_cosine"], geometry["learned"]["same_minus_different_train_test_cosine"], "higher"),
        ("Temporal user retrieval top1", geometry["baseline"]["temporal_user_retrieval"]["train_query_test_gallery_top1"], geometry["learned"]["temporal_user_retrieval"]["train_query_test_gallery_top1"], "higher"),
        ("Effective-rank ratio at test", geometry["baseline"]["test_geometry"]["effective_rank_ratio"], geometry["learned"]["test_geometry"]["effective_rank_ratio"], "diagnostic"),
        ("Held-out event-count probe R2", activity["baseline"]["held_out_count_probe_r2"], activity["learned"]["held_out_count_probe_r2"], "diagnostic"),
    ]
    lines = [
        "# Baseline vs learned embedding comparison",
        "",
        f"Shared users: **{contract['shared_users']}**  ",
        f"Probe split: **{contract['probe_train_users']} train / {contract['probe_test_users']} test users**  ",
        f"Dimensions: **{contract['baseline_embedding_dim']} baseline / {contract['learned_embedding_dim']} learned**",
        "",
        "## Core comparison",
        "",
        "| Metric | Baseline | Learned | Learned - baseline | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for label, baseline_value, learned_value, direction in rows:
        lines.append(
            f"| {label} | {_fmt(baseline_value)} | {_fmt(learned_value)} | "
            f"{_fmt(learned_value - baseline_value)} | {direction} |"
        )

    lines.extend([
        "",
        "Stability is interpreted together with distinctiveness. A high same-user cosine is not "
        "evidence of a useful persistent representation when different users are also nearly identical.",
        "",
        "## Common future-event probes",
        "",
        "| Target | Known-label coverage | Baseline accuracy | Learned accuracy | Delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for target, values in report["common_future_event_probes"].items():
        if target == "protocol":
            continue
        if "baseline" not in values:
            lines.append(
                f"| {target} | {_fmt(values.get('known_label_coverage'))} | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {target} | {_fmt(values['known_label_coverage'])} | "
            f"{_fmt(values['baseline']['accuracy'])} | {_fmt(values['learned']['accuracy'])} | "
            f"{_fmt(values['learned_minus_baseline_accuracy'])} |"
        )

    lines.extend([
        "",
        "## Requirement coverage",
        "",
        "| Requirement | Status | Still missing |",
        "|---|---|---|",
    ])
    for name, values in report["requirements"].items():
        lines.append(
            f"| {name} | {values['status']} | {values.get('missing', '')} |"
        )
    lines.extend([
        "",
        "No single aggregate winner is reported. The intended representation must trade off persistent "
        "information, temporal continuity, distinctiveness, contextual prediction, and robustness; "
        "collapsing those axes into one score would hide the failure modes this suite is designed to expose.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
