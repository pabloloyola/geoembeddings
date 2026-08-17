#!/usr/bin/env python3
"""Pre-model recoverability gate for GeoEmbeddings.

This is evaluator-only diagnostic code.  It deliberately reads ``truth/`` and
must never be imported by prepare, training, export, or production inference.

The script can either:

1. generate matched simulator pairs using the repository's existing
   ``simulate-pair`` command, then use the first reference run as the audit run;
2. audit an existing synthetic run with ``--existing-run``.

It writes:

* ``recoverability_report.json`` -- complete machine-readable evidence;
* ``oracle_probes.csv`` -- held-out-user oracle probe results;
* ``recoverability_summary.md`` -- short decision-oriented summary.

No simulator, model, or dataset artifact is edited or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from geoembeddings.pair_integrity import require_passing_pair_integrity


SCRIPT_SCHEMA = "geoembeddings-recoverability-gate/1.2"
FACTOR_REGISTRY_SCHEMA = "geoembeddings-recoverability-factor-registry/1.0"
DEFAULT_FACTOR_REGISTRY = Path("configs/recoverability/recoverability_factor_registry.json")
DEFAULT_SUSTAINED_PAIR = Path("pairs/recoverable-benchmark-v1/track_a/clean_sustained")
DEFAULT_TEMPORARY_SCHEDULE_PAIR = Path("pairs/recoverable-benchmark-v1/track_a/clean_temporary_schedule")
RECOVERABILITY_GATES = {
    "held_out_balanced_accuracy_min": 0.70,
    "held_out_auroc_min": 0.70,
    "matched_knn_purity_at_10_min": 0.70,
    "standardized_within_between_separation_min": 0.20,
    "bootstrap_lower_ci_above_stratified_permutation_null": True,
}
V2_RECOVERABILITY_GATES = {
    "held_out_balanced_accuracy_min": 0.70,
    "held_out_auroc_min": 0.70,
    "bootstrap_lower_ci_above_stratified_permutation_null": True,
}
DEFAULT_INTERVENTIONS = ("sustained-preference", "schedule-shift", "observation")
CORE_TRAITS = (
    "price_sensitivity",
    "distance_sensitivity",
    "novelty_seeking",
    "family_orientation",
    "travel_propensity",
    "time_flexibility",
    "transit_preference",
    "digital_engagement",
)

# Commit-specific findings from the DGP audit.  These are hypotheses to verify
# again if simulator.py changes; the script reports the current git revision.
STRUCTURAL_AUDIT = {
    "time_flexibility": {
        "status": "exclude_until_repaired",
        "reason": "Generated but does not causally affect the observed history at commit 49125a9.",
    },
    "transit_preference": {
        "status": "exclude_until_repaired",
        "reason": "No direct behavioral mechanism; at most geography supplies an indirect proxy.",
    },
    "pref_*": {
        "status": "test_after_category_choice_repair",
        "reason": "Config-v5 category selection supplies the observable path; older runs remain structurally unrecoverable.",
    },
    "price_sensitivity": {
        "status": "test_with_enriched_observables",
        "reason": "Affects choice, but price/POI metadata is absent from the current event encoder.",
    },
    "distance_sensitivity": {
        "status": "test_with_enriched_observables",
        "reason": "May be recoverable from chosen locations and request/candidate travel time.",
    },
    "digital_engagement": {
        "status": "shortcut_risk",
        "reason": "Affects observation volume, so event-count-only recovery is not preference recovery.",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if pd.isna(value):
        return None
    return value


def _read(run_dir: Path, relative: str, *, required: bool = True) -> pd.DataFrame:
    path = run_dir / relative
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing required table: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cli_prefix() -> list[str]:
    executable = shutil.which("geoembed")
    if executable:
        return [executable]
    return [sys.executable, "-m", "geoembeddings.cli"]


def _run(command: Sequence[str]) -> None:
    print("\n+", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def generate_pairs(args: argparse.Namespace, output_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    audit_run: Path | None = None
    for intervention in args.interventions:
        root = output_dir / "pairs" / intervention
        reference = root / "reference_run"
        changed = root / "intervention_run"
        pair_dir = root / "pair"
        command = [
            *_cli_prefix(),
            "simulate-pair",
            "--config", str(args.config),
            "--intervention", intervention,
            "--reference-run-dir", str(reference),
            "--intervention-run-dir", str(changed),
            "--pair-dir", str(pair_dir),
            "--users", str(args.users),
            "--days", str(args.days),
            "--seed", str(args.seed),
        ]
        _run(command)
        audit_run = audit_run or reference
        pairs.append({
            "intervention": intervention,
            "reference_run": reference,
            "intervention_run": changed,
            "pair_dir": pair_dir,
        })
    if audit_run is None:
        raise ValueError("At least one intervention is required when --existing-run is absent")
    return audit_run, pairs


def _stable_fold(user_id: str, folds: int) -> int:
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _profile_features(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["user_id"] = users["user_id"].astype(str)
    raw = users.set_index("user_id")
    numeric = raw.select_dtypes(include=[np.number, "bool"]).astype(float)
    categorical = raw.drop(columns=numeric.columns, errors="ignore").fillna("<missing>").astype(str)
    encoded = pd.get_dummies(categorical, prefix=categorical.columns, dtype=float)
    return pd.concat([numeric, encoded], axis=1).sort_index(axis=1)


def _normalized_crosstab(events: pd.DataFrame, column: str, user_index: pd.Index) -> pd.DataFrame:
    if column not in events.columns:
        return pd.DataFrame(index=user_index)
    table = pd.crosstab(events["user_id"], events[column].fillna("<missing>").astype(str)).astype(float)
    table = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    table.columns = [f"{column}={value}" for value in table.columns]
    return table.reindex(user_index, fill_value=0.0)


def _event_features(events: pd.DataFrame, user_index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["user_id"] = events["user_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.sort_values(["user_id", "timestamp"])
    group = events.groupby("user_id", sort=False)

    basic = pd.DataFrame(index=user_index)
    basic["log1p_event_count"] = np.log1p(group.size()).reindex(user_index, fill_value=0).astype(float)
    basic["active_days"] = group["timestamp"].apply(lambda s: s.dt.date.nunique()).reindex(user_index, fill_value=0)
    basic["active_hours"] = group["timestamp"].apply(lambda s: s.dt.floor("h").nunique()).reindex(user_index, fill_value=0)
    for column in ("service_id", "action_type", "observation_mode", "object_category", "region_id"):
        if column in events:
            basic[f"distinct_{column}"] = group[column].nunique().reindex(user_index, fill_value=0)

    hours = events["timestamp"].dt.hour + events["timestamp"].dt.minute / 60.0
    weekday = events["timestamp"].dt.dayofweek
    events["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    events["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    events["dow_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    events["dow_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    events["delta_minutes"] = group["timestamp"].diff().dt.total_seconds().div(60).clip(lower=0)

    aggregates: list[pd.DataFrame] = [basic]
    numeric_columns = [
        column for column in (
            "latitude", "longitude", "location_accuracy_m", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "delta_minutes",
        ) if column in events.columns
    ]
    if numeric_columns:
        numeric = events.groupby("user_id")[numeric_columns].agg(["mean", "std", "min", "max"])
        numeric.columns = [f"{column}_{stat}" for column, stat in numeric.columns]
        aggregates.append(numeric.reindex(user_index).fillna(0.0))
    for column in (
        "service_id", "action_type", "observation_mode", "object_category",
        "region_id", "geohash_5",
    ):
        aggregates.append(_normalized_crosstab(events, column, user_index))

    all_events = _numeric_frame(pd.concat(aggregates, axis=1).reindex(user_index).fillna(0.0))
    volume = all_events[[column for column in all_events if column in {
        "log1p_event_count", "active_days", "active_hours"
    }]].copy()
    return all_events, volume


def _enriched_poi_features(
    events: pd.DataFrame, catalog: pd.DataFrame, user_index: pd.Index
) -> pd.DataFrame:
    if events.empty or catalog.empty or "object_id" not in events or "poi_id" not in catalog:
        return pd.DataFrame(index=user_index)
    joined = events.copy()
    joined["user_id"] = joined["user_id"].astype(str)
    catalog = catalog.copy()
    catalog["poi_id"] = catalog["poi_id"].astype(str)
    joined = joined.merge(catalog, left_on="object_id", right_on="poi_id", how="inner", suffixes=("_event", "_poi"))
    if joined.empty:
        return pd.DataFrame(index=user_index)
    numeric_columns = [
        column for column in ("price_level", "family_suitability", "local_popularity")
        if column in joined
    ]
    parts: list[pd.DataFrame] = []
    if numeric_columns:
        numeric = joined.groupby("user_id")[numeric_columns].agg(["mean", "std", "min", "max"])
        numeric.columns = [f"poi_{column}_{stat}" for column, stat in numeric.columns]
        parts.append(numeric)
    for column in ("category", "environment"):
        if column in joined:
            table = pd.crosstab(joined["user_id"], joined[column].fillna("<missing>")).astype(float)
            table = table.div(table.sum(axis=1).replace(0, 1), axis=0)
            table.columns = [f"poi_{column}={value}" for value in table.columns]
            parts.append(table)
    if not parts:
        return pd.DataFrame(index=user_index)
    return _numeric_frame(pd.concat(parts, axis=1).reindex(user_index).fillna(0.0))


def build_feature_sets(run_dir: Path) -> tuple[pd.Index, dict[str, pd.DataFrame], dict[str, Any]]:
    users = _read(run_dir, "observed/users_observed.csv.gz")
    events = _read(run_dir, "observed/observed_events.csv.gz")
    catalog = _read(run_dir, "observed/poi_catalog.csv.gz", required=False)
    user_index = pd.Index(sorted(users["user_id"].astype(str).unique()), name="user_id")

    profile = _profile_features(users).reindex(user_index, fill_value=0.0)
    event_features, volume = _event_features(events, user_index)
    poi_features = _enriched_poi_features(events, catalog, user_index)
    feature_sets = {
        "volume_only": volume,
        "profile_only": profile,
        "current_events": event_features,
        "current_plus_profile": pd.concat([event_features, profile], axis=1),
        "enriched_observables": pd.concat([event_features, profile, poi_features], axis=1),
    }
    feature_sets = {name: _numeric_frame(frame.loc[user_index]) for name, frame in feature_sets.items()}
    metadata = {
        "users": len(user_index),
        "events": len(events),
        "profile_features": profile.shape[1],
        "current_event_features": event_features.shape[1],
        "enriched_poi_features": poi_features.shape[1],
        "feature_dimensions": {name: frame.shape[1] for name, frame in feature_sets.items()},
    }
    return user_index, feature_sets, metadata


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    # Remove fold-local zero/rare columns before scaling.  Otherwise a category
    # observed in one held-out user can be hundreds of standard deviations from
    # the training mean and make an otherwise regularized oracle explode.
    scale_all = x.std(axis=0)
    support = np.count_nonzero(np.abs(x) > 1e-12, axis=0)
    minimum_support = max(5, int(math.ceil(0.02 * len(x))))
    keep = (scale_all >= 1e-8) & (support >= minimum_support)
    if not keep.any():
        keep[np.argmax(scale_all)] = True
    selected = x[:, keep]
    mean = selected.mean(axis=0, keepdims=True)
    scale = selected.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    z = np.clip((selected - mean) / scale, -10.0, 10.0)
    target_mean = float(y.mean())
    centered = y - target_mean
    if z.shape[1] > z.shape[0]:
        dual = np.linalg.solve(z @ z.T + alpha * np.eye(z.shape[0]), centered)
        weights = z.T @ dual
    else:
        weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
    return weights, mean.ravel(), target_mean, scale.ravel(), keep


def _cross_validated_ridge_fixed(
    x: np.ndarray, y: np.ndarray, users: Sequence[str], folds: int, alpha: float
) -> dict[str, Any]:
    assignments = np.asarray([_stable_fold(str(user), folds) for user in users])
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        if train.sum() < 2 or test.sum() < 2:
            continue
        weights, mean, target_mean, scale, keep = _fit_ridge(x[train], y[train], alpha)
        z_test = np.clip((x[test][:, keep] - mean) / scale, -10.0, 10.0)
        predictions[test] = target_mean + z_test @ weights
        fold_rows.append({"fold": fold, "train_users": int(train.sum()), "test_users": int(test.sum())})
    valid = np.isfinite(predictions)
    if valid.sum() < max(10, folds * 2):
        return {"status": "insufficient_users", "evaluated_users": int(valid.sum())}
    denominator = float(np.square(y[valid] - y[valid].mean()).sum())
    r2 = 1.0 - float(np.square(y[valid] - predictions[valid]).sum()) / max(denominator, 1e-12)
    correlation = float(np.corrcoef(y[valid], predictions[valid])[0, 1]) if np.std(predictions[valid]) > 0 and np.std(y[valid]) > 0 else None
    return {
        "status": "ok",
        "users": int(valid.sum()),
        "r2": r2,
        "pearson": correlation,
        "alpha": alpha,
        "folds": fold_rows,
    }


def _cross_validated_ridge(
    x: np.ndarray, y: np.ndarray, users: Sequence[str], folds: int,
    alphas: Sequence[float],
) -> dict[str, Any]:
    candidates = [_cross_validated_ridge_fixed(x, y, users, folds, float(alpha)) for alpha in alphas]
    valid = [result for result in candidates if result.get("status") == "ok"]
    if not valid:
        return candidates[0]
    selected = max(valid, key=lambda result: float(result["r2"]))
    return {
        **selected,
        "selection": "best stable-user cross-validated R2 over declared alpha grid",
        "alpha_candidates": [
            {"alpha": result.get("alpha"), "r2": result.get("r2")}
            for result in candidates
        ],
    }


def oracle_probes(
    run_dir: Path,
    users: pd.Index,
    feature_sets: dict[str, pd.DataFrame],
    *,
    folds: int,
    alphas: Sequence[float],
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latent = _read(run_dir, "truth/user_latents.csv.gz")
    latent["user_id"] = latent["user_id"].astype(str)
    latent = latent.set_index("user_id").reindex(users)
    traits = [trait for trait in CORE_TRAITS if trait in latent.columns]
    traits.extend(sorted(column for column in latent.columns if column.startswith("pref_")))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    full_report: dict[str, Any] = {}

    for trait in traits:
        y = pd.to_numeric(latent[trait], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < max(20, folds * 4) or np.std(y[valid]) < 1e-10:
            full_report[trait] = {"status": "insufficient_or_constant_target", "users": int(valid.sum())}
            continue
        trait_report: dict[str, Any] = {}
        for name, frame in feature_sets.items():
            x = frame.to_numpy(dtype=float)[valid]
            result = _cross_validated_ridge(x, y[valid], users[valid], folds, alphas)
            trait_report[name] = result
            rows.append({
                "trait": trait,
                "feature_set": name,
                "features": x.shape[1],
                "users": int(valid.sum()),
                "r2": result.get("r2"),
                "pearson": result.get("pearson"),
            })

        enriched = trait_report.get("enriched_observables", {})
        null_scores: list[float] = []
        if enriched.get("status") == "ok" and permutations:
            x = feature_sets["enriched_observables"].to_numpy(dtype=float)[valid]
            for _ in range(permutations):
                null = _cross_validated_ridge(x, rng.permutation(y[valid]), users[valid], folds, alphas)
                if null.get("r2") is not None:
                    null_scores.append(float(null["r2"]))
        null_p95 = float(np.quantile(null_scores, 0.95)) if null_scores else None
        enriched_r2 = enriched.get("r2")
        structural_key = "pref_*" if trait.startswith("pref_") else trait
        structural = STRUCTURAL_AUDIT.get(structural_key)
        if structural and structural["status"] == "exclude_until_repaired":
            decision = "exclude_until_repaired"
        elif enriched_r2 is None:
            decision = "not_measurable"
        elif enriched_r2 > 0.05 and (null_p95 is None or enriched_r2 > null_p95):
            decision = "observable_signal_demonstrated"
        elif enriched_r2 > 0:
            decision = "weak_observable_signal"
        else:
            decision = "observable_signal_not_demonstrated"
        trait_report["permutation_null"] = {
            "permutations": len(null_scores),
            "r2_p95": null_p95,
        }
        trait_report["structural_audit"] = structural
        trait_report["decision"] = decision
        full_report[trait] = trait_report
    return rows, full_report


def _load_factor_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != FACTOR_REGISTRY_SCHEMA:
        raise ValueError(f"Unsupported recoverability factor registry: {path}")
    factors = registry.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("Recoverability factor registry must declare factors")
    names = [factor.get("name") for factor in factors]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("Recoverability factor registry has missing or duplicate factor names")
    return registry


def _history_cutoff(run_dir: Path) -> pd.Timestamp:
    config_path = run_dir / "config.resolved.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        run = config.get("run", {})
        if run.get("start_date") is not None and run.get("days") is not None:
            start = pd.Timestamp(run["start_date"], tz="Asia/Tokyo")
            return (start + pd.Timedelta(int(run["days"]), unit="D")).tz_convert("UTC")
    events = _read(run_dir, "observed/observed_events.csv.gz")
    timestamps = pd.to_datetime(events["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("Cannot determine recoverability evaluation cutoff")
    return timestamps.max() + pd.Timedelta(nanoseconds=1)


def _observed_history_matrix(run_dir: Path) -> tuple[pd.Index, pd.DataFrame, dict[str, Any]]:
    users = _read(run_dir, "observed/users_observed.csv.gz")
    events = _read(run_dir, "observed/observed_events.csv.gz")
    cutoff = _history_cutoff(run_dir)
    events = events.copy()
    events["user_id"] = events["user_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.loc[events["timestamp"].notna() & (events["timestamp"] < cutoff)].copy()
    user_index = pd.Index(sorted(users["user_id"].astype(str).unique()), name="user_id")
    event_features, _ = _event_features(events, user_index)
    profile = _profile_features(users).reindex(user_index, fill_value=0.0)
    features = _numeric_frame(pd.concat([event_features, profile], axis=1).reindex(user_index).fillna(0.0))
    return user_index, features, {
        "cutoff": cutoff.isoformat(),
        "users": int(len(user_index)),
        "events_before_cutoff": int(len(events)),
        "feature_columns": int(features.shape[1]),
        "feature_source": "observed/users_observed.csv.gz and observed/observed_events.csv.gz only",
    }


def _observed_category_count_difference(
    run_dir: Path, user_index: pd.Index, pair_map: dict[str, Sequence[str]],
    *, pair_column: str = "stable_affinity_pair_id",
) -> pd.DataFrame:
    """Return the declared-pair count difference from model-visible events.

    The pair mapping is evaluator-only metadata; the feature itself is built
    exclusively from observed ``object_category`` tokens before the canonical
    history cutoff.  This is intentionally a raw count feature rather than a
    normalized crosstab so event volume cannot erase the declared signal.
    """
    latent = _read(run_dir, "truth/user_latents.csv.gz")
    latent["user_id"] = latent["user_id"].astype(str)
    pair_by_user = latent.set_index("user_id")[pair_column].reindex(user_index)
    events = _read(run_dir, "observed/observed_events.csv.gz")
    events["user_id"] = events["user_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    cutoff = _history_cutoff(run_dir)
    events = events.loc[
        events["timestamp"].notna() & (events["timestamp"] < cutoff)
    ]
    differences = pd.Series(0.0, index=user_index, name="stable_affinity_count_difference")
    for user_id, rows in events.groupby("user_id", sort=False):
        pair_id = str(pair_by_user.get(user_id, ""))
        if pair_id not in pair_map or len(pair_map[pair_id]) != 2:
            raise ValueError(f"missing or malformed stable-affinity pair for user {user_id}")
        first, second = pair_map[pair_id]
        differences.loc[user_id] = float(
            (rows["object_category"] == second).sum()
            - (rows["object_category"] == first).sum()
        )
    return differences.to_frame()


def _history_matching_strata(run_dir: Path, user_index: pd.Index) -> pd.Series:
    users = _read(run_dir, "observed/users_observed.csv.gz")
    events = _read(run_dir, "observed/observed_events.csv.gz")
    events = events.copy()
    events["user_id"] = events["user_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce").dt.tz_convert("Asia/Tokyo")
    events = events.loc[events["timestamp"].notna()].copy()
    group = events.groupby("user_id", sort=False)
    dominant_region = group["region_id"].agg(lambda values: str(values.mode().iloc[0]) if not values.mode().empty else "<missing>")
    hour_bucket = group["timestamp"].agg(lambda values: int((values.dt.hour.mode().iloc[0] if not values.dt.hour.mode().empty else 0) // 4))
    weekday_bucket = group["timestamp"].agg(lambda values: int(values.dt.dayofweek.mode().iloc[0]) if not values.dt.dayofweek.mode().empty else 0)
    volume = group.size().reindex(user_index, fill_value=0)
    volume_rank = volume.rank(method="first", pct=True)
    volume_bucket = np.minimum(9, np.floor(volume_rank.to_numpy(dtype=float) * 10).astype(int))
    fallback_region = users.assign(user_id=users["user_id"].astype(str)).set_index("user_id").get("home_region_id", pd.Series(dtype=str))
    rows = []
    for position, user in enumerate(user_index.astype(str)):
        region = dominant_region.get(user, fallback_region.get(user, "<missing>"))
        rows.append("|".join((str(region), str(int(hour_bucket.get(user, 0))),
                               str(int(weekday_bucket.get(user, 0))), str(int(volume_bucket[position])))))
    return pd.Series(rows, index=user_index, name="matching_stratum")


def _extreme_binary_labels(values: pd.Series, low_quantile: float = 0.20, high_quantile: float = 0.80) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce")
    low, high = float(numeric.quantile(low_quantile)), float(numeric.quantile(high_quantile))
    labels = pd.Series(np.nan, index=values.index, dtype=float)
    labels.loc[numeric <= low] = 0.0
    labels.loc[numeric >= high] = 1.0
    labels = labels.dropna()
    return labels.astype(int), {
        "label_policy": "low <= p20; high >= p80",
        "low_threshold": low,
        "high_threshold": high,
        "low_count": int((labels == 0).sum()),
        "high_count": int((labels == 1).sum()),
    }


def _matched_user_mask(labels: pd.Series, strata: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    selected: list[str] = []
    group_frame = pd.DataFrame({"label": labels, "stratum": strata.reindex(labels.index).fillna("<missing>")})
    counts: Counter[str] = Counter()
    for stratum, rows in group_frame.groupby("stratum", sort=True):
        zeros = sorted(rows.index[rows["label"] == 0].astype(str))
        ones = sorted(rows.index[rows["label"] == 1].astype(str))
        count = min(len(zeros), len(ones))
        selected.extend(zeros[:count] + ones[:count])
        counts[str(stratum)] = count
    mask = labels.index.astype(str).isin(set(selected))
    return pd.Series(mask, index=labels.index), {
        "matched_users": int(mask.sum()),
        "matched_pairs": int(mask.sum() // 2),
        "strata_with_pairs": int(sum(value > 0 for value in counts.values())),
        "stratum_pair_counts": dict(counts),
    }


def _matched_opportunity_users(run_dir: Path, factor_name: str, minimum: int) -> set[str]:
    """Return users with matched preferred/nonpreferred choice opportunities.

    This protected evaluator-side population definition is part of the v2
    coverage contract. It is never available to observed-only model code.
    """
    if not factor_name.startswith("pref_"):
        return set()
    category = factor_name.removeprefix("pref_")
    latent = _read(run_dir, "truth/user_latents.csv.gz")
    choices = _read(run_dir, "truth/choices_truth.csv.gz")
    if category not in {"grocery", "restaurant", "cafe"}:
        return set()
    latent["user_id"] = latent["user_id"].astype(str)
    preferred = latent.set_index("user_id")[f"pref_{category}"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in choices.to_dict("records"):
        if int(row.get("candidate_count", 0)) < 2:
            continue
        user = str(row["user_id"])
        chosen = str(row.get("chosen_category", ""))
        counts[user]["preferred" if chosen == category else "nonpreferred"] += 1
    return {
        user for user in preferred.index.astype(str)
        if counts[user]["preferred"] >= minimum and counts[user]["nonpreferred"] >= minimum
    }


def _auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if not len(positives) or not len(negatives):
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def _balanced_accuracy(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(scores) >= 0.5
    recalls = []
    for value in (0, 1):
        present = y == value
        if not present.any():
            return None
        recalls.append(float(np.mean(prediction[present] == value)))
    return float(np.mean(recalls))


def _standardized_separation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if not np.any(y == 0) or not np.any(y == 1):
        return None
    mean_zero, mean_one = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
    within_zero = x[y == 0] - mean_zero
    within_one = x[y == 1] - mean_one
    within = math.sqrt(float((np.square(within_zero).sum() + np.square(within_one).sum()) / max(1, len(x))))
    return float(np.linalg.norm(mean_one - mean_zero) / max(within, 1e-12))


def _knn_neighbors(x: np.ndarray, k: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # The observed-history matrix contains hundreds of sparse one-hot columns.
    # Use a deterministic top-variance projection for neighborhood distances so
    # the evaluator remains bounded; classification and separation retain the
    # complete standardized observed vector.
    if x.shape[1] > 64:
        variance = np.var(x, axis=0)
        keep = np.argsort(-variance, kind="mergesort")[:64]
        x = x[:, keep]
    squared = np.maximum(0.0, (x * x).sum(axis=1)[:, None] + (x * x).sum(axis=1)[None, :] - 2.0 * x @ x.T)
    np.fill_diagonal(squared, np.inf)
    count = min(k, max(1, len(x) - 1))
    return np.argpartition(squared, kth=count - 1, axis=1)[:, :count]


def _knn_purity(y: np.ndarray, neighbors: np.ndarray) -> float | None:
    if len(y) == 0 or neighbors.size == 0:
        return None
    return float(np.mean(np.mean(np.asarray(y)[neighbors] == np.asarray(y)[:, None], axis=1)))


def _knn_subset_indices(y: np.ndarray, maximum: int = 512) -> np.ndarray:
    """Choose a deterministic, class-balanced neighborhood-evaluation subset."""
    y = np.asarray(y, dtype=int)
    if len(y) <= maximum:
        return np.arange(len(y))
    per_class = max(1, maximum // 2)
    selected: list[np.ndarray] = []
    for value in (0, 1):
        positions = np.flatnonzero(y == value)
        selected.append(positions[np.linspace(0, len(positions) - 1, min(per_class, len(positions)), dtype=int)])
    return np.sort(np.unique(np.concatenate(selected)))


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return np.clip((x - mean) / scale, -10.0, 10.0)


def _metric_bundle(
    x: np.ndarray, y: np.ndarray, scores: np.ndarray,
    neighbors: np.ndarray | None = None, standardized: np.ndarray | None = None,
) -> dict[str, Any]:
    standardized = _standardize(x) if standardized is None else standardized
    return {
        "balanced_accuracy": _balanced_accuracy(y, scores),
        "auroc": _auc(y, scores),
        "knn_purity_at_10": _knn_purity(y, neighbors) if neighbors is not None else None,
        "standardized_separation": _standardized_separation(standardized, y),
    }


def _binary_probe(x: np.ndarray, y: np.ndarray, users: Sequence[str], folds: int, alpha: float = 10.0) -> dict[str, Any]:
    assignments = np.asarray([_stable_fold(str(user), folds) for user in users])
    scores = np.full(len(y), np.nan, dtype=float)
    calibrated_scores = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        if train.sum() < 4 or test.sum() < 2 or len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        weights, mean, target_mean, scale, keep = _fit_ridge(x[train], y[train].astype(float), alpha)
        z_test = np.clip((x[test][:, keep] - mean) / scale, -10.0, 10.0)
        raw_train = 1.0 / (1.0 + np.exp(-np.clip(target_mean + np.clip((x[train][:, keep] - mean) / scale, -10.0, 10.0) @ weights, -30.0, 30.0)))
        raw_test = 1.0 / (1.0 + np.exp(-np.clip(target_mean + z_test @ weights, -30.0, 30.0)))
        threshold = float((raw_train[y[train] == 0].mean() + raw_train[y[train] == 1].mean()) / 2.0)
        scores[test] = raw_test
        calibrated_scores[test] = raw_test - threshold + 0.5
        fold_rows.append({"fold": fold, "train_users": int(len(set(np.asarray(users)[train]))), "test_users": int(len(set(np.asarray(users)[test]))), "train_calibration_threshold": threshold})
    valid = np.isfinite(scores)
    if valid.sum() < max(4, folds * 2) or len(np.unique(y[valid])) < 2:
        return {"status": "insufficient_user_disjoint_folds", "evaluated_rows": int(valid.sum())}
    return {"status": "ok", "scores": calibrated_scores, "raw_scores": scores, "valid": valid, "folds": fold_rows,
            **_metric_bundle(x[valid], y[valid], calibrated_scores[valid])}


def _cluster_bootstrap(
    x: np.ndarray, y: np.ndarray, scores: np.ndarray, clusters: Sequence[str],
    *, neighbors: np.ndarray | None, replicates: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters).astype(str)
    unique = np.asarray(sorted(set(clusters)))
    indices_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    standardized = _standardize(x)
    rows: list[dict[str, float]] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled])
        metric = _metric_bundle(x[indices], y[indices], scores[indices], None, standardized[indices])
        if neighbors is not None:
            metric["knn_purity_at_10"] = float(np.mean(np.mean(y[neighbors[indices]] == y[indices, None], axis=1)))
        required = ("balanced_accuracy", "auroc", "knn_purity_at_10", "standardized_separation") if neighbors is not None else (
            "balanced_accuracy", "auroc", "standardized_separation")
        if all(metric.get(name) is not None for name in required):
            rows.append(metric)
    result: dict[str, Any] = {"replicates": len(rows), "seed": seed, "confidence": 0.95}
    for name in ("balanced_accuracy", "auroc", "knn_purity_at_10", "standardized_separation"):
        if name == "knn_purity_at_10" and neighbors is None:
            result[name] = {"lower": None, "upper": None}
            continue
        values = [row[name] for row in rows]
        result[name] = {"lower": float(np.quantile(values, 0.025)) if values else None,
                        "upper": float(np.quantile(values, 0.975)) if values else None}
    return result


def _stratified_permutation_null(
    x: np.ndarray, y: np.ndarray, scores: np.ndarray, strata: Sequence[str],
    *, neighbors: np.ndarray | None, permutations: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata).astype(str)
    groups = [np.flatnonzero(strata == value) for value in sorted(set(strata))]
    standardized = _standardize(x)
    values = {name: [] for name in ("balanced_accuracy", "auroc", "knn_purity_at_10", "standardized_separation")}
    for _ in range(permutations):
        permuted = y.copy()
        for indices in groups:
            if len(indices) > 1:
                permuted[indices] = rng.permutation(permuted[indices])
        metric = _metric_bundle(x, permuted, scores, neighbors, standardized)
        for name in values:
            if metric.get(name) is not None:
                values[name].append(float(metric[name]))
    return {"permutations": permutations, "seed": seed,
            "p95": {name: float(np.quantile(items, 0.95)) if items else None for name, items in values.items()}}


def _gate_result(
    metrics: dict[str, Any], bootstrap: dict[str, Any], null: dict[str, Any],
    *, profile: str = "legacy",
) -> dict[str, Any]:
    if profile == "v2":
        required = {
            "balanced_accuracy": ("held_out_balanced_accuracy_min", 0.70),
            "auroc": ("held_out_auroc_min", 0.70),
        }
    elif profile == "legacy":
        required = {
            "balanced_accuracy": ("held_out_balanced_accuracy_min", 0.70),
            "auroc": ("held_out_auroc_min", 0.70),
            "knn_purity_at_10": ("matched_knn_purity_at_10_min", 0.70),
            "standardized_separation": ("standardized_within_between_separation_min", 0.20),
        }
    else:
        raise ValueError(f"unknown recoverability gate profile: {profile}")
    checks: dict[str, Any] = {}
    for name in ("balanced_accuracy", "auroc", "knn_purity_at_10", "standardized_separation"):
        value = metrics.get(name)
        lower = bootstrap.get(name, {}).get("lower")
        null_value = null.get("p95", {}).get(name)
        threshold_name, threshold = required.get(name, (None, None))
        checks[name] = {
            "value": value, "threshold": threshold, "threshold_name": threshold_name,
            "is_feasibility_gate": name in required,
            "threshold_pass": None if name not in required else bool(value is not None and value >= threshold),
            "bootstrap_lower": lower, "permutation_null_p95": null_value,
            "null_separation_pass": None if name not in required else bool(
                lower is not None and null_value is not None and lower > null_value
            ),
        }
    passed = all(
        item["threshold_pass"] and item["null_separation_pass"]
        for item in checks.values() if item["is_feasibility_gate"]
    )
    return {
        "profile": profile,
        "status": "pass" if passed else "fail",
        "checks": checks,
        "raw_feature_diagnostics_are_gates": profile == "legacy",
    }


def _score_binary_factor(
    name: str, values: pd.Series, features: pd.DataFrame, users: pd.Index, strata: pd.Series,
    registry_factor: dict[str, Any], *, folds: int, bootstrap_replicates: int,
    permutation_count: int, seed: int, gate_profile: str = "legacy",
    eligible_users: set[str] | None = None, probe_alpha: float = 10.0,
    feature_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if feature_override is not None:
        if not feature_override.index.equals(users) or not feature_override.index.is_unique:
            raise ValueError(f"feature override for {name} must have the exact evaluator user index")
        if not np.isfinite(feature_override.to_numpy(dtype=float)).all():
            raise ValueError(f"feature override for {name} contains non-finite values")
    labels, label_meta = _extreme_binary_labels(values)
    labels = labels.reindex(users).dropna().astype(int)
    if eligible_users:
        labels = labels.loc[labels.index.astype(str).isin(eligible_users)]
    if labels.empty or labels.nunique() < 2:
        return {"factor": name, "status": "inconclusive", "reason": "insufficient high/low truth labels", "label_definition": label_meta}
    matched_mask, matching = _matched_user_mask(labels, strata)
    if matching["matched_users"] < 20:
        return {"factor": name, "status": "inconclusive", "reason": "insufficient matched users", "label_definition": label_meta, "matching": matching}
    selected_features = feature_override if feature_override is not None else features
    selected_features = selected_features.reindex(labels.index)
    if selected_features.isna().any().any():
        raise ValueError(f"feature override for {name} is missing eligible user rows")
    x_all = selected_features.to_numpy(dtype=float)
    y_all = labels.to_numpy(dtype=int)
    user_values = labels.index.astype(str).to_numpy()
    probe = _binary_probe(x_all, y_all, user_values, folds, alpha=probe_alpha)
    if probe.get("status") != "ok":
        return {"factor": name, "status": "inconclusive", "reason": probe.get("status"), "label_definition": label_meta, "matching": matching}
    selected = matched_mask.to_numpy(dtype=bool) & probe["valid"]
    x = x_all[selected]
    y = y_all[selected]
    scores = probe["scores"][selected]
    cluster_ids = user_values[selected]
    selected_strata = strata.reindex(labels.index).to_numpy()[selected]
    metrics = _metric_bundle(x, y, scores)
    bootstrap = _cluster_bootstrap(x, y, scores, cluster_ids, neighbors=None,
                                   replicates=bootstrap_replicates, seed=seed)
    null = _stratified_permutation_null(x, y, scores, selected_strata, neighbors=None,
                                        permutations=permutation_count, seed=seed + 1)
    knn_subset = _knn_subset_indices(y)
    knn_x = _standardize(x[knn_subset])
    neighbors = _knn_neighbors(knn_x, 10)
    metrics["knn_purity_at_10"] = _knn_purity(y[knn_subset], neighbors)
    knn_bootstrap = _cluster_bootstrap(
        knn_x, y[knn_subset], scores[knn_subset], cluster_ids[knn_subset], neighbors=neighbors,
        replicates=bootstrap_replicates, seed=seed + 2,
    )
    knn_null = _stratified_permutation_null(
        knn_x, y[knn_subset], scores[knn_subset], selected_strata[knn_subset], neighbors=neighbors,
        permutations=permutation_count, seed=seed + 3,
    )
    bootstrap["knn_purity_at_10"] = knn_bootstrap["knn_purity_at_10"]
    null["p95"]["knn_purity_at_10"] = knn_null["p95"]["knn_purity_at_10"]
    gate = _gate_result(metrics, bootstrap, null, profile=gate_profile)
    return {
        "factor": name, "factor_class": registry_factor.get("factor_class"),
        "shortcut_risk": bool(registry_factor.get("shortcut_risk", False)),
        "status": gate["status"], "label_definition": label_meta,
        "matching": matching, "held_out": {key: value for key, value in probe.items() if key not in {"scores", "raw_scores", "valid"}},
        "evaluated_users": int(len(set(cluster_ids))), "evaluated_rows": int(len(y)),
        "knn_evaluated_rows": int(len(knn_subset)),
        "metrics": metrics, "cluster_bootstrap": bootstrap,
        "stratified_permutation_null": null, "gate": gate,
    }


def evaluate_sustained_preference(
    run_dir: Path, registry: dict[str, Any], *, folds: int, bootstrap_replicates: int,
    permutation_count: int, seed: int, gate_profile: str = "legacy",
    opportunity_minimum: int | None = None, probe_alpha: float | None = None,
    feature_overrides: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    users, features, metadata = _observed_history_matrix(run_dir)
    strata = _history_matching_strata(run_dir, users)
    latent = _read(run_dir, "truth/user_latents.csv.gz").assign(user_id=lambda frame: frame["user_id"].astype(str)).set_index("user_id")
    factors: list[dict[str, Any]] = []
    for factor in registry["factors"]:
        if not factor.get("eligible_for_sustained_preference_benchmark"):
            continue
        name = factor["name"]
        if name not in latent.columns:
            factors.append({"factor": name, "status": "inconclusive", "reason": "declared truth column is absent"})
            continue
        eligible_users = (
            _matched_opportunity_users(run_dir, name, int(opportunity_minimum))
            if gate_profile == "v2" and opportunity_minimum is not None else None
        )
        factors.append(_score_binary_factor(name, latent[name].reindex(users), features, users, strata,
                                            factor, folds=folds, bootstrap_replicates=bootstrap_replicates,
                                            permutation_count=permutation_count, seed=seed + len(factors) * 17,
                                            gate_profile=gate_profile, eligible_users=eligible_users,
                                            probe_alpha=float(probe_alpha if probe_alpha is not None else (1000.0 if gate_profile == "v2" else 10.0)),
                                            feature_override=(feature_overrides or {}).get(name)))
    shortcut_factor = next((factor for factor in registry["factors"] if factor["name"] == "digital_engagement"), None)
    shortcut = None
    if shortcut_factor is not None and shortcut_factor["name"] in latent.columns:
        shortcut = _score_binary_factor(shortcut_factor["name"], latent[shortcut_factor["name"]].reindex(users), features, users, strata,
                                        shortcut_factor, folds=folds, bootstrap_replicates=bootstrap_replicates,
                                        permutation_count=permutation_count, seed=seed + 999,
                                        gate_profile=gate_profile)
        shortcut["interpretation"] = "diagnostic shortcut-risk only; excluded from the aggregate sustained-preference gate"
    eligible_statuses = [item["status"] for item in factors]
    aggregate = "pass" if factors and all(status == "pass" for status in eligible_statuses) else (
        "fail" if any(status == "fail" for status in eligible_statuses) else "inconclusive")
    return {"status": aggregate, "feature_metadata": metadata, "matching_strata": registry["label_policy"]["matching_strata"],
            "factors": factors, "shortcut_risk_factor": shortcut,
            "aggregate_rule": "all registry-eligible non-shortcut factors must pass; no averaging"}


def _schedule_pair_examples(pair_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    integrity = require_passing_pair_integrity(pair_dir / "pair_manifest.json")
    manifest = json.loads((pair_dir / "pair_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("intervention_type") != "temporary_schedule_shift_v1":
        raise ValueError("Temporary schedule evaluator requires temporary_schedule_shift_v1")
    reference = Path(manifest["reference"]["run_dir"])
    intervention = Path(manifest["intervention"]["run_dir"])
    truth_path = intervention / "truth/temporary_schedule_shift_truth.csv.gz"
    affected_path = intervention / "truth/temporary_schedule_shift_events.csv.gz"
    if not truth_path.is_file() or not affected_path.is_file():
        raise FileNotFoundError("Temporary schedule intervention records are required")
    truth = _read(intervention, "truth/temporary_schedule_shift_truth.csv.gz")
    affected = _read(intervention, "truth/temporary_schedule_shift_events.csv.gz")
    required_truth = {"user_id", "selected", "applied", "change_start_time", "change_end_time"}
    if not required_truth.issubset(truth.columns) or affected.empty:
        raise ValueError("Temporary schedule intervention records are missing or empty")
    applied = truth.loc[(truth["selected"].astype(int) == 1) & (truth["applied"].astype(int) == 1)].copy()
    if applied.empty:
        raise ValueError("Temporary schedule intervention records contain no applied users")
    starts = pd.to_datetime(applied["change_start_time"], utc=True).dropna().unique()
    ends = pd.to_datetime(applied["change_end_time"], utc=True).dropna().unique()
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("Temporary schedule interval must be one finite common interval")
    start, end = pd.Timestamp(starts[0]), pd.Timestamp(ends[0])
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    reference_events = _read(reference, "observed/observed_events.csv.gz")
    intervention_events = _read(intervention, "observed/observed_events.csv.gz")
    for frame in (reference_events, intervention_events):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    reference_by_user = {str(user): rows for user, rows in reference_events.groupby("user_id", sort=False)}
    intervention_by_user = {str(user): rows for user, rows in intervention_events.groupby("user_id", sort=False)}
    rows: list[dict[str, Any]] = []
    feature_events: list[pd.DataFrame] = []
    applied_users = set(applied["user_id"].astype(str))
    local_start = start.tz_convert("Asia/Tokyo")
    local_end = end.tz_convert("Asia/Tokyo")
    for user in sorted(applied_users):
        for local_day in pd.date_range(start=local_start.normalize(), end=local_end.normalize() - pd.Timedelta(1, unit="D"), freq="D"):
            day = local_day.tz_convert("UTC")
            window_end = day + pd.Timedelta(1, unit="D")
            if day < start or window_end > end:
                continue
            sample_parts = []
            for side, source_by_user, label in (("reference", reference_by_user, 0), ("intervention", intervention_by_user, 1)):
                source = source_by_user.get(user, pd.DataFrame())
                selected = source.loc[(source["timestamp"] >= day) & (source["timestamp"] < window_end)].copy() if not source.empty else source
                if selected.empty:
                    sample_parts = []
                    break
                # Filtering is timestamp-group atomic: every row at a timestamp
                # enters the same window, and no row-level tie-breaker is used.
                selected["sample_id"] = f"{user}|{day.date().isoformat()}|{side}"
                sample_parts.append((selected, side, label))
            if len(sample_parts) != 2:
                continue
            for selected, side, label in sample_parts:
                feature_events.append(selected)
                rows.append({"sample_id": selected["sample_id"].iloc[0], "user_id": user,
                             "calendar_day": local_day.date().isoformat(), "side": side, "label": label})
    if not rows:
        raise ValueError("No complete same-user schedule episode windows were found")
    return pd.DataFrame(rows), {
        "pair_integrity": integrity,
        "selected_users": int(len(applied_users)),
        "affected_event_count": int(len(affected)),
        "affected_event_users": int(affected["user_id"].astype(str).nunique()),
        "interval_start": start.isoformat(), "interval_end": end.isoformat(),
        "interval_start_asia_tokyo": start.tz_convert("Asia/Tokyo").isoformat(),
        "interval_end_asia_tokyo": end.tz_convert("Asia/Tokyo").isoformat(),
        "boundary_policy": "half-open [start,end); windows must be wholly contained and timestamp groups remain atomic",
        "feature_events": pd.concat(feature_events, ignore_index=True),
    }


def evaluate_temporary_schedule(
    pair_dir: Path, *, folds: int, bootstrap_replicates: int,
    permutation_count: int, seed: int, gate_profile: str = "legacy", probe_alpha: float | None = None,
) -> dict[str, Any]:
    examples, metadata = _schedule_pair_examples(pair_dir)
    if metadata["selected_users"] < 100 or len(examples) < 200:
        return {"status": "inconclusive", "reason": "intervention cohort is too small for the frozen evaluation protocol",
                "selected_users": metadata["selected_users"], "eligible_episode_windows": int(len(examples)),
                "metadata": {key: value for key, value in metadata.items() if key != "feature_events"}}
    events = metadata.pop("feature_events")
    events["user_id"] = events["sample_id"]
    sample_index = pd.Index(sorted(examples["sample_id"]), name="user_id")
    features, _ = _event_features(events, sample_index)
    features = _numeric_frame(features.reindex(sample_index).fillna(0.0))
    examples = examples.set_index("sample_id").reindex(sample_index)
    x = features.to_numpy(dtype=float)
    y = examples["label"].to_numpy(dtype=int)
    clusters = examples["user_id"].astype(str).to_numpy()
    strata = examples["calendar_day"].astype(str).to_numpy()
    probe = _binary_probe(x, y, clusters, folds, alpha=float(probe_alpha if probe_alpha is not None else 10.0))
    if probe.get("status") != "ok":
        return {"status": "inconclusive", "reason": probe.get("status"),
                "metadata": {key: value for key, value in metadata.items() if key != "feature_events"}}
    valid = probe["valid"]
    x, y, scores = x[valid], y[valid], probe["scores"][valid]
    clusters, strata = clusters[valid], strata[valid]
    x = _standardize(x)
    metrics = _metric_bundle(x, y, scores)
    bootstrap = _cluster_bootstrap(x, y, scores, clusters, neighbors=None,
                                   replicates=bootstrap_replicates, seed=seed)
    null = _stratified_permutation_null(x, y, scores, strata, neighbors=None,
                                        permutations=permutation_count, seed=seed + 1)
    knn_subset = _knn_subset_indices(y)
    neighbors = _knn_neighbors(x[knn_subset], 10)
    metrics["knn_purity_at_10"] = _knn_purity(y[knn_subset], neighbors)
    knn_bootstrap = _cluster_bootstrap(
        x[knn_subset], y[knn_subset], scores[knn_subset], clusters[knn_subset], neighbors=neighbors,
        replicates=bootstrap_replicates, seed=seed + 2,
    )
    knn_null = _stratified_permutation_null(
        x[knn_subset], y[knn_subset], scores[knn_subset], strata[knn_subset], neighbors=neighbors,
        permutations=permutation_count, seed=seed + 3,
    )
    bootstrap["knn_purity_at_10"] = knn_bootstrap["knn_purity_at_10"]
    null["p95"]["knn_purity_at_10"] = knn_null["p95"]["knn_purity_at_10"]
    gate = _gate_result(metrics, bootstrap, null, profile=gate_profile)
    return {"status": gate["status"], "metadata": {key: value for key, value in metadata.items() if key != "feature_events"},
            "eligible_episode_windows": int(len(y)), "held_out": {key: value for key, value in probe.items() if key not in {"scores", "raw_scores", "valid"}},
            "metrics": metrics, "knn_evaluated_rows": int(len(knn_subset)), "cluster_bootstrap": bootstrap,
            "within_user_permutation_null": null, "gate": gate}


def evaluate_track_a_recoverability(
    run_dir: Path, registry_path: Path, sustained_pair_dir: Path, temporary_pair_dir: Path,
    *, folds: int, bootstrap_replicates: int, permutation_count: int, seed: int,
    gate_profile: str = "legacy",
) -> dict[str, Any]:
    registry = _load_factor_registry(registry_path)
    sustained_authentication = require_passing_pair_integrity(sustained_pair_dir / "pair_manifest.json")
    sustained = evaluate_sustained_preference(run_dir, registry, folds=folds,
                                              bootstrap_replicates=bootstrap_replicates,
                                              permutation_count=permutation_count, seed=seed,
                                              gate_profile=gate_profile)
    temporary = evaluate_temporary_schedule(temporary_pair_dir, folds=folds,
                                            bootstrap_replicates=bootstrap_replicates,
                                            permutation_count=permutation_count, seed=seed + 2000,
                                            gate_profile=gate_profile)
    overall = "pass" if sustained["status"] == "pass" and temporary["status"] == "pass" else (
        "fail" if sustained["status"] == "fail" or temporary["status"] == "fail" else "inconclusive")
    return {
        "schema_version": "geoembeddings-track-a-recoverability/1.0",
        "status": overall,
        "gates": V2_RECOVERABILITY_GATES if gate_profile == "v2" else RECOVERABILITY_GATES,
        "factor_registry": {"path": str(registry_path), "schema_version": registry["schema_version"]},
        "authenticated_pair_inputs": {
            "sustained_preference": {"pair_dir": str(sustained_pair_dir), "integrity_status": "passed",
                                      "pair_integrity": sustained_authentication},
            "temporary_schedule": {"pair_dir": str(temporary_pair_dir), "integrity_status": "validated_by_evaluator"},
        },
        "sustained_preference": sustained,
        "temporary_schedule": temporary,
        "decision_rule": "sustained preference AND temporary schedule state must pass; feasibility uses BA/AUROC and bootstrap-vs-stratified-permutation evidence; raw-feature purity/separation are reported diagnostics only in v2; no aggregate averaging; unavailable metrics are inconclusive",
    }


def _semantic_rows(frame: pd.DataFrame) -> Counter[tuple[str, ...]]:
    clean = frame.fillna("<NA>").astype(str)
    return Counter(tuple(row) for row in clean.itertuples(index=False, name=None))


def _changed_users(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any] | None:
    if "user_id" not in left or "user_id" not in right:
        return None
    def signatures(frame: pd.DataFrame) -> dict[str, str]:
        result: dict[str, str] = {}
        for user, rows in frame.fillna("<NA>").astype(str).groupby("user_id", sort=True):
            payload = rows.sort_values(list(rows.columns)).to_csv(index=False).encode("utf-8")
            result[str(user)] = hashlib.sha256(payload).hexdigest()
        return result
    a, b = signatures(left), signatures(right)
    users = sorted(set(a) | set(b))
    changed = [user for user in users if a.get(user) != b.get(user)]
    return {"users": len(users), "changed_users": len(changed), "changed_share": len(changed) / max(1, len(users))}


def compare_pair(pair: dict[str, Any]) -> dict[str, Any]:
    reference = Path(pair["reference_run"])
    changed = Path(pair["intervention_run"])
    table_reports: dict[str, Any] = {}
    observed_dir = reference / "observed"
    for path in sorted(observed_dir.glob("*.csv.gz")):
        relative = f"observed/{path.name}"
        left = _read(reference, relative)
        right = _read(changed, relative)
        a, b = _semantic_rows(left), _semantic_rows(right)
        removed = sum((a - b).values())
        added = sum((b - a).values())
        table_reports[path.name] = {
            "reference_rows": len(left),
            "intervention_rows": len(right),
            "semantically_identical": a == b,
            "removed_rows": removed,
            "added_rows": added,
            "user_effect": _changed_users(left, right),
        }
    latent_left = _read(reference, "truth/user_latents.csv.gz")
    latent_right = _read(changed, "truth/user_latents.csv.gz")
    events = table_reports.get("observed_events.csv.gz", {})
    diagnostic_path = Path(pair["pair_dir"]) / "behavioral_diagnostics.json"
    behavioral = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else None
    return {
        "intervention": pair["intervention"],
        "latent_table_identical": _semantic_rows(latent_left) == _semantic_rows(latent_right),
        "observed_events_identical": events.get("semantically_identical"),
        "observed_event_user_effect": events.get("user_effect"),
        "tables": table_reports,
        "repository_behavioral_diagnostics": behavioral,
    }


def _solve_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if x.shape[1] > x.shape[0]:
        return x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(x.shape[0]), y)
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def _ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    reciprocal: list[float] = []
    top1: list[float] = []
    ndcg5: list[float] = []
    for _, rows in frame.groupby("request_id"):
        if rows["label"].sum() < 1:
            continue
        ranked = rows.sort_values([score_column, "poi_id"], ascending=[False, True]).reset_index(drop=True)
        positions = np.flatnonzero(ranked["label"].to_numpy() > 0)
        if not len(positions):
            continue
        rank = int(positions[0]) + 1
        top1.append(float(rank == 1))
        reciprocal.append(1.0 / rank)
        ndcg5.append(1.0 / math.log2(rank + 1) if rank <= 5 else 0.0)
    return {
        "requests": len(reciprocal),
        "top1_hit_rate": float(np.mean(top1)) if top1 else None,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "ndcg_at_5": float(np.mean(ndcg5)) if ndcg5 else None,
    }


def candidate_only_control(run_dir: Path, alpha: float) -> dict[str, Any]:
    requests = _read(run_dir, "observed/recommendation_requests.csv.gz", required=False)
    impressions = _read(run_dir, "observed/impressions.csv.gz", required=False)
    interactions = _read(run_dir, "observed/interactions.csv.gz", required=False)
    catalog = _read(run_dir, "observed/poi_catalog.csv.gz", required=False)
    if any(frame.empty for frame in (requests, impressions, interactions, catalog)):
        return {"status": "unavailable", "reason": "Recommendation tables are missing or empty"}

    requests = requests.copy()
    requests["request_timestamp"] = pd.to_datetime(requests["request_timestamp"], utc=True)
    data = impressions.merge(requests, on="request_id", how="inner", suffixes=("_impression", "_request"))
    data = data.merge(catalog, on="poi_id", how="inner", suffixes=("_request", "_poi"))
    data = data[pd.to_numeric(data["is_available"], errors="coerce").fillna(0).astype(int) == 1].copy()
    data = data.reset_index(drop=True)
    positives = set(zip(interactions["request_id"].astype(str), interactions["poi_id"].astype(str)))
    data["request_id"] = data["request_id"].astype(str)
    data["poi_id"] = data["poi_id"].astype(str)
    data["label"] = [int((request, poi) in positives) for request, poi in zip(data["request_id"], data["poi_id"])]
    positive_requests = sorted(data.loc[data["label"] == 1, "request_id"].unique())
    if len(positive_requests) < 20:
        return {"status": "insufficient_requests", "positive_requests": len(positive_requests)}

    numeric_names = [
        "travel_time_minutes", "is_shown", "shown_rank", "price_level",
        "family_suitability", "local_popularity", "latitude_request", "longitude_request",
        "latitude_poi", "longitude_poi",
    ]
    numeric = pd.DataFrame(index=data.index)
    for column in numeric_names:
        numeric[column] = pd.to_numeric(data[column], errors="coerce") if column in data else 0.0
    numeric["shown_rank_missing"] = numeric["shown_rank"].isna().astype(float)
    numeric = numeric.fillna(0.0)
    if {"latitude_request", "latitude_poi"}.issubset(numeric):
        numeric["abs_latitude_delta"] = (numeric["latitude_request"] - numeric["latitude_poi"]).abs()
        numeric["abs_longitude_delta"] = (numeric["longitude_request"] - numeric["longitude_poi"]).abs()
    categorical_columns = [column for column in ("category", "environment", "context_source", "region_id_request", "region_id_poi") if column in data]
    categorical = pd.get_dummies(data[categorical_columns].fillna("<missing>").astype(str), dtype=float)
    design = pd.concat([numeric, categorical], axis=1).to_numpy(dtype=float)

    request_order = requests.sort_values(["request_timestamp", "request_id"])["request_id"].astype(str).tolist()
    request_order = [request for request in request_order if request in set(positive_requests)]
    split = max(1, min(len(request_order) - 1, int(round(0.7 * len(request_order)))))
    train_ids, test_ids = set(request_order[:split]), set(request_order[split:])
    train_rows = data["request_id"].isin(train_ids).to_numpy()
    mean, scale = design[train_rows].mean(axis=0), design[train_rows].std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (design - mean) / scale

    pair_differences: list[np.ndarray] = []
    for request, positions in data[train_rows].groupby("request_id").groups.items():
        indices = np.asarray(list(positions), dtype=int)
        positive = indices[data.loc[indices, "label"].to_numpy() > 0]
        negative = indices[data.loc[indices, "label"].to_numpy() == 0]
        for pos in positive:
            for neg in negative:
                difference = z[pos] - z[neg]
                pair_differences.extend([difference, -difference])
    if not pair_differences:
        return {"status": "insufficient_training_pairs"}
    pair_x = np.stack(pair_differences)
    pair_y = np.tile(np.asarray([1.0, -1.0]), len(pair_differences) // 2)
    weights = _solve_ridge(pair_x, pair_y, alpha)
    data["candidate_only_score"] = z @ weights
    data["nearest_score"] = -pd.to_numeric(data["travel_time_minutes"], errors="coerce").fillna(np.inf)
    data["popularity_score"] = pd.to_numeric(data["local_popularity"], errors="coerce").fillna(0.0)
    data["shown_score"] = (
        1000.0 * pd.to_numeric(data["is_shown"], errors="coerce").fillna(0.0)
        - pd.to_numeric(data["shown_rank"], errors="coerce").fillna(999.0)
    )
    test = data[data["request_id"].isin(test_ids)].copy()
    models = {
        name: _ranking_metrics(test, column)
        for name, column in {
            "candidate_only_linear": "candidate_only_score",
            "nearest": "nearest_score",
            "popularity": "popularity_score",
            "shown_rank": "shown_score",
        }.items()
    }
    saturated = any(
        (metrics.get("top1_hit_rate") or 0.0) >= 0.98
        for metrics in models.values()
    )
    return {
        "status": "ok",
        "protocol": "chronological 70/30 request split; public candidate/request fields only; pairwise ridge",
        "train_requests": len(train_ids),
        "test_requests": len(test_ids),
        "features": design.shape[1],
        "models": models,
        "saturated": saturated,
        "interpretation": (
            "At least one public-feature control is saturated; exclude synthetic ranking from model selection until the response generator is repaired."
            if saturated else
            "Any later embedding ranker must beat the candidate-only control on the same authenticated request split."
        ),
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Recoverability gate",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Git revision: `{report.get('git_revision') or 'unknown'}`",
        f"- Audit run: `{report['audit_run']}`",
        f"- Users/events: {report['feature_metadata']['users']} / {report['feature_metadata']['events']}",
        "",
        "## Trait decisions",
        "",
        "| Trait | Decision | Volume R² | Profile R² | Current events R² | Enriched R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for trait, item in report["oracle_probes"].items():
        def r2(name: str) -> str:
            value = item.get(name, {}).get("r2") if isinstance(item, dict) else None
            return "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {trait} | {item.get('decision', item.get('status', 'unknown'))} | "
            f"{r2('volume_only')} | {r2('profile_only')} | {r2('current_events')} | {r2('enriched_observables')} |"
        )
    lines.extend(["", "## Matched interventions", ""])
    if report["matched_interventions"]:
        lines.extend([
            "| Intervention | Latents identical | Observed events identical | Changed users |",
            "|---|---:|---:|---:|",
        ])
        for item in report["matched_interventions"]:
            effect = item.get("observed_event_user_effect") or {}
            lines.append(
                f"| {item['intervention']} | {item['latent_table_identical']} | "
                f"{item['observed_events_identical']} | {effect.get('changed_users', '—')}/{effect.get('users', '—')} |"
            )
    else:
        lines.append("No matched pairs were generated; this run contains observational evidence only.")
    lines.extend(["", "## Candidate-only ranking control", ""])
    ranking = report["candidate_only_ranking"]
    if ranking.get("status") == "ok":
        lines.extend([
            "| Model | Top-1 | MRR | nDCG@5 |",
            "|---|---:|---:|---:|",
        ])
        for name, metrics in ranking["models"].items():
            lines.append(
                f"| {name} | {metrics['top1_hit_rate']:.3f} | {metrics['mrr']:.3f} | {metrics['ndcg_at_5']:.3f} |"
            )
    else:
        lines.append(f"Status: `{ranking.get('status')}` — {ranking.get('reason', '')}")
    lines.extend([
        "",
        "## Track A matched recoverability",
        "",
        f"Overall status: **{report.get('track_a_recoverability', {}).get('status', 'unavailable')}**.",
        "The frozen gate requires every eligible non-shortcut sustained-preference factor and the temporary schedule-state block to pass; results are not averaged.",
        "",
        "| Block | Status |",
        "|---|---|",
        f"| Sustained preference | {report.get('track_a_recoverability', {}).get('sustained_preference', {}).get('status', '—')} |",
        f"| Temporary schedule state | {report.get('track_a_recoverability', {}).get('temporary_schedule', {}).get('status', '—')} |",
        "",
        "## Decision rule",
        "",
        "- Do not grade a representation on `exclude_until_repaired` traits.",
        "- Treat high `volume_only` R² as an observation-process shortcut, not semantic recovery.",
        "- Require the new model to beat `current_events` on recoverable long-horizon probes.",
        "- If a public-feature ranking control is saturated, exclude synthetic ranking from model selection until its response generator is repaired.",
        "- Synthetic protected truth is evaluator-only; the new model must train from observables so the procedure transfers to real data.",
        "- Do not implement a model or open Track B unless both matched Track A blocks pass.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation/kanto_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-run", type=Path, help="Analyze this run instead of generating matched pairs")
    parser.add_argument("--factor-registry", type=Path, default=DEFAULT_FACTOR_REGISTRY)
    parser.add_argument("--sustained-pair-dir", type=Path, default=DEFAULT_SUSTAINED_PAIR)
    parser.add_argument("--temporary-schedule-pair-dir", type=Path, default=DEFAULT_TEMPORARY_SCHEDULE_PAIR)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--ranking-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--recoverability-permutations", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    parser.add_argument(
        "--interventions", nargs="+", default=list(DEFAULT_INTERVENTIONS),
        choices=("sustained-preference", "temporary-trip", "schedule-shift", "exposure", "opportunity", "observation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "recoverability_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.existing_run:
        audit_run = args.existing_run.expanduser().resolve()
        pairs: list[dict[str, Any]] = []
    else:
        args.config = args.config.expanduser().resolve()
        audit_run, pairs = generate_pairs(args, output_dir)

    users, feature_sets, feature_metadata = build_feature_sets(audit_run)
    probe_rows, probes = oracle_probes(
        audit_run, users, feature_sets,
        folds=args.folds,
        alphas=args.ridge_alphas,
        permutations=args.permutations,
        seed=args.seed,
    )
    matched = [compare_pair(pair) for pair in pairs]
    ranking = candidate_only_control(audit_run, args.ranking_alpha)
    recoverability = evaluate_track_a_recoverability(
        audit_run,
        args.factor_registry.expanduser().resolve(),
        args.sustained_pair_dir.expanduser().resolve(),
        args.temporary_schedule_pair_dir.expanduser().resolve(),
        folds=args.folds,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_count=args.recoverability_permutations,
        seed=args.seed,
    )
    report = {
        "schema_version": SCRIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(),
        "audit_run": str(audit_run),
        "information_boundary": {
            "feature_construction": "observed/ only",
            "oracle_labels": "truth/user_latents.csv.gz opened only inside this evaluator script",
            "ranking_control": "observed/ only",
        },
        "protocol": {
            "users": args.users if not args.existing_run else feature_metadata["users"],
            "days": args.days if not args.existing_run else None,
            "seed": args.seed,
            "stable_user_folds": args.folds,
            "ridge_alphas": args.ridge_alphas,
            "ranking_alpha": args.ranking_alpha,
            "permutations": args.permutations,
            "recoverability_permutations": args.recoverability_permutations,
            "cluster_bootstrap_replicates": args.bootstrap_replicates,
            "interventions": [pair["intervention"] for pair in pairs],
        },
        "feature_metadata": feature_metadata,
        "structural_audit": STRUCTURAL_AUDIT,
        "oracle_probes": probes,
        "matched_interventions": matched,
        "candidate_only_ranking": ranking,
        "track_a_recoverability": recoverability,
        "limitations": [
            "Oracle probes are observational predictability tests, not causal identification.",
            "Matched interventions cover only mechanisms already implemented in the simulator.",
            "The structural audit is commit-specific and must be repeated after DGP changes.",
            "Real-data acceptance must use chronological downstream outcomes and cannot rely on protected synthetic labels.",
            "Recoverability probes are evaluator-only and do not authorize model implementation when either Track A factor block fails or is inconclusive.",
        ],
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(probe_rows).to_csv(output_dir / "oracle_probes.csv", index=False)
    summary = _summary_markdown(report)
    (output_dir / "recoverability_summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
