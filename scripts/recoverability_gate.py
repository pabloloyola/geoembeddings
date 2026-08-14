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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_SCHEMA = "geoembeddings-recoverability-gate/1.1"
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
        "## Decision rule",
        "",
        "- Do not grade a representation on `exclude_until_repaired` traits.",
        "- Treat high `volume_only` R² as an observation-process shortcut, not semantic recovery.",
        "- Require the new model to beat `current_events` on recoverable long-horizon probes.",
        "- If a public-feature ranking control is saturated, exclude synthetic ranking from model selection until its response generator is repaired.",
        "- Synthetic protected truth is evaluator-only; the new model must train from observables so the procedure transfers to real data.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation/kanto_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-run", type=Path, help="Analyze this run instead of generating matched pairs")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--ranking-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=5)
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
            "interventions": [pair["intervention"] for pair in pairs],
        },
        "feature_metadata": feature_metadata,
        "structural_audit": STRUCTURAL_AUDIT,
        "oracle_probes": probes,
        "matched_interventions": matched,
        "candidate_only_ranking": ranking,
        "limitations": [
            "Oracle probes are observational predictability tests, not causal identification.",
            "Matched interventions cover only mechanisms already implemented in the simulator.",
            "The structural audit is commit-specific and must be repeated after DGP changes.",
            "Real-data acceptance must use chronological downstream outcomes and cannot rely on protected synthetic labels.",
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
