"""Evaluator-only mobility diagnostics that back the Kanto validation plots.

This module accepts protected tables only through an explicit ``truth``
argument.  It is intentionally not imported by preparation or model code.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .simulator import haversine_km

METRICS = (
    "consecutive_stop_distance_km", "elapsed_time_hours", "straight_line_speed_kmh",
    "daily_displacement_km", "radius_of_gyration_km", "unique_location_count",
    "home_work_distance_km", "events_per_day", "gps_error_m", "missing_event_rate",
    "region_transitions",
)


def _distance(a_lat: Any, a_lon: Any, b_lat: Any, b_lon: Any) -> float:
    values = pd.to_numeric(pd.Series([a_lat, a_lon, b_lat, b_lon]), errors="coerce")
    if values.isna().any():
        return float("nan")
    return haversine_km(*map(float, values))


def _summarize(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    return {
        "count": int(valid.size), "missing_count": int(numeric.isna().sum()),
        "missing_rate": float(numeric.isna().mean()) if len(numeric) else None,
        "mean": float(valid.mean()) if len(valid) else None,
        "p50": float(valid.quantile(.5)) if len(valid) else None,
        "p95": float(valid.quantile(.95)) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
    }


def _median(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def calculate_mobility_diagnostics(
    events: pd.DataFrame,
    users: pd.DataFrame,
    *,
    truth: Mapping[str, pd.DataFrame] | None = None,
    timezone: str = "Asia/Tokyo",
    unique_location_decimals: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-user metrics and transition rows in local-time order."""
    required = {"user_id", "timestamp", "latitude", "longitude", "region_id"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"events missing diagnostic columns: {sorted(missing)}")
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise").dt.tz_convert(timezone)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.sort_values(["user_id", "timestamp"], kind="stable")
    transitions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    demographics = users.set_index("user_id")
    truth_latents = None if truth is None else truth.get("latents")
    truth_observation = None if truth is None else truth.get("observation")
    gps_by_user: dict[str, list[float]] = {}
    if truth is not None and truth.get("trajectories") is not None:
        passive = frame.loc[frame.get("observation_mode", "").eq("passive")].copy()
        trajectories = truth["trajectories"].copy()
        trajectories["timestamp"] = pd.to_datetime(trajectories["timestamp"], utc=True).dt.tz_convert(timezone)
        for user_id, group in passive.groupby("user_id"):
            candidates = trajectories.loc[trajectories.user_id.eq(user_id)].sort_values("timestamp")
            if candidates.empty:
                continue
            matched = pd.merge_asof(group.sort_values("timestamp"), candidates, on="timestamp", direction="nearest", tolerance=timedelta(minutes=15))
            gps_by_user[user_id] = [_distance(r.latitude, r.longitude, r.true_latitude, r.true_longitude) * 1000 for r in matched.itertuples()]
    for user_id in users["user_id"].astype(str):
        group = frame.loc[frame.user_id.astype(str).eq(user_id)]
        valid = group.dropna(subset=["latitude", "longitude"])
        distances: list[float] = []
        elapsed: list[float] = []
        speeds: list[float] = []
        region_changes = 0
        for previous, current in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            hours = (current.timestamp - previous.timestamp).total_seconds() / 3600
            distance = _distance(previous.latitude, previous.longitude, current.latitude, current.longitude)
            region_changed = pd.notna(previous.region_id) and pd.notna(current.region_id) and previous.region_id != current.region_id
            region_changes += int(region_changed)
            transitions.append({"user_id": user_id, "timestamp": current.timestamp.isoformat(), "distance_km": distance,
                                "elapsed_hours": hours, "speed_kmh": distance / hours if hours > 0 and np.isfinite(distance) else np.nan,
                                "zero_duration": hours == 0, "region_transition": region_changed})
            distances.append(distance); elapsed.append(hours)
            speeds.append(distance / hours if hours > 0 and np.isfinite(distance) else np.nan)
        daily = valid.assign(local_date=valid.timestamp.dt.date).groupby("local_date", sort=True)
        daily_displacement = [sum(_distance(a.latitude, a.longitude, b.latitude, b.longitude) for a, b in zip(day.iloc[:-1].itertuples(), day.iloc[1:].itertuples())) for _, day in daily]
        if len(valid):
            center_lat, center_lon = valid.latitude.mean(), valid.longitude.mean()
            radius = float(np.sqrt(np.mean([_distance(r.latitude, r.longitude, center_lat, center_lon) ** 2 for r in valid.itertuples()])))
        else:
            radius = np.nan
        demo = demographics.loc[user_id]
        record = {"user_id": user_id, "age_group": demo.get("age_group"), "household_type": demo.get("household_type"),
                  "coordinate_missing_event_count": int(len(group) - len(valid)),
                  "consecutive_stop_distance_km": _median(distances),
                  "elapsed_time_hours": _median(elapsed),
                  "straight_line_speed_kmh": _median(speeds),
                  "daily_displacement_km": np.mean(daily_displacement) if daily_displacement else np.nan,
                  "radius_of_gyration_km": radius,
                  "unique_location_count": float(len(valid[["latitude", "longitude"]].round(unique_location_decimals).drop_duplicates())) if len(valid) else np.nan,
                  "events_per_day": len(group) / max(1, group.timestamp.dt.date.nunique()) if len(group) else 0.0,
                  "region_transitions": float(region_changes), "gps_error_m": _median(gps_by_user.get(user_id, []))}
        if truth_latents is not None:
            latent = truth_latents.loc[truth_latents.user_id.astype(str).eq(user_id)]
            record["home_work_distance_km"] = _distance(latent.iloc[0].home_latitude, latent.iloc[0].home_longitude, latent.iloc[0].work_latitude, latent.iloc[0].work_longitude) if len(latent) else np.nan
        else:
            record["home_work_distance_km"] = np.nan
        if truth_observation is not None:
            obs = truth_observation.loc[truth_observation.user_id.astype(str).eq(user_id)]
            if "record_probability" in obs:
                record["missing_event_rate"] = float(1 - pd.to_numeric(obs.record_probability, errors="coerce").mean())
            elif "recorded" in obs:
                record["missing_event_rate"] = float(1 - pd.to_numeric(obs.recorded, errors="coerce").mean())
            else:
                record["missing_event_rate"] = np.nan
        else:
            record["missing_event_rate"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def summarize_diagnostics(per_user: pd.DataFrame, thresholds: Mapping[str, Any], transitions: pd.DataFrame | None = None) -> dict[str, Any]:
    """Build machine-readable overall/stratum distributions and separate warnings."""
    distributions: dict[str, Any] = {"overall": {m: _summarize(per_user[m]) for m in METRICS}, "strata": {}}
    for column in ("age_group", "household_type"):
        groups = {}
        for value, group in per_user.groupby(column, dropna=False):
            key = "<missing>" if pd.isna(value) else str(value)
            groups[key] = {"sample_size": len(group), "demographic_missing_count": int(group[column].isna().sum()),
                           "metrics": {m: _summarize(group[m]) for m in METRICS}}
        distributions["strata"][column] = groups
    warnings = []
    for metric, limits in thresholds.items():
        if metric not in per_user:
            continue
        values = pd.to_numeric(per_user[metric], errors="coerce")
        mask = pd.Series(False, index=values.index)
        if "min" in limits: mask |= values < float(limits["min"])
        if "max" in limits: mask |= values > float(limits["max"])
        if mask.any():
            warnings.append({"metric": metric, "severity": "behavioral_warning", "affected_users": int(mask.sum()), "threshold": dict(limits)})
    integrity = []
    missing_event_coordinates = int(per_user.get("coordinate_missing_event_count", pd.Series(dtype=int)).sum())
    if missing_event_coordinates:
        integrity.append({"check": "event_coordinates_present", "severity": "integrity_failure",
                          "affected_events": missing_event_coordinates})
    if transitions is not None and len(transitions):
        negative = int((pd.to_numeric(transitions.elapsed_hours, errors="coerce") < 0).sum())
        if negative:
            integrity.append({"check": "chronological_elapsed_time", "severity": "integrity_failure", "affected_transitions": negative})
        for metric, column in (("straight_line_speed_kmh", "speed_kmh"), ("consecutive_stop_distance_km", "distance_km")):
            if metric in thresholds and "max" in thresholds[metric]:
                count = int((pd.to_numeric(transitions[column], errors="coerce") > float(thresholds[metric]["max"])).sum())
                if count:
                    warnings.append({"metric": metric, "severity": "behavioral_warning", "affected_transitions": count, "threshold": dict(thresholds[metric])})
    return {"schema_version": "geoembeddings-visualization-diagnostics/1.0", "distributions": distributions,
            "behavioral_warnings": warnings, "integrity_failures": integrity,
            "interpretation": "Synthetic strata describe configured support only; small differences are not evidence about real demographic groups."}
