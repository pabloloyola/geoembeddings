#!/usr/bin/env python3
"""Create a bounded, provenance-recorded Folium view of a simulator run."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from geoembeddings.layout import DatasetLayout

LOCAL_TIMEZONE = "Asia/Tokyo"
SERVICE_COLORS = {"location": "#1f77b4", "local_commerce": "#2ca02c", "ecommerce": "#9467bd", "travel": "#ff7f0e"}
MODE_STYLES = {"passive": {"dash_array": "5,7", "radius": 3}, "active": {"dash_array": None, "radius": 6}}


class EmptySelectionError(ValueError):
    """Raised when filters select no observed events."""


@dataclass(frozen=True)
class ExplorerData:
    users: pd.DataFrame
    events: pd.DataFrame
    pois: pd.DataFrame
    requests: pd.DataFrame
    truth: pd.DataFrame | None = None
    anchors: pd.DataFrame | None = None


def resolve_paths(run_dir: Path | str, *, include_truth: bool = False) -> dict[str, Path]:
    """Resolve all inputs through the canonical layout contract."""
    layout = DatasetLayout.from_path(run_dir)
    paths = {name: layout.observed_file(name) for name in ("users", "events", "poi_catalog", "recommendation_requests")}
    if include_truth:
        paths.update(truth=layout.truth_file("trajectories"), anchors=layout.truth_file("user_latents"))
    return paths


def load_data(run_dir: Path | str, *, include_truth: bool = False, include_anchors: bool = False) -> tuple[ExplorerData, dict]:
    layout = DatasetLayout.from_path(run_dir)
    manifest = layout.validate(require_truth=include_truth)
    if include_anchors and not include_truth:
        raise ValueError("--include-anchors requires evaluator-only --include-truth")
    paths = resolve_paths(run_dir, include_truth=include_truth)
    frames = {name: pd.read_csv(path, dtype={"user_id": "string"}) for name, path in paths.items()}
    events = frames["events"]
    required = {"user_id", "timestamp", "latitude", "longitude"}
    if missing := required - set(events):
        raise ValueError(f"Observed events are missing columns: {sorted(missing)}")
    for name, field in (("events", "timestamp"), ("requests", "request_timestamp"), ("truth", "timestamp")):
        if name in frames and field in frames[name]:
            frames[name][field] = pd.to_datetime(frames[name][field], utc=True, errors="raise").dt.tz_convert(LOCAL_TIMEZONE)
    events = events.sort_values(["user_id", "timestamp", "service_id", "action_type"], kind="stable").reset_index(drop=True)
    return ExplorerData(frames["users"], events, frames["poi_catalog"], frames["recommendation_requests"],
                        frames.get("truth"), frames.get("anchors") if include_anchors else None), manifest


def filter_events(data: ExplorerData, filters: Mapping[str, str | None]) -> pd.DataFrame:
    users = data.users
    for arg, col in (("age_group", "age_group"), ("household_type", "household_type")):
        if filters.get(arg): users = users[users[col].astype(str) == filters[arg]]
    rows = data.events[data.events.user_id.isin(users.user_id)]
    for arg, col in (("user_id", "user_id"), ("service", "service_id"), ("region", "region_id")):
        if filters.get(arg): rows = rows[rows[col].astype(str) == filters[arg]]
    if filters.get("date"):
        date = pd.Timestamp(filters["date"]).date()
        rows = rows[rows.timestamp.dt.date == date]
    return rows.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def deterministic_users(events: pd.DataFrame, max_users: int, seed: int) -> tuple[list[str], bool]:
    users = sorted(events.user_id.astype(str).unique())
    if max_users < 1: raise ValueError("--max-users must be positive")
    if len(users) <= max_users: return users, False
    ranked = sorted(users, key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).hexdigest())
    return sorted(ranked[:max_users]), True


def _sample(frame: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, bool]:
    if limit < 1: raise ValueError("display limits must be positive")
    if len(frame) <= limit: return frame, False
    indices = np.linspace(0, len(frame) - 1, limit, dtype=int)
    return frame.iloc[np.unique(indices)], True


def _tooltip(row: pd.Series, user: Mapping[str, object], *, truth: bool = False) -> str:
    fields = {
        "timestamp": row.get("timestamp", ""), "user": row.get("user_id", ""),
        "age band": user.get("age_group", "unknown"), "household band": user.get("household_type", "unknown"),
        "region": row.get("true_region_id" if truth else "region_id", ""),
        "action": row.get("activity" if truth else "action_type", ""),
        "accuracy (m)": "evaluator-only exact" if truth else row.get("location_accuracy_m", "unknown"),
    }
    return "<br>".join(f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}" for k, v in fields.items())


def render_map(data: ExplorerData, events: pd.DataFrame, output: Path, *, manifest: dict, filters: dict,
               seed: int, max_users: int, max_markers: int, max_path_points: int, include_truth: bool,
               include_anchors: bool) -> dict:
    if events.empty: raise EmptySelectionError("No observed events match the requested filters")
    selected_users, users_truncated = deterministic_users(events, max_users, seed)
    events = events[events.user_id.astype(str).isin(selected_users)]
    import folium
    from folium.plugins import MarkerCluster
    center = [float(events.latitude.mean()), float(events.longitude.mean())]
    fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=10, control_scale=True)
    trajectories = folium.FeatureGroup(name="Observed trajectories", show=True).add_to(fmap)
    points = folium.FeatureGroup(name="Observed event points", show=True).add_to(fmap)
    cluster = MarkerCluster().add_to(points)
    user_lookup = data.users.set_index(data.users.user_id.astype(str)).to_dict("index")
    rendered_markers = 0; marker_truncated = False; path_truncated = False
    per_user_marker_limit = max(1, max_markers // len(selected_users))
    for user_id in selected_users:
        rows = events[events.user_id.astype(str) == user_id]
        path_rows, cut = _sample(rows, max_path_points); path_truncated |= cut
        for service, segment in path_rows.groupby("service_id", sort=True):
            mode = str(segment.observation_mode.mode().iloc[0]) if "observation_mode" in segment else "active"
            folium.PolyLine(segment[["latitude", "longitude"]].values.tolist(), color=SERVICE_COLORS.get(str(service), "#555555"),
                            weight=3, opacity=.7, dash_array=MODE_STYLES.get(mode, {}).get("dash_array"), tooltip=f"{user_id} · {service} · {mode}").add_to(trajectories)
        marker_rows, cut = _sample(rows, per_user_marker_limit); marker_truncated |= cut
        for _, row in marker_rows.iterrows():
            service, mode = str(row.get("service_id", "unknown")), str(row.get("observation_mode", "active"))
            folium.CircleMarker([row.latitude, row.longitude], radius=MODE_STYLES.get(mode, {"radius": 4})["radius"],
                color=SERVICE_COLORS.get(service, "#555555"), fill=True, fill_opacity=.75,
                tooltip=_tooltip(row, user_lookup.get(user_id, {}))).add_to(cluster); rendered_markers += 1
    pois = folium.FeatureGroup(name="Synthetic POIs", show=False).add_to(fmap)
    poi_rows, poi_cut = _sample(data.pois, max_markers)
    for _, row in poi_rows.iterrows():
        folium.CircleMarker([row.latitude, row.longitude], radius=3, color="#008080", fill=True,
            tooltip=html.escape(f"POI {row.poi_id} · {row.category} · {row.region_id}")).add_to(pois)
    requests = folium.FeatureGroup(name="Request locations", show=False).add_to(fmap)
    req = data.requests[data.requests.user_id.astype(str).isin(selected_users)]
    req_rows, req_cut = _sample(req, max_markers)
    for _, row in req_rows.iterrows():
        folium.Marker([row.request_latitude, row.request_longitude], icon=folium.Icon(color="cadetblue", icon="info-sign"),
            tooltip=html.escape(f"Request {row.request_id} · {row.request_timestamp} · {row.region_id}")).add_to(requests)
    truth_count = 0
    if include_truth and data.truth is not None:
        layer = folium.FeatureGroup(name="Evaluator-only latent trajectories", show=False).add_to(fmap)
        truth = data.truth[data.truth.user_id.astype(str).isin(selected_users)]
        for user_id, rows in truth.groupby("user_id", sort=True):
            rows, cut = _sample(rows.sort_values("timestamp"), max_path_points); path_truncated |= cut
            folium.PolyLine(rows[["true_latitude", "true_longitude"]].values.tolist(), color="#d62728", weight=2,
                opacity=.75, tooltip=f"Evaluator-only latent · {html.escape(str(user_id))}").add_to(layer); truth_count += len(rows)
    if include_anchors and data.anchors is not None:
        layer = folium.FeatureGroup(name="Evaluator-only home/work anchors", show=False).add_to(fmap)
        for _, row in data.anchors[data.anchors.user_id.astype(str).isin(selected_users)].iterrows():
            for label, color in (("home", "green"), ("work", "darkpurple")):
                folium.Marker([row[f"{label}_latitude"], row[f"{label}_longitude"]], icon=folium.Icon(color=color),
                    tooltip=html.escape(f"Evaluator-only {label} anchor · {row.user_id}")).add_to(layer)
    truncated = users_truncated or marker_truncated or path_truncated or poi_cut or req_cut
    identity = manifest.get("identity", {})
    metadata = {"schema_version": "geoembeddings-trajectory-visualization/1.0", "dataset_contract": manifest.get("dataset_contract"),
        "manifest_identity": identity, "source_identity": {"config_sha256": manifest.get("config_sha256"),
        "random_streams": manifest.get("random_streams"), "scenario": manifest.get("scenario")},
        "filters": filters, "seed": seed, "truth_access": include_truth,
        "anchors_included": include_anchors, "selected_users": len(selected_users), "selected_events": len(events),
        "rendered_event_markers": rendered_markers, "rendered_truth_points": truth_count, "truncated": truncated,
        "limits": {"max_users": max_users, "max_markers": max_markers, "max_path_points_per_user": max_path_points}}
    banner = folium.Element(f'<div style="position:fixed;bottom:10px;left:10px;z-index:9999;background:white;padding:8px;border:1px solid #555"><b>GeoEmbeddings visualization</b><br>{"Display truncated to configured limits" if truncated else "No display truncation"}<br>Seed: {seed}; truth access: {str(include_truth).lower()}</div>')
    fmap.get_root().html.add_child(banner); folium.LayerControl(collapsed=False).add_to(fmap)
    output.parent.mkdir(parents=True, exist_ok=True); fmap.save(str(output))
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    return metadata


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--output", type=Path)
    for name in ("user-id", "age-group", "household-type", "date", "service", "region"): p.add_argument(f"--{name}")
    p.add_argument("--max-users", type=int, default=50); p.add_argument("--max-markers", type=int, default=2000)
    p.add_argument("--max-path-points", type=int, default=250); p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--include-truth", action="store_true", help="Evaluator-only: permit protected truth access")
    p.add_argument("--include-anchors", action="store_true", help="Show protected home/work coordinates (requires --include-truth)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv); layout = DatasetLayout.from_path(args.run_dir)
    output = (args.output or Path("visualizations") / f"{layout.root.name}_trajectories.html").expanduser().resolve()
    try:
        if output == layout.root or layout.root in output.parents: raise ValueError("visualization output must be outside the immutable run directory")
        if output.exists() or output.with_suffix(output.suffix + ".metadata.json").exists(): raise FileExistsError(f"Refusing to overwrite visualization output: {output}")
        data, manifest = load_data(layout.root, include_truth=args.include_truth, include_anchors=args.include_anchors)
        filters = {k: getattr(args, k) for k in ("user_id", "age_group", "household_type", "date", "service", "region") if getattr(args, k)}
        metadata = render_map(data, filter_events(data, filters), output, manifest=manifest, filters=filters, seed=args.seed,
            max_users=args.max_users, max_markers=args.max_markers, max_path_points=args.max_path_points,
            include_truth=args.include_truth, include_anchors=args.include_anchors)
        print(json.dumps({"output": str(output), **{k: metadata[k] for k in ("selected_users", "selected_events", "truncated", "seed", "truth_access")}}, sort_keys=True)); return 0
    except (ValueError, FileNotFoundError, FileExistsError, EmptySelectionError) as exc:
        print(f"error: {exc}"); return 2

if __name__ == "__main__": raise SystemExit(main())
