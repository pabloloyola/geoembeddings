"""Deterministic evaluator-only user-journey report (R1, R4, R8, R9).

Observed inputs are loaded and authenticated before the optional, explicit truth
branch.  Keeping that branch here prevents episode labels from entering model or
ranking APIs.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contract import OBSERVED_FILES
from .embedding_visualization import fit_pca
from .io import sha256_file
from .layout import DatasetLayout, ExperimentLayout
from .representation_schema import load_embedding_export

SCHEMA = "geoembeddings-user-journey/1.0"
PROTECTED_LABEL = "SYNTHETIC PROTECTED TRUTH — EVALUATOR ONLY"
FORBIDDEN = ("utility", "latent", "episode_id", "true_", "chosen")


def _time(value: str, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert(None)
    if pd.isna(result):
        raise ValueError(f"invalid {name}")
    return result


def reject_protected_columns(frame: pd.DataFrame, *, source: str) -> None:
    bad = sorted(c for c in frame.columns if any(token in c.lower() for token in FORBIDDEN))
    if bad:
        raise ValueError(f"protected fields in observed {source}: {bad}")


def deterministic_rows(frame: pd.DataFrame, *, timestamp: str, limit: int,
                       tie_breakers: Iterable[str]) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("truncation limits must be positive")
    keys = [timestamp, *(key for key in tie_breakers if key in frame.columns)]
    return frame.sort_values(keys, kind="mergesort").head(limit).reset_index(drop=True)


def annotate_boundaries(timestamps: Iterable[Any], episodes: pd.DataFrame) -> list[list[dict[str, str]]]:
    boundaries: list[tuple[pd.Timestamp, str, str]] = []
    for row in episodes.sort_values(["start_time", "end_time", "episode_id"], kind="mergesort").itertuples():
        boundaries.extend([(_time(row.start_time, "episode start"), "start", str(row.episode_id)),
                           (_time(row.end_time, "episode end"), "end", str(row.episode_id))])
    output = []
    for timestamp in map(pd.Timestamp, timestamps):
        output.append([{"kind": kind, "episode_id": episode_id, "timestamp": str(boundary)}
                       for boundary, kind, episode_id in boundaries if boundary == timestamp])
    return output


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    columns = [c for c in columns if c in frame]
    if frame.empty:
        return '<p class="empty">No rows in the selected interval.</p>'
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in frame[columns].itertuples(index=False, name=None))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _svg(points: list[dict[str, Any]], *, geographic: bool = False) -> str:
    if not points:
        return '<p class="empty">No history available.</p>'
    xs = np.asarray([p["x"] for p in points], float); ys = np.asarray([p["y"] for p in points], float)
    def scale(values: np.ndarray, reverse: bool = False) -> np.ndarray:
        span = float(np.ptp(values)) or 1.; out = 20 + 360 * (values-values.min())/span
        return 220-out if reverse else out
    px, py = scale(xs), scale(ys, True)
    path = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i,(x,y) in enumerate(zip(px,py)))
    dots = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{html.escape(points[i]["label"])}</title></circle>' for i,(x,y) in enumerate(zip(px,py)))
    return f'<svg viewBox="0 0 400 240" role="img"><path d="{path}"/>{dots}</svg>'


def build_user_journey_report(run: DatasetLayout, experiment: ExperimentLayout, *, user_id: str,
                              start: str, end: str, truth_access: bool = False,
                              ranking_models: tuple[str, ...] = ("popularity", "nearest"),
                              max_events: int = 500, max_requests: int = 50,
                              max_candidates: int = 20, overwrite: bool = False) -> dict[str, Any]:
    start_at, end_at = _time(start, "start"), _time(end, "end")
    if start_at >= end_at:
        raise ValueError("time interval must be bounded with start before end")
    manifest = run.validate(require_truth=truth_access)
    export_path = experiment.dense_embeddings
    loaded = load_embedding_export(export_path, dense=True)
    source_hashes = dict(zip(loaded.arrays.get("source_file_names", []).astype(str),
                             loaded.arrays.get("source_hashes", []).astype(str)))
    for name, digest in source_hashes.items():
        path = run.observed / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"dense export source authentication failed: {name}")
    def observed(name: str) -> pd.DataFrame:
        frame = pd.read_csv(run.observed_file(name), keep_default_na=False)
        reject_protected_columns(frame, source=name)
        return frame
    events = observed("events"); requests = observed("recommendation_requests")
    impressions = observed("impressions"); catalog = observed("poi_catalog")
    events["_time"] = pd.to_datetime(events["timestamp"])
    requests["_time"] = pd.to_datetime(requests["timestamp"])
    events = deterministic_rows(events[(events.user_id.astype(str)==user_id)&events._time.between(start_at,end_at, inclusive="both")], timestamp="_time", limit=max_events, tie_breakers=("service_id","event_id"))
    requests = deterministic_rows(requests[(requests.user_id.astype(str)==user_id)&requests._time.between(start_at,end_at, inclusive="both")], timestamp="_time", limit=max_requests, tie_breakers=("request_id",))
    request_ids = set(requests.request_id.astype(str))
    candidate = impressions[impressions.request_id.astype(str).isin(request_ids)].merge(catalog, on="poi_id", how="left", validate="many_to_one")
    candidate = deterministic_rows(candidate, timestamp="request_id", limit=max_requests*max_candidates, tie_breakers=("position","poi_id"))
    times = pd.to_datetime(loaded.arrays["timestamp"].astype(str)); users = loaded.arrays["user_id"].astype(str)
    interval = (users==user_id)&(times>=start_at)&(times<=end_at)
    # One earliest authenticated row per user is the fixed reference population.
    order = np.lexsort((times.astype("int64"), users)); reference=[]; seen=set()
    for i in order:
        if users[i] not in seen: reference.append(i); seen.add(users[i])
    projections: dict[str,list[dict[str,Any]]] = {}
    reducer_meta = {}
    for component in ("persistent", "context", "combined"):
        values = loaded.components[component]
        reducer = fit_pca(values[reference])
        coords = reducer.transform(values)
        projections[component] = [{"x":float(coords[i,0]),"y":float(coords[i,1]),"label":str(times[i])} for i in np.flatnonzero(interval)]
        reducer_meta[component] = {"normalization":"standard","reference_rows":len(reference),
            "mean":reducer.mean.tolist(),"scale":reducer.scale.tolist(),"components":reducer.components.tolist()}
    episodes = pd.DataFrame()
    if truth_access:
        episodes = pd.read_csv(run.truth_file("episodes"), keep_default_na=False)
        episodes = episodes[episodes.user_id.astype(str)==user_id].copy()
        episodes = episodes[(pd.to_datetime(episodes.start_time)<=end_at)&(pd.to_datetime(episodes.end_time)>=start_at)]
    ranking_html=[]; ranking_hashes={}
    for model in ranking_models:
        prediction = experiment.ranking_predictions(model); report = experiment.ranking_report(model)
        if prediction.is_file() and report.is_file():
            with np.load(prediction, allow_pickle=False) as data:
                frame=pd.DataFrame({k:data[k] for k in ("request_id","poi_id","rank","score")})
            frame=frame[frame.request_id.astype(str).isin(request_ids)].sort_values(["request_id","rank","poi_id"],kind="mergesort").head(max_requests*max_candidates)
            ranking_html.append(f"<h3>{html.escape(model)}</h3>"+_table(frame,["request_id","rank","poi_id","score"]))
            ranking_hashes[model]={"predictions":sha256_file(prediction),"report":sha256_file(report)}
    protected = ""
    if truth_access:
        protected=f'<section class="protected"><div class="truth">{PROTECTED_LABEL}</div><h2>Episode boundaries</h2>{_table(episodes,["episode_id","start_time","end_time","primary_intent","secondary_intent"])}<div class="truth">{PROTECTED_LABEL}</div></section>'
    panels="".join(f"<article><h3>{c}</h3>{_svg(projections[c])}</article>" for c in projections)
    trajectory=[{"x":float(r.longitude),"y":float(r.latitude),"label":str(r.timestamp)} for r in events.itertuples() if str(r.latitude) and str(r.longitude)]
    document=f'''<!doctype html><meta charset="utf-8"><title>User journey {html.escape(user_id)}</title><style>body{{font:14px sans-serif;max-width:1200px;margin:auto}}section{{border:1px solid #bbb;padding:1rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.25rem}}svg{{width:100%;max-width:500px;background:#f7f7f7}}svg path{{fill:none;stroke:#1769aa}}svg circle{{fill:#d84315}}.grid{{display:grid;grid-template-columns:repeat(3,1fr)}}.protected{{border:4px solid #b00020;background:#fff1f2}}.truth{{position:sticky;top:0;background:#b00020;color:white;font-weight:bold;padding:.5rem}}</style><h1>User journey: {html.escape(user_id)}</h1><p>{start_at} — {end_at}</p><section><h2>Observed timeline by service</h2>{_table(events,["timestamp","service_id","action","category","region_id"])}</section><section><h2>Observed trajectory</h2>{_svg(trajectory,geographic=True)}</section><section><h2>Observed representation trajectories</h2><div class="grid">{panels}</div></section><section><h2>Observed recommendation requests and public candidates</h2>{_table(requests,["request_id","timestamp","region_id","context_source"])}{_table(candidate,["request_id","poi_id","position","is_available","travel_time_minutes","category","price_level","family_suitability","environment","local_popularity"])}</section><section><h2>Selected observed ranking reports</h2>{''.join(ranking_html) or '<p class="empty">No selected ranking artifacts.</p>'}</section>{protected}'''
    output=experiment.user_journey_report(user_id); metadata_path=experiment.user_journey_metadata(user_id)
    if (output.exists() or metadata_path.exists()) and not overwrite: raise FileExistsError("user journey report exists")
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(document,encoding="utf-8")
    metadata={"schema_version":SCHEMA,"requirements":["R1","R4","R8","R9"],"selected_user":user_id,
      "time_range":{"start":str(start_at),"end":str(end_at)},"truth_access":truth_access,"protected_label":PROTECTED_LABEL if truth_access else None,
      "truncation_limits":{"events":max_events,"requests":max_requests,"candidates_per_request":max_candidates},
      "export":{"path":str(export_path),"sha256":sha256_file(export_path),"source_hashes":source_hashes},
      "run_source_hash":manifest.get("source_hash"),"reducers":reducer_meta,"reference_population":{"selection":"earliest authenticated dense row per user","users":len(reference)},
      "ranking_hashes":ranking_hashes,"counts":{"events":len(events),"embedding_timestamps":int(interval.sum()),"requests":len(requests),"candidates":len(candidate),"episodes":len(episodes)}}
    metadata_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"status":"complete","report":str(output),"metadata":str(metadata_path),"counts":metadata["counts"]}
