"""Versioned deterministic observed-only robustness views for R6/R7."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline import export_statistical_baseline
from .export import export_embeddings
from .io import read_json
from .schema import load_observed

ALGORITHM = "sha256-robustness-views/2.0"
REQUIRED = {"user_id", "timestamp"}
SUPPORTED_VIEWS = {"event-removal", "gps", "timestamp", "leave-one-service-out", "recent-truncation"}


def _canonical(value: Any) -> str:
    if pd.isna(value): return "<NA>"
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return str(value)


def _prepare(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED - set(events.columns)
    if missing: raise ValueError(f"Observed events are missing required columns: {sorted(missing)}")
    frame = events.copy()
    try: frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    except (TypeError, ValueError) as exc: raise ValueError("Observed event timestamps must be valid") from exc
    if frame["timestamp"].isna().any() or frame["user_id"].isna().any():
        raise ValueError("Observed event user IDs and timestamps must be non-null")
    columns = sorted(frame.columns)
    discriminators = frame.apply(lambda row: hashlib.sha256(
        "\x1f".join(_canonical(row[c]) for c in columns).encode()).hexdigest(), axis=1)
    keys = pd.DataFrame({"user_id": frame.user_id.astype(str), "timestamp": frame.timestamp,
                         "row_discriminator": discriminators}, index=frame.index)
    if keys.duplicated().any(): raise ValueError("Observed events contain duplicate event keys")
    return frame, keys


def _fraction(source_hash: str, seed: int, key: Any, label: str) -> float:
    material = f"{source_hash}\x1f{seed}\x1f{key.user_id}\x1f{key.timestamp.isoformat()}\x1f{key.row_discriminator}\x1f{label}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") / 2**64


def _finish(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy(); frame["_row_discriminator"] = keys.loc[frame.index, "row_discriminator"]
    return frame.sort_values(["user_id", "timestamp", "_row_discriminator"], kind="mergesort").drop(
        columns="_row_discriminator").reset_index(drop=True)


def deterministic_event_removal(events: pd.DataFrame, *, source_hash: str, seed: int, rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(rate) or not 0 <= rate <= 1: raise ValueError("removal rate must be finite and in [0, 1]")
    frame, keys = _prepare(events)
    keys["removed"] = [_fraction(source_hash, seed, k, "event-removal") < rate for k in keys.itertuples()]
    kept = frame.loc[~keys.removed]
    return _finish(kept, keys), keys.sort_values(["user_id", "timestamp", "row_discriminator"]).reset_index(drop=True)


def perturb_view(events: pd.DataFrame, *, source_hash: str, seed: int, kind: str,
                 parameters: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Apply one deterministic view without consulting truth or process-global RNG."""
    frame, keys = _prepare(events)
    changed = pd.Series(False, index=frame.index)
    if kind == "gps":
        sigma = float(parameters["sigma_m"])
        if not np.isfinite(sigma) or sigma < 0: raise ValueError("gps sigma_m must be finite and non-negative")
        if not {"latitude", "longitude"} <= set(frame): raise ValueError("gps view requires latitude and longitude")
        for idx, key in zip(frame.index, keys.itertuples()):
            u1=max(_fraction(source_hash,seed,key,"gps-radius"),1e-15); u2=_fraction(source_hash,seed,key,"gps-angle")
            radius=sigma*math.sqrt(-2*math.log(u1)); north=radius*math.cos(2*math.pi*u2); east=radius*math.sin(2*math.pi*u2)
            lat=float(frame.at[idx,"latitude"]); lon=float(frame.at[idx,"longitude"])
            frame.at[idx,"latitude"]=lat+north/111_320.0
            denom=max(111_320.0*math.cos(math.radians(lat)),1.0); frame.at[idx,"longitude"]=lon+east/denom
            changed.at[idx] = sigma > 0
        if (frame.latitude.abs()>90).any() or (frame.longitude.abs()>180).any():
            raise ValueError("gps perturbation produced invalid coordinates")
    elif kind == "timestamp":
        maximum=float(parameters["max_jitter_seconds"])
        if not np.isfinite(maximum) or maximum < 0: raise ValueError("max_jitter_seconds must be finite and non-negative")
        offsets=[(2*_fraction(source_hash,seed,k,"timestamp-jitter")-1)*maximum for k in keys.itertuples()]
        frame["timestamp"] += pd.to_timedelta(offsets, unit="s"); changed[:] = maximum > 0
    elif kind == "leave-one-service-out":
        service=str(parameters["service_id"])
        if "service_id" not in frame: raise ValueError("leave-one-service-out requires service_id")
        changed = frame.service_id.astype(str).eq(service); frame=frame.loc[~changed]
    elif kind == "recent-truncation":
        count=int(parameters["remove_recent_events"])
        if count < 0: raise ValueError("remove_recent_events must be non-negative")
        ordered=_finish(frame, keys)
        # Re-key after canonical ordering; remove the N most recent observations per user.
        remove_idx=ordered.groupby("user_id", sort=False).tail(count).index if count else []
        ordered_frame, ordered_keys = _prepare(ordered)
        removed_keys=set(zip(ordered_frame.loc[remove_idx,"user_id"].astype(str),
            ordered_frame.loc[remove_idx,"timestamp"].map(pd.Timestamp),
            ordered_keys.loc[remove_idx,"row_discriminator"]))
        changed=pd.Series([(str(u),pd.Timestamp(t),d) in removed_keys for u,t,d in
            zip(frame.user_id,frame.timestamp,keys.row_discriminator)],index=frame.index)
        frame=frame.loc[~changed]
    else: raise ValueError(f"Unsupported robustness view: {kind!r}")
    output=_finish(frame, keys)
    details={"original_events":len(events), "changed_events":int(changed.sum()), "kept_events":len(output),
             "realized_corruption":float(changed.mean()) if len(changed) else 0.0}
    manifest=keys.copy(); manifest["changed_or_removed"]=changed.to_numpy()
    return output, details, manifest.sort_values(["user_id","timestamp","row_discriminator"]).reset_index(drop=True)


def _specs(config: dict[str, Any], requested: list[str]) -> tuple[list[dict[str,Any]], str]:
    root=config["evaluation"]["robustness"]
    if root.get("schema_version") != "robustness-spec/1.0": raise ValueError("Unsupported robustness specification version")
    if root.get("algorithm_version", ALGORITHM) != ALGORITHM: raise ValueError("Unsupported robustness algorithm version")
    unknown=set(requested)-SUPPORTED_VIEWS
    if unknown: raise ValueError(f"Unsupported robustness views: {sorted(unknown)}")
    specs=[]
    for kind in requested:
        raw=root["views"][kind]
        variants=raw if isinstance(raw,list) else [raw]
        for params in variants:
            spec={"schema_version":root["schema_version"],"kind":kind,"parameters":params}
            digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            spec["view_id"]=f"{kind}-{digest[:12]}"; specs.append(spec)
    config_hash=hashlib.sha256(json.dumps(specs,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return specs, config_hash


def export_robustness_views(observed_dir: str|Path, prepared_dir: str|Path, checkpoint_path: str|Path,
                            output_dir: str|Path, config: dict[str,Any], *, kind: str,
                            views: list[str]|None=None) -> dict[str,Any]:
    _, events=load_observed(observed_dir); metadata=read_json(Path(prepared_dir)/"prepared_metadata.json")
    requested=views or list(config["evaluation"]["robustness"]["views"])
    specs, config_hash=_specs(config,requested); seed=int(config["evaluation"]["robustness"]["seed"])
    source_hashes=metadata["source_files"]; event_hash=source_hashes.get("observed_events.csv.gz") or source_hashes.get("events")
    if not event_hash: raise ValueError("Prepared metadata lacks the observed-event source hash")
    minimum=int(config["data"]["min_history_events"]); output_dir=Path(output_dir); artifacts=[]
    all_keys={(str(u),c) for u in events.user_id.unique() for c in ("train","validation","test")}
    for spec in specs:
        view_events, details, mask=perturb_view(events,source_hash=event_hash,seed=seed,kind=spec["kind"],parameters=spec["parameters"])
        mask_hash=hashlib.sha256(mask.to_csv(index=False).encode()).hexdigest(); path=output_dir/kind/f"{spec['view_id']}.npz"; result=None
        if len(view_events) and bool((view_events.groupby("user_id").size()>=minimum).any()):
            fn=export_statistical_baseline if kind=="baseline" else export_embeddings if kind=="learned" else None
            if fn is None: raise ValueError("kind must be learned or baseline")
            args=(observed_dir,prepared_dir,path,config) if kind=="baseline" else (observed_dir,prepared_dir,checkpoint_path,path,config)
            result=fn(*args,events=view_events,min_history_events=minimum)
        arrays={}
        if result:
            with np.load(path,allow_pickle=False) as payload: arrays={n:payload[n] for n in payload.files}
            arrays.update(view_id=np.asarray(spec["view_id"]),view_spec_json=np.asarray(json.dumps(spec,sort_keys=True)),
                view_config_hash=np.asarray(config_hash),mask_hash=np.asarray(mask_hash),source_hashes_json=np.asarray(json.dumps(source_hashes,sort_keys=True)),
                perturbation_algorithm=np.asarray(ALGORITHM),evaluation_seed=np.asarray(seed),model_kind=np.asarray(kind),
                categorical_fields=np.asarray(metadata["categorical_fields"],dtype=str),continuous_fields=np.asarray(metadata["continuous_fields"],dtype=str))
            np.savez_compressed(path,**arrays)
        encoded=[] if not result else list(zip(arrays["user_id"].astype(str),arrays["cutoff"].astype(str)))
        artifacts.append({**spec,**details,"mask_hash":mask_hash,"path":str(path.resolve()) if result else None,
            "encoded_rows":len(encoded),"encoded_keys":[list(k) for k in encoded],
            "unencodable_keys":[list(k) for k in sorted(all_keys-set(encoded))],
            "exclusion_reasons":{"insufficient_history":len(all_keys-set(encoded))}})
    return {"artifact_schema":"robustness-view-exports/2.0","algorithm":ALGORITHM,"specification_hash":config_hash,
        "source_hashes":source_hashes,"seed":seed,"kind":kind,"requested_views":requested,
        "field_order":{"categorical":metadata["categorical_fields"],"continuous":metadata["continuous_fields"]},
        "min_history_events":minimum,"artifacts":artifacts,"information_boundary":"view construction and encoding read observed/ only"}
