"""Deterministic, observed-only event-removal views for R7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline import export_statistical_baseline
from .export import export_embeddings
from .io import read_json
from .schema import load_observed

ALGORITHM = "sha256-event-removal/1.0"
REQUIRED = {"user_id", "timestamp"}


def _canonical(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def deterministic_event_removal(
    events: pd.DataFrame, *, source_hash: str, seed: int, rate: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a chronological thinned view and a stable per-event mask manifest."""
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("removal rate must be finite and in [0, 1]")
    missing = REQUIRED - set(events.columns)
    if missing:
        raise ValueError(f"Observed events are missing required columns: {sorted(missing)}")
    frame = events.copy()
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Observed event timestamps must be valid") from exc
    if frame["timestamp"].isna().any() or frame["user_id"].isna().any():
        raise ValueError("Observed event user IDs and timestamps must be non-null")
    columns = sorted(frame.columns)
    discriminators = frame.apply(
        lambda row: hashlib.sha256("\x1f".join(_canonical(row[c]) for c in columns).encode()).hexdigest(),
        axis=1,
    )
    keys = pd.DataFrame({"user_id": frame["user_id"].astype(str),
                         "timestamp": frame["timestamp"], "row_discriminator": discriminators})
    if keys.duplicated().any():
        raise ValueError("Observed events contain duplicate event keys")
    scores = []
    for key in keys.itertuples(index=False):
        material = f"{source_hash}\x1f{seed}\x1f{key.user_id}\x1f{key.timestamp.isoformat()}\x1f{key.row_discriminator}"
        scores.append(int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") / 2**64)
    keys["removed"] = np.asarray(scores) < rate
    kept = frame.loc[~keys["removed"].to_numpy()].copy()
    kept["_row_discriminator"] = discriminators.loc[~keys["removed"].to_numpy()].to_numpy()
    kept = kept.sort_values(["user_id", "timestamp", "_row_discriminator"], kind="mergesort")
    kept = kept.drop(columns="_row_discriminator").reset_index(drop=True)
    return kept, keys.sort_values(["user_id", "timestamp", "row_discriminator"]).reset_index(drop=True)


def export_robustness_views(
    observed_dir: str | Path, prepared_dir: str | Path, checkpoint_path: str | Path,
    output_dir: str | Path, config: dict[str, Any], *, kind: str,
) -> dict[str, Any]:
    """Re-encode thinned observed histories through the normal encoder."""
    _, events = load_observed(observed_dir)
    metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
    settings = config["evaluation"]["event_removal"]
    if settings.get("algorithm_version") != ALGORITHM:
        raise ValueError(f"Unsupported event-removal algorithm: {settings.get('algorithm_version')!r}")
    rates = [float(value) for value in settings["rates"]]
    if len(rates) != len(set(rates)):
        raise ValueError("event-removal rates must be unique")
    seed = int(settings["seed"])
    source_hashes = metadata["source_files"]
    event_hash = source_hashes.get("observed_events.csv.gz") or source_hashes.get("events")
    if not event_hash:
        raise ValueError("Prepared metadata lacks the observed-event source hash")
    minimum = int(config["data"]["min_history_events"])
    output_dir = Path(output_dir)
    artifacts = []
    for rate in rates:
        thinned, mask = deterministic_event_removal(events, source_hash=event_hash, seed=seed, rate=rate)
        path = output_dir / kind / f"removal_{rate:.6f}.npz"
        result = None
        if len(thinned) and bool((thinned.groupby("user_id").size() >= minimum).any()):
            if kind == "baseline":
                result = export_statistical_baseline(observed_dir, prepared_dir, path, config,
                    events=thinned, min_history_events=minimum)
            elif kind == "learned":
                result = export_embeddings(observed_dir, prepared_dir, checkpoint_path, path, config,
                    events=thinned, min_history_events=minimum)
            else:
                raise ValueError("kind must be learned or baseline")
        # Empty or wholly unencodable views intentionally have no fabricated vectors.
        if result is not None:
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: payload[name] for name in payload.files}
            arrays.update(source_hashes_json=np.asarray(json.dumps(source_hashes, sort_keys=True)),
                perturbation_algorithm=np.asarray(ALGORITHM), evaluation_seed=np.asarray(seed),
                requested_removal_rate=np.asarray(rate), realized_removed=np.asarray(int(mask.removed.sum())),
                original_events=np.asarray(len(events)), model_kind=np.asarray(kind),
                categorical_fields=np.asarray(metadata["categorical_fields"], dtype=str),
                continuous_fields=np.asarray(metadata["continuous_fields"], dtype=str))
            np.savez_compressed(path, **arrays)
        encoded_keys = [] if result is None else list(zip(arrays["user_id"].astype(str), arrays["cutoff"].astype(str)))
        all_keys = {(str(u), c) for u in events.user_id.unique() for c in ("train", "validation", "test")}
        artifacts.append({"rate": rate, "path": str(path.resolve()) if result else None,
            "original_events": len(events), "removed_events": int(mask.removed.sum()),
            "realized_removal_rate": float(mask.removed.mean()) if len(mask) else 0.0,
            "kept_events": len(thinned), "encoded_rows": len(encoded_keys),
            "encoded_keys": [list(k) for k in encoded_keys],
            "unencodable_keys": [list(k) for k in sorted(all_keys - set(encoded_keys))]})
    return {"artifact_schema": "event-removal-exports/1.0", "algorithm": ALGORITHM,
        "source_hashes": source_hashes, "seed": seed, "kind": kind,
        "field_order": {"categorical": metadata["categorical_fields"], "continuous": metadata["continuous_fields"]},
        "min_history_events": minimum, "artifacts": artifacts,
        "information_boundary": "view construction and encoding read observed/ only"}
