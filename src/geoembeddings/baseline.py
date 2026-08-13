from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_json, sha256_file
from .prepare import UNK_TOKEN, derive_continuous_features
from .representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION
from .schema import load_observed
from .data import _dense_cutoff_offsets
from .user_roles import authenticate_roles


def export_statistical_baseline(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    events: pd.DataFrame | None = None,
    min_history_events: int = 1,
) -> dict[str, Any]:
    """Export normalized event histograms and continuous moments, with no learned parameters."""
    if events is None:
        users, events = load_observed(observed_dir)
    else:
        users, _ = load_observed(observed_dir)
    events = events.copy()
    prepared_dir = Path(prepared_dir)
    metadata = read_json(prepared_dir / "prepared_metadata.json")
    assignments = authenticate_roles(metadata, config, users["user_id"].astype(str))
    if assignments is not None:
        eligible_users = {user for user, role in assignments.items() if role == "target_test"}
        events = events[events["user_id"].astype(str).isin(eligible_users)].copy()
    vocabularies: dict[str, dict[str, int]] = read_json(prepared_dir / "vocabularies.json")
    categorical_fields = list(metadata["categorical_fields"])
    continuous_fields = list(metadata["continuous_fields"])
    categorical = _encode_categories(events, categorical_fields, vocabularies)
    continuous = _encode_continuous(events, continuous_fields, metadata)
    maximum = int(config["data"]["max_sequence_length"])
    cutoffs = {
        "train": pd.Timestamp(metadata["train_end"]),
        "validation": pd.Timestamp(metadata["validation_end"]),
        "test": events["timestamp"].max(),
    }

    user_ids: list[str] = []
    cutoff_names: list[str] = []
    vectors: list[np.ndarray] = []
    history_counts: list[int] = []
    for user_id, indices in events.groupby("user_id", sort=False).indices.items():
        ordered = np.asarray(indices, dtype=np.int64)
        for cutoff_name, cutoff in cutoffs.items():
            eligible = ordered[(events.iloc[ordered]["timestamp"] <= cutoff).to_numpy()][-maximum:]
            if len(eligible) < min_history_events:
                continue
            components = []
            for field_index, field in enumerate(categorical_fields):
                size = len(vocabularies[field])
                counts = np.bincount(categorical[eligible, field_index], minlength=size).astype(np.float32)
                counts[0] = 0.0
                components.append(counts / max(counts.sum(), 1.0))
            components.append(continuous[eligible].mean(axis=0).astype(np.float32))
            components.append(continuous[eligible].std(axis=0).astype(np.float32))
            vectors.append(np.concatenate(components))
            user_ids.append(str(user_id))
            cutoff_names.append(cutoff_name)
            history_counts.append(len(eligible))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors)
    zeros = np.zeros_like(matrix)
    source_files = dict(metadata["source_files"])
    np.savez_compressed(
        output_path,
        user_id=np.asarray(user_ids, dtype=str),
        cutoff=np.asarray(cutoff_names, dtype=str),
        embedding=matrix,
        history_event_count=np.asarray(history_counts, dtype=np.int64),
        schema_version=np.asarray(EXPORT_SCHEMA_VERSION),
        component_names=np.asarray(COMPONENT_NAMES),
        component_dimensions=np.asarray([matrix.shape[1]] * len(COMPONENT_NAMES)),
        model_variant=np.asarray("statistical_baseline"),
        categorical_fields=np.asarray(categorical_fields),
        continuous_fields=np.asarray(continuous_fields),
        preparation_hash=np.asarray(sha256_file(prepared_dir / "prepared_metadata.json")),
        source_file_names=np.asarray(list(source_files)),
        source_hashes=np.asarray(list(source_files.values())),
        train_end=np.asarray(metadata["train_end"]),
        validation_end=np.asarray(metadata["validation_end"]),
        export_cutoffs=np.asarray(["train", "validation", "test"]),
        compatibility=np.asarray("statistical baseline mapped to persistent/combined; context=zeros"),
        user_role_protocol=np.asarray(__import__("json").dumps(
            metadata.get("user_role_protocol"), sort_keys=True, separators=(",", ":"))),
        component_persistent=matrix,
        component_context=zeros,
        component_combined=matrix,
    )
    return {
        "output": str(output_path.resolve()),
        "kind": "non_learned_statistical_baseline",
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "cutoffs": sorted(set(cutoff_names)),
    }


def export_dense_statistical_baseline(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    event_stride: int = 1,
) -> dict[str, Any]:
    """Export the same observed-only statistical representation after events."""
    _, events = load_observed(observed_dir)
    metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
    vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
    categorical_fields = list(metadata["categorical_fields"])
    continuous_fields = list(metadata["continuous_fields"])
    categorical = _encode_categories(events, categorical_fields, vocabularies)
    continuous = _encode_continuous(events, continuous_fields, metadata)
    maximum = int(config["data"]["max_sequence_length"])
    users, timestamps, counts, vectors = [], [], [], []
    for user_id, indices in events.groupby("user_id", sort=False).indices.items():
        ordered = np.asarray(indices, dtype=np.int64)
        offsets = _dense_cutoff_offsets(len(ordered), event_stride)
        by_timestamp = {pd.Timestamp(events.iloc[int(ordered[offset])]["timestamp"]): offset
                        for offset in offsets}
        for offset in sorted(by_timestamp.values()):
            eligible = ordered[max(0, offset + 1 - maximum) : offset + 1]
            components = []
            for field_index, field in enumerate(categorical_fields):
                size = len(vocabularies[field])
                values = np.bincount(categorical[eligible, field_index], minlength=size).astype(np.float32)
                values[0] = 0
                components.append(values / max(values.sum(), 1.0))
            components.extend((continuous[eligible].mean(0), continuous[eligible].std(0)))
            vectors.append(np.concatenate(components).astype(np.float32))
            users.append(str(user_id))
            timestamps.append(pd.Timestamp(events.iloc[ordered[offset]]["timestamp"]).isoformat())
            counts.append(offset + 1)
    matrix = np.stack(vectors)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, user_id=np.asarray(users, dtype=str), timestamp=np.asarray(timestamps, dtype=str),
        cutoff_kind=np.asarray(["observed_event"] * len(users), dtype=str), embedding=matrix,
        history_event_count=np.asarray(counts, dtype=np.int64),
        categorical_fields=np.asarray(categorical_fields, dtype=str),
        continuous_fields=np.asarray(continuous_fields, dtype=str),
    )
    return {"output": str(output_path.resolve()), "rows": len(users), "users": len(set(users)),
            "embedding_dim": int(matrix.shape[1]), "event_stride": event_stride,
            "information_boundary": "observed/ only; protected episode labels are not exported"}


def _encode_categories(
    events: pd.DataFrame,
    fields: list[str],
    vocabularies: dict[str, dict[str, int]],
) -> np.ndarray:
    columns = []
    for field in fields:
        vocabulary = vocabularies[field]
        unk = vocabulary[UNK_TOKEN]
        values = events[field].fillna(UNK_TOKEN).astype(str)
        columns.append(values.map(lambda value: vocabulary.get(value, unk)).to_numpy(dtype=np.int64))
    return np.stack(columns, axis=1)


def _encode_continuous(
    events: pd.DataFrame,
    fields: list[str],
    metadata: dict[str, Any],
) -> np.ndarray:
    derived = derive_continuous_features(events)
    columns = []
    for field in fields:
        statistics = metadata["continuous_statistics"][field]
        values = derived[field].astype(float).fillna(float(statistics["mean"]))
        normalized = (values - float(statistics["mean"])) / float(statistics["std"])
        columns.append(normalized.to_numpy(dtype=np.float32))
    return np.stack(columns, axis=1)
