from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .data import DenseUserCutoffDataset, UserCutoffDataset, collate_user_cutoffs
from .io import read_json, sha256_file
from .model import build_model
from .representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION
from .training import _checkpoint_categorical_fields, resolve_device
from .user_roles import protocol_config
from .schema import EVENT_FILE, USER_FILE


def export_embeddings(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    events: pd.DataFrame | None = None,
    min_history_events: int = 1,
) -> dict[str, Any]:
    device = resolve_device(str(config["training"].get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if protocol_config(config) != protocol_config(checkpoint["config"]):
        raise ValueError("Export user-role configuration drifted from the frozen checkpoint")
    dataset = UserCutoffDataset(observed_dir, prepared_dir, checkpoint["config"], events=events,
                                min_history_events=min_history_events)
    model = build_model(
        checkpoint["vocabularies"],
        len(checkpoint["continuous_fields"]),
        checkpoint["config"],
        categorical_fields=_checkpoint_categorical_fields(
            checkpoint, dataset.base.categorical_fields
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_user_cutoffs,
    )
    user_ids: list[str] = []
    cutoffs: list[str] = []
    components: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENT_NAMES}
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode_components(
                batch["categorical"].to(device),
                batch["continuous"].to(device),
                batch["lengths"],
                augment=False,
                elapsed_hours=batch.get("elapsed_hours", None).to(device)
                if batch.get("elapsed_hours") is not None else None,
            )
            user_ids.extend(batch["user_id"])
            cutoffs.extend(batch["cutoff"])
            for name in COMPONENT_NAMES:
                components[name].append(getattr(encoded, name).cpu().numpy())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrices = {name: np.concatenate(values, axis=0) for name, values in components.items()}
    matrix = matrices["combined"]
    metadata = _export_metadata(prepared_dir, checkpoint, sorted(set(cutoffs)), matrices)
    np.savez_compressed(
        output_path,
        user_id=np.asarray(user_ids, dtype=str),
        cutoff=np.asarray(cutoffs, dtype=str),
        embedding=matrix,
        **metadata,
        **{f"component_{name}": value for name, value in matrices.items()},
    )
    return {
        "output": str(output_path.resolve()),
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "cutoffs": sorted(set(cutoffs)),
        "user_role_protocol": dataset.base.metadata.get("user_role_protocol"),
    }


def export_dense_embeddings(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    event_stride: int = 1,
    allow_source_drift: bool = False,
) -> dict[str, Any]:
    """Export event-aligned histories using observed timestamps and no truth labels."""
    device = resolve_device(str(config["training"].get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if protocol_config(config) != protocol_config(checkpoint["config"]):
        raise ValueError("Export user-role configuration drifted from the frozen checkpoint")
    dataset = DenseUserCutoffDataset(
        observed_dir, prepared_dir, checkpoint["config"], event_stride=event_stride,
        allow_source_drift=allow_source_drift,
    )
    model = build_model(
        checkpoint["vocabularies"],
        len(checkpoint["continuous_fields"]),
        checkpoint["config"],
        categorical_fields=_checkpoint_categorical_fields(
            checkpoint, dataset.base.categorical_fields
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_user_cutoffs,
    )
    user_ids: list[str] = []
    timestamps: list[str] = []
    cutoff_kinds: list[str] = []
    history_event_counts: list[int] = []
    components: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENT_NAMES}
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode_components(
                batch["categorical"].to(device),
                batch["continuous"].to(device),
                batch["lengths"],
                augment=False,
                elapsed_hours=batch.get("elapsed_hours", None).to(device)
                if batch.get("elapsed_hours") is not None else None,
            )
            user_ids.extend(batch["user_id"])
            timestamps.extend(batch["timestamp"])
            cutoff_kinds.extend(batch["cutoff_kind"])
            history_event_counts.extend(batch["history_event_count"])
            for name in COMPONENT_NAMES:
                components[name].append(getattr(encoded, name).cpu().numpy())

    matrices = {name: np.concatenate(values, axis=0) for name, values in components.items()}
    matrix = matrices["combined"]
    if not np.isfinite(matrix).all():
        raise ValueError("Dense embedding export contains non-finite values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        user_id=np.asarray(user_ids, dtype=str),
        timestamp=np.asarray(timestamps, dtype=str),
        cutoff_kind=np.asarray(cutoff_kinds, dtype=str),
        embedding=matrix,
        history_event_count=np.asarray(history_event_counts, dtype=np.int64),
        **_export_metadata(
            prepared_dir, checkpoint, sorted(set(timestamps)), matrices,
            source_hashes_override=dataset.base.metadata.get("source_files")
            if not allow_source_drift else {
                USER_FILE: sha256_file(Path(observed_dir) / USER_FILE),
                EVENT_FILE: sha256_file(Path(observed_dir) / EVENT_FILE),
            },
        ),
        **{f"component_{name}": value for name, value in matrices.items()},
    )
    return {
        "output": str(output_path.resolve()),
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "event_stride": event_stride,
        "cutoff_kinds": sorted(set(cutoff_kinds)),
        "information_boundary": "observed/ only; protected episode labels are not exported",
        "user_role_protocol": dataset.base.metadata.get("user_role_protocol"),
    }


def _export_metadata(
    prepared_dir: str | Path,
    checkpoint: dict[str, Any],
    cutoffs: list[str],
    matrices: dict[str, np.ndarray],
    source_hashes_override: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    prepared_path = Path(prepared_dir) / "prepared_metadata.json"
    prepared = read_json(prepared_path)
    preparation_hash = sha256_file(prepared_path)
    schema = checkpoint.get("representation_schema")
    if schema is not None:
        mismatches = []
        if schema.get("preparation_hash") != preparation_hash:
            mismatches.append("preparation_hash")
        if schema.get("source_files") != prepared.get("source_files"):
            mismatches.append("source_files")
        if schema.get("categorical_fields") != prepared.get("categorical_fields"):
            mismatches.append("categorical_fields")
        if schema.get("continuous_fields") != prepared.get("continuous_fields"):
            mismatches.append("continuous_fields")
        if mismatches:
            raise ValueError(f"Checkpoint/preparation representation schema mismatch: {mismatches}")
    source_files = dict(source_hashes_override or prepared["source_files"])
    return {
        "schema_version": np.asarray(EXPORT_SCHEMA_VERSION),
        "component_names": np.asarray(COMPONENT_NAMES, dtype=str),
        "component_dimensions": np.asarray([matrices[name].shape[1] for name in COMPONENT_NAMES], dtype=np.int64),
        "model_variant": np.asarray(checkpoint.get("model_variant", "single_vector")),
        "categorical_fields": np.asarray(checkpoint["categorical_fields"], dtype=str),
        "continuous_fields": np.asarray(checkpoint["continuous_fields"], dtype=str),
        "preparation_hash": np.asarray(preparation_hash),
        "source_file_names": np.asarray(list(source_files), dtype=str),
        "source_hashes": np.asarray(list(source_files.values()), dtype=str),
        "train_end": np.asarray(str(prepared["train_end"])),
        "validation_end": np.asarray(str(prepared["validation_end"])),
        "export_cutoffs": np.asarray(cutoffs, dtype=str),
        "compatibility": np.asarray("embedding aliases component_combined"),
        "user_role_protocol": np.asarray(__import__("json").dumps(
            prepared.get("user_role_protocol"), sort_keys=True, separators=(",", ":"))),
    }
