from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .data import DenseUserCutoffDataset, UserCutoffDataset, collate_user_cutoffs
from .model import build_model
from .training import _checkpoint_categorical_fields, resolve_device


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
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode(
                batch["categorical"].to(device),
                batch["continuous"].to(device),
                batch["lengths"],
                augment=False,
            )
            user_ids.extend(batch["user_id"])
            cutoffs.extend(batch["cutoff"])
            embeddings.append(encoded.cpu().numpy())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.concatenate(embeddings, axis=0)
    np.savez_compressed(
        output_path,
        user_id=np.asarray(user_ids, dtype=str),
        cutoff=np.asarray(cutoffs, dtype=str),
        embedding=matrix,
    )
    return {
        "output": str(output_path.resolve()),
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "cutoffs": sorted(set(cutoffs)),
    }


def export_dense_embeddings(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    event_stride: int = 1,
) -> dict[str, Any]:
    """Export event-aligned histories using observed timestamps and no truth labels."""
    device = resolve_device(str(config["training"].get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dataset = DenseUserCutoffDataset(
        observed_dir, prepared_dir, checkpoint["config"], event_stride=event_stride
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
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode(
                batch["categorical"].to(device),
                batch["continuous"].to(device),
                batch["lengths"],
                augment=False,
            )
            user_ids.extend(batch["user_id"])
            timestamps.extend(batch["timestamp"])
            cutoff_kinds.extend(batch["cutoff_kind"])
            history_event_counts.extend(batch["history_event_count"])
            embeddings.append(encoded.cpu().numpy())

    matrix = np.concatenate(embeddings, axis=0)
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
        categorical_fields=np.asarray(checkpoint["categorical_fields"], dtype=str),
        continuous_fields=np.asarray(checkpoint["continuous_fields"], dtype=str),
    )
    return {
        "output": str(output_path.resolve()),
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "event_stride": event_stride,
        "cutoff_kinds": sorted(set(cutoff_kinds)),
        "information_boundary": "observed/ only; protected episode labels are not exported",
    }
