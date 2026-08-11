from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_json
from .prepare import UNK_TOKEN, derive_continuous_features
from .schema import load_observed


def export_statistical_baseline(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Export normalized event histograms and continuous moments, with no learned parameters."""
    _, events = load_observed(observed_dir)
    prepared_dir = Path(prepared_dir)
    metadata = read_json(prepared_dir / "prepared_metadata.json")
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
    for user_id, indices in events.groupby("user_id", sort=False).indices.items():
        ordered = np.asarray(indices, dtype=np.int64)
        for cutoff_name, cutoff in cutoffs.items():
            eligible = ordered[(events.iloc[ordered]["timestamp"] <= cutoff).to_numpy()][-maximum:]
            if len(eligible) == 0:
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors)
    np.savez_compressed(
        output_path,
        user_id=np.asarray(user_ids, dtype=str),
        cutoff=np.asarray(cutoff_names, dtype=str),
        embedding=matrix,
    )
    return {
        "output": str(output_path.resolve()),
        "kind": "non_learned_statistical_baseline",
        "rows": len(user_ids),
        "users": len(set(user_ids)),
        "embedding_dim": int(matrix.shape[1]),
        "cutoffs": sorted(set(cutoff_names)),
    }


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

