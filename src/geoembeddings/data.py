from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .io import read_json
from .prepare import UNK_TOKEN, derive_continuous_features
from .schema import load_observed


TARGET_FIELDS = {
    "next_service": "service_id",
    "next_action": "action_type",
    "next_category": "object_category",
    "next_region": "region_id",
    "next_geohash_5": "geohash_5",
    "next_geohash_7": "geohash_7",
}


@dataclass(frozen=True)
class SampleReference:
    user_id: str
    context_indices: tuple[int, ...]
    target_index: int


class EventWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        observed_dir: str | Path,
        prepared_dir: str | Path,
        split: str,
        config: dict[str, Any],
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        _, events = load_observed(observed_dir)
        self.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        self.vocabularies: dict[str, dict[str, int]] = read_json(
            Path(prepared_dir) / "vocabularies.json"
        )
        self.categorical_fields = list(self.metadata["categorical_fields"])
        self.continuous_fields = list(self.metadata["continuous_fields"])
        self._validate_prepared_contract(config)
        self.events = events
        self.encoded_categories = self._encode_categories(events)
        self.continuous = self._encode_continuous(events)
        self.references = self._make_references(events, split, config["data"])
        if not self.references:
            raise ValueError(f"No usable {split} windows after applying min_history_events")

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.references[index]
        context_indices = np.asarray(reference.context_indices, dtype=np.int64)
        categorical = torch.from_numpy(self.encoded_categories[context_indices]).long()
        continuous = torch.from_numpy(self.continuous[context_indices]).float()

        midpoint = max(1, len(context_indices) // 2)
        early_indices = context_indices[:midpoint]
        late_indices = context_indices[midpoint:]
        if len(late_indices) == 0:
            late_indices = context_indices

        target_row = self.events.iloc[reference.target_index]
        targets = {
            objective: self._token_id(field, target_row[field])
            for objective, field in TARGET_FIELDS.items()
            if field in self.vocabularies
        }
        return {
            "user_id": reference.user_id,
            "categorical": categorical,
            "continuous": continuous,
            "early_categorical": torch.from_numpy(self.encoded_categories[early_indices]).long(),
            "early_continuous": torch.from_numpy(self.continuous[early_indices]).float(),
            "late_categorical": torch.from_numpy(self.encoded_categories[late_indices]).long(),
            "late_continuous": torch.from_numpy(self.continuous[late_indices]).float(),
            "targets": targets,
        }

    def _encode_categories(self, events: pd.DataFrame) -> np.ndarray:
        columns = []
        for field in self.categorical_fields:
            vocabulary = self.vocabularies[field]
            unk = vocabulary[UNK_TOKEN]
            values = events[field].fillna(UNK_TOKEN).astype(str)
            columns.append(values.map(lambda value: vocabulary.get(value, unk)).to_numpy(dtype=np.int64))
        return np.stack(columns, axis=1)

    def _validate_prepared_contract(self, config: dict[str, Any]) -> None:
        configured_fields = list(config["data"]["categorical_fields"])
        if bool(config["data"].get("include_object_id", False)):
            configured_fields.append("object_id")
        if configured_fields != self.categorical_fields:
            raise ValueError(
                "Embedding configuration categorical field order does not match prepared data: "
                f"config={configured_fields}, prepared={self.categorical_fields}. "
                "Rerun prepare with this embedding configuration."
            )
        vocabulary_fields = set(self.vocabularies)
        metadata_fields = set(self.categorical_fields)
        if vocabulary_fields != metadata_fields:
            raise ValueError(
                "Prepared categorical vocabularies do not match metadata: "
                f"missing={sorted(metadata_fields - vocabulary_fields)}, "
                f"unexpected={sorted(vocabulary_fields - metadata_fields)}"
            )

    def _encode_continuous(self, events: pd.DataFrame) -> np.ndarray:
        derived = derive_continuous_features(events)
        columns = []
        for field in self.continuous_fields:
            if field not in derived.columns:
                raise ValueError(f"Unsupported continuous field: {field}")
            statistics = self.metadata["continuous_statistics"][field]
            values = derived[field].astype(float).fillna(float(statistics["mean"]))
            normalized = (values - float(statistics["mean"])) / float(statistics["std"])
            columns.append(normalized.to_numpy(dtype=np.float32))
        return np.stack(columns, axis=1)

    def _make_references(
        self,
        events: pd.DataFrame,
        split: str,
        data_config: dict[str, Any],
    ) -> list[SampleReference]:
        train_end = pd.Timestamp(self.metadata["train_end"])
        validation_end = pd.Timestamp(self.metadata["validation_end"])
        minimum = int(data_config["min_history_events"])
        maximum = int(data_config["max_sequence_length"])
        references: list[SampleReference] = []
        for user_id, indices in events.groupby("user_id", sort=False).indices.items():
            ordered = np.asarray(indices, dtype=np.int64)
            for offset in range(minimum, len(ordered)):
                target_index = int(ordered[offset])
                timestamp = events.iloc[target_index]["timestamp"]
                if _timestamp_split(timestamp, train_end, validation_end) != split:
                    continue
                context = ordered[max(0, offset - maximum) : offset]
                references.append(
                    SampleReference(
                        user_id=str(user_id),
                        context_indices=tuple(int(value) for value in context),
                        target_index=target_index,
                    )
                )
        return references

    def _token_id(self, field: str, value: Any) -> int:
        vocabulary = self.vocabularies[field]
        normalized = UNK_TOKEN if pd.isna(value) else str(value)
        return int(vocabulary.get(normalized, vocabulary[UNK_TOKEN]))


def _timestamp_split(
    timestamp: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> str:
    if timestamp <= train_end:
        return "train"
    if timestamp <= validation_end:
        return "validation"
    return "test"


def collate_windows(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    batch: dict[str, Any] = {"user_id": [sample["user_id"] for sample in samples]}
    for prefix in ("", "early_", "late_"):
        categorical_key = f"{prefix}categorical"
        continuous_key = f"{prefix}continuous"
        categorical_sequences = [sample[categorical_key] for sample in samples]
        continuous_sequences = [sample[continuous_key] for sample in samples]
        batch[categorical_key] = pad_sequence(categorical_sequences, batch_first=True, padding_value=0)
        batch[continuous_key] = pad_sequence(continuous_sequences, batch_first=True, padding_value=0.0)
        batch[f"{prefix}lengths"] = torch.tensor(
            [len(sequence) for sequence in categorical_sequences], dtype=torch.long
        )

    target_names = sorted(samples[0]["targets"])
    batch["targets"] = {
        name: torch.tensor([sample["targets"][name] for sample in samples], dtype=torch.long)
        for name in target_names
    }
    return batch


class UserCutoffDataset(Dataset[dict[str, Any]]):
    """One complete observed history per user at one or more temporal cutoffs."""

    def __init__(
        self,
        observed_dir: str | Path,
        prepared_dir: str | Path,
        config: dict[str, Any],
        *,
        events: pd.DataFrame | None = None,
        min_history_events: int = 1,
    ) -> None:
        base = EventWindowDataset.__new__(EventWindowDataset)
        if events is None:
            _, events = load_observed(observed_dir)
        events = events.copy()
        base.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        base.vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
        base.categorical_fields = list(base.metadata["categorical_fields"])
        base.continuous_fields = list(base.metadata["continuous_fields"])
        base._validate_prepared_contract(config)
        base.events = events
        base.encoded_categories = base._encode_categories(events)
        base.continuous = base._encode_continuous(events)
        self.base = base
        self.maximum = int(config["data"]["max_sequence_length"])
        self.items: list[tuple[str, str, np.ndarray]] = []
        cutoffs = {
            "train": pd.Timestamp(base.metadata["train_end"]),
            "validation": pd.Timestamp(base.metadata["validation_end"]),
            "test": pd.Timestamp(base.metadata.get("timestamp_max", events["timestamp"].max())),
        }
        for user_id, indices in events.groupby("user_id", sort=False).indices.items():
            ordered = np.asarray(indices, dtype=np.int64)
            for cutoff_name, cutoff in cutoffs.items():
                eligible_mask = (events.iloc[ordered]["timestamp"] <= cutoff).to_numpy()
                eligible = ordered[eligible_mask]
                if len(eligible) >= min_history_events:
                    self.items.append((str(user_id), cutoff_name, eligible[-self.maximum :]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user_id, cutoff, indices = self.items[index]
        return {
            "user_id": user_id,
            "cutoff": cutoff,
            "categorical": torch.from_numpy(self.base.encoded_categories[indices]).long(),
            "continuous": torch.from_numpy(self.base.continuous[indices]).float(),
        }


def _dense_cutoff_offsets(event_count: int, event_stride: int) -> list[int]:
    """Return event-aligned cutoff offsets, always retaining both endpoints."""
    if event_stride < 1:
        raise ValueError("event_stride must be at least 1")
    if event_count < 1:
        return []
    offsets = list(range(0, event_count, event_stride))
    if offsets[-1] != event_count - 1:
        offsets.append(event_count - 1)
    return offsets


class DenseUserCutoffDataset(Dataset[dict[str, Any]]):
    """Observed histories at event-aligned timestamps, without protected labels."""

    def __init__(
        self,
        observed_dir: str | Path,
        prepared_dir: str | Path,
        config: dict[str, Any],
        event_stride: int = 1,
    ) -> None:
        base = EventWindowDataset.__new__(EventWindowDataset)
        _, events = load_observed(observed_dir)
        base.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        base.vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
        base.categorical_fields = list(base.metadata["categorical_fields"])
        base.continuous_fields = list(base.metadata["continuous_fields"])
        base._validate_prepared_contract(config)
        base.events = events
        base.encoded_categories = base._encode_categories(events)
        base.continuous = base._encode_continuous(events)
        self.base = base
        self.maximum = int(config["data"]["max_sequence_length"])
        self.items: list[tuple[str, str, int, np.ndarray]] = []
        for user_id, indices in events.groupby("user_id", sort=False).indices.items():
            ordered = np.asarray(indices, dtype=np.int64)
            offsets = _dense_cutoff_offsets(len(ordered), event_stride)
            # Several services can emit at one instant. Keep the latest history
            # at that instant so the public dense key remains user/timestamp.
            by_timestamp = {
                pd.Timestamp(events.iloc[int(ordered[offset])]["timestamp"]): offset
                for offset in offsets
            }
            for offset in sorted(by_timestamp.values()):
                history_count = offset + 1
                event_index = int(ordered[offset])
                timestamp = pd.Timestamp(events.iloc[event_index]["timestamp"]).isoformat()
                history = ordered[max(0, history_count - self.maximum) : history_count]
                self.items.append((str(user_id), timestamp, history_count, history))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user_id, timestamp, history_count, indices = self.items[index]
        return {
            "user_id": user_id,
            "timestamp": timestamp,
            "cutoff_kind": "observed_event",
            "history_event_count": history_count,
            "categorical": torch.from_numpy(self.base.encoded_categories[indices]).long(),
            "continuous": torch.from_numpy(self.base.continuous[indices]).float(),
        }


def collate_user_cutoffs(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    categorical = [sample["categorical"] for sample in samples]
    continuous = [sample["continuous"] for sample in samples]
    batch = {
        "user_id": [sample["user_id"] for sample in samples],
        "categorical": pad_sequence(categorical, batch_first=True, padding_value=0),
        "continuous": pad_sequence(continuous, batch_first=True, padding_value=0.0),
        "lengths": torch.tensor([len(sequence) for sequence in categorical], dtype=torch.long),
    }
    for key in ("cutoff", "timestamp", "cutoff_kind", "history_event_count"):
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]
    return batch
