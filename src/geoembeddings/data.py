from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .io import read_json, sha256_file
from .prepare import UNK_TOKEN, derive_continuous_features
from .schema import EVENT_FILE, USER_FILE, load_observed
from .user_roles import authenticate_roles


TARGET_FIELDS = {
    "next_service": "service_id",
    "next_action": "action_type",
    "next_category": "object_category",
    "next_region": "region_id",
    "next_geohash_5": "geohash_5",
    "next_geohash_7": "geohash_7",
}

SPECIAL_TARGETS = {
    "persistent_future_category_histogram": "distribution",
    "next_time_bucket": "classification",
    "next_elapsed_time_bucket": "classification",
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
        *,
        allow_source_drift: bool = False,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        users, events = load_observed(observed_dir)
        self.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        self.vocabularies: dict[str, dict[str, int]] = read_json(
            Path(prepared_dir) / "vocabularies.json"
        )
        self.categorical_fields = list(self.metadata["categorical_fields"])
        self.continuous_fields = list(self.metadata["continuous_fields"])
        self._validate_prepared_contract(config)
        self._targets_config = {
            name: value for name, value in config.get("targets", {}).items()
            if name in SPECIAL_TARGETS
        }
        current_sources = {USER_FILE: sha256_file(Path(observed_dir) / USER_FILE),
                           EVENT_FILE: sha256_file(Path(observed_dir) / EVENT_FILE)}
        if not allow_source_drift and current_sources != self.metadata.get("source_files"):
            raise ValueError("Observed source files changed after preparation")
        self.user_roles = authenticate_roles(self.metadata, config, users["user_id"].astype(str))
        self.events = events
        self.encoded_categories = self._encode_categories(events)
        self.continuous = self._encode_continuous(events)
        self.elapsed_hours = self._elapsed_hours(events)
        self.references = self._make_references(events, split, config["data"])
        self._future_user_index = self._build_future_user_index()
        self._special_target_cache: dict[int, tuple[dict[str, Any], dict[str, bool]]] = {}
        if not self.references:
            raise ValueError(f"No usable {split} windows after applying min_history_events")

    @property
    def participating_users(self) -> tuple[str, ...]:
        """Users with actual eligible target windows, independent of iteration order."""
        return tuple(sorted({reference.user_id for reference in self.references}))

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
        special_targets, target_masks = self._special_targets(reference)
        targets.update(special_targets)
        elapsed = torch.from_numpy(self.elapsed_hours[context_indices]).float()
        return {
            "user_id": reference.user_id,
            "categorical": categorical,
            "continuous": continuous,
            "elapsed_hours": elapsed,
            "early_categorical": torch.from_numpy(self.encoded_categories[early_indices]).long(),
            "early_continuous": torch.from_numpy(self.continuous[early_indices]).float(),
            "late_categorical": torch.from_numpy(self.encoded_categories[late_indices]).long(),
            "late_continuous": torch.from_numpy(self.continuous[late_indices]).float(),
            "early_elapsed_hours": torch.from_numpy(self.elapsed_hours[early_indices]).float(),
            "late_elapsed_hours": torch.from_numpy(self.elapsed_hours[late_indices]).float(),
            "targets": targets,
            "target_masks": target_masks,
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

    @staticmethod
    def _elapsed_hours(events: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(events["timestamp"], utc=True)
        delta = (
            events.assign(_timestamp=timestamps)
            .groupby("user_id", sort=False)["_timestamp"]
            .diff()
            .dt.total_seconds()
            .div(3600.0)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        return delta.to_numpy(dtype=np.float32)

    def _special_targets(self, reference: SampleReference) -> tuple[dict[str, Any], dict[str, bool]]:
        cached = self._special_target_cache.get(reference.target_index)
        if cached is not None:
            return cached
        targets: dict[str, Any] = {}
        masks: dict[str, bool] = {}
        if not self._targets_config:
            return targets, masks
        anchor_index = int(reference.context_indices[-1])
        anchor_time = pd.Timestamp(self.events.iloc[anchor_index]["timestamp"])
        target_time = pd.Timestamp(self.events.iloc[reference.target_index]["timestamp"])
        if "next_time_bucket" in self._targets_config:
            edges = [float(value) for value in self._targets_config["next_time_bucket"]["edges_hours"]]
            local = target_time.tz_convert("Asia/Tokyo")
            targets["next_time_bucket"] = _bucket_index(local.hour + local.minute / 60.0, edges)
            masks["next_time_bucket"] = True
        if "next_elapsed_time_bucket" in self._targets_config:
            edges = [float(value) for value in self._targets_config["next_elapsed_time_bucket"]["edges_hours"]]
            delta = max(0.0, (target_time - anchor_time).total_seconds() / 3600.0)
            targets["next_elapsed_time_bucket"] = _bucket_index(delta, edges)
            masks["next_elapsed_time_bucket"] = True
        if "persistent_future_category_histogram" in self._targets_config:
            spec = self._targets_config["persistent_future_category_histogram"]
            user_index = self._future_user_index[reference.user_id]
            times = user_index["times_ns"]
            anchor_ns = int(anchor_time.value)
            split = self._timestamp_split_for(reference.target_index)
            split_start, split_end = self._split_bounds_ns(split)
            lower = max(anchor_ns, split_start)
            upper = min(anchor_ns + int(pd.Timedelta(days=float(spec["horizon_days"])).value), split_end)
            left = int(np.searchsorted(times, lower, side="right"))
            right = int(np.searchsorted(times, upper, side="right"))
            histogram = user_index["prefix_counts"][right] - user_index["prefix_counts"][left]
            valid = bool(histogram.sum() > 0)
            if valid:
                histogram /= histogram.sum()
            targets["persistent_future_category_histogram"] = histogram
            masks["persistent_future_category_histogram"] = valid
        result = (targets, masks)
        self._special_target_cache[reference.target_index] = result
        return result

    def _build_future_user_index(self) -> dict[str, dict[str, np.ndarray]]:
        vocabulary = self.vocabularies["object_category"]
        unknown = int(vocabulary[UNK_TOKEN])
        result: dict[str, dict[str, np.ndarray]] = {}
        for user_id, rows in self.events.groupby("user_id", sort=False):
            ordered = rows.sort_values("timestamp")
            times_ns = pd.to_datetime(ordered["timestamp"], utc=True).astype("int64").to_numpy()
            codes = ordered["object_category"].fillna(UNK_TOKEN).astype(str).map(
                lambda value: vocabulary.get(value, unknown)
            ).to_numpy(dtype=np.int64)
            one_hot = np.zeros((len(codes), len(vocabulary)), dtype=np.float32)
            if len(codes):
                one_hot[np.arange(len(codes)), codes] = 1.0
            result[str(user_id)] = {
                "times_ns": times_ns,
                "prefix_counts": np.vstack([np.zeros((1, len(vocabulary)), dtype=np.float32), np.cumsum(one_hot, axis=0)]),
            }
        return result

    def _split_bounds_ns(self, split: str) -> tuple[int, int]:
        minimum = pd.Timestamp.min.value
        maximum = pd.Timestamp.max.value
        train_end = pd.Timestamp(self.metadata["train_end"]).value
        validation_end = pd.Timestamp(self.metadata["validation_end"]).value
        if split == "train":
            return minimum, train_end
        if split == "validation":
            return train_end, validation_end
        if split == "test":
            return validation_end, maximum
        raise ValueError(f"Unknown timestamp split: {split}")

    def _timestamp_split_for(self, index: int) -> str:
        return _timestamp_split(
            pd.Timestamp(self.events.iloc[index]["timestamp"]),
            pd.Timestamp(self.metadata["train_end"]),
            pd.Timestamp(self.metadata["validation_end"]),
        )

    def _timestamp_split_value(self, value: Any) -> str:
        return _timestamp_split(
            pd.Timestamp(value),
            pd.Timestamp(self.metadata["train_end"]),
            pd.Timestamp(self.metadata["validation_end"]),
        )

    def _make_references(
        self,
        events: pd.DataFrame,
        split: str,
        data_config: dict[str, Any],
    ) -> list[SampleReference]:
        train_end = pd.Timestamp(self.metadata["train_end"]).value
        validation_end = pd.Timestamp(self.metadata["validation_end"]).value
        minimum = int(data_config["min_history_events"])
        maximum = int(data_config["max_sequence_length"])
        event_times = pd.to_datetime(events["timestamp"], utc=True).astype("int64").to_numpy()
        split_values = np.where(event_times <= train_end, "train", np.where(event_times <= validation_end, "validation", "test"))
        references: list[SampleReference] = []
        for user_id, indices in events.groupby("user_id", sort=False).indices.items():
            if self.user_roles is not None and self.user_roles[str(user_id)] != f"target_{split}":
                continue
            ordered = np.asarray(indices, dtype=np.int64)
            for offset in range(minimum, len(ordered)):
                target_index = int(ordered[offset])
                if split_values[target_index] != split:
                    continue
                context = ordered[max(0, offset - maximum) : offset]
                if event_times[target_index] <= event_times[int(context[-1])]:
                    # A target must be strictly after the final visible event;
                    # rows sharing a timestamp are an atomic observation group.
                    continue
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


def _bucket_index(value: float, edges: list[float]) -> int:
    values = np.asarray(edges, dtype=float)
    if len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("target bucket edges must be strictly increasing")
    return min(int(np.searchsorted(values, value, side="right") - 1), len(values) - 2)


PARTICIPATION_HASH_DEFINITION = "sha256-canonical-sorted-identifiers/1.0"


def canonical_user_set(users: Iterable[str]) -> dict[str, Any]:
    """Return a non-identifying, deterministic identity-set summary."""
    values = sorted({str(user) for user in users})
    if any(not value for value in values):
        raise ValueError("participation identities must be non-empty strings")
    return {
        "count": len(values),
        "identity_sha256": hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest(),
    }


def participation_roles(
    train_dataset: EventWindowDataset,
    validation_dataset: EventWindowDataset,
) -> dict[str, Any]:
    """Derive training roles solely from observed preprocessing and model datasets."""
    train_users = set(train_dataset.participating_users)
    validation_users = set(validation_dataset.participating_users)
    train_end = pd.Timestamp(train_dataset.metadata["train_end"])
    preprocessing_users = set(
        train_dataset.events.loc[train_dataset.events["timestamp"] <= train_end, "user_id"]
        .astype(str)
    )
    if getattr(train_dataset, "user_roles", None) is not None:
        preprocessing_users &= {user for user, role in train_dataset.user_roles.items()
                                if role == "target_train"}
    export_users = set(train_dataset.events["user_id"].astype(str))
    export_only_users = export_users - train_users - validation_users
    # A frozen checkpoint may export one row for every available named cutoff.
    boundaries = (
        train_end,
        pd.Timestamp(train_dataset.metadata["validation_end"]),
        pd.Timestamp(train_dataset.metadata.get("timestamp_max", train_dataset.events["timestamp"].max())),
    )
    export_only_windows = sum(
        int((group["timestamp"] <= cutoff).any())
        for user, group in train_dataset.events.groupby("user_id", sort=False)
        if str(user) in export_only_users
        for cutoff in boundaries
    )
    return {
        "eligible_training_windows": {
            **canonical_user_set(train_users), "window_count": len(train_dataset),
        },
        "validation_checkpoint_selection_windows": {
            **canonical_user_set(validation_users), "window_count": len(validation_dataset),
        },
        "train_fitted_preprocessing": {
            **canonical_user_set(preprocessing_users),
            "event_count": int((train_dataset.events["timestamp"] <= train_end).sum()),
            "window_count": 0,
        },
        "exported_only_after_checkpoint_freezing": {
            **canonical_user_set(export_only_users), "window_count": export_only_windows,
        },
    }


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
        elapsed_key = f"{prefix}elapsed_hours"
        batch[elapsed_key] = pad_sequence(
            [sample[elapsed_key] for sample in samples], batch_first=True, padding_value=0.0
        )

    target_names = sorted(samples[0]["targets"])
    batch["targets"] = {}
    batch["target_masks"] = {}
    for name in target_names:
        values = [sample["targets"][name] for sample in samples]
        if np.asarray(values[0]).ndim == 1:
            batch["targets"][name] = torch.tensor(np.asarray(values), dtype=torch.float32)
        else:
            batch["targets"][name] = torch.tensor(values, dtype=torch.long)
        batch["target_masks"][name] = torch.tensor(
            [bool(sample.get("target_masks", {}).get(name, True)) for sample in samples],
            dtype=torch.bool,
        )
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
        allow_source_drift: bool = False,
    ) -> None:
        base = EventWindowDataset.__new__(EventWindowDataset)
        if events is None:
            users, events = load_observed(observed_dir)
        else:
            users, _ = load_observed(observed_dir)
        events = events.copy()
        base.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        base.vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
        base.categorical_fields = list(base.metadata["categorical_fields"])
        base.continuous_fields = list(base.metadata["continuous_fields"])
        base._validate_prepared_contract(config)
        current_sources = {USER_FILE: sha256_file(Path(observed_dir) / USER_FILE),
                           EVENT_FILE: sha256_file(Path(observed_dir) / EVENT_FILE)}
        if not allow_source_drift and current_sources != base.metadata.get("source_files"):
            raise ValueError("Observed source files changed after preparation")
        base.user_roles = authenticate_roles(base.metadata, config, users["user_id"].astype(str))
        base.events = events
        base.encoded_categories = base._encode_categories(events)
        base.continuous = base._encode_continuous(events)
        base.elapsed_hours = base._elapsed_hours(events)
        self.base = base
        self.maximum = int(config["data"]["max_sequence_length"])
        self.items: list[tuple[str, str, np.ndarray]] = []
        cutoffs = {
            "train": pd.Timestamp(base.metadata["train_end"]),
            "validation": pd.Timestamp(base.metadata["validation_end"]),
            "test": pd.Timestamp(base.metadata.get("timestamp_max", events["timestamp"].max())),
        }
        for user_id, indices in events.groupby("user_id", sort=False).indices.items():
            if base.user_roles is not None and base.user_roles[str(user_id)] != "target_test":
                continue
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
            "elapsed_hours": torch.from_numpy(self.base.elapsed_hours[indices]).float(),
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
        *,
        allow_source_drift: bool = False,
    ) -> None:
        base = EventWindowDataset.__new__(EventWindowDataset)
        users, events = load_observed(observed_dir)
        base.metadata = read_json(Path(prepared_dir) / "prepared_metadata.json")
        base.vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
        base.categorical_fields = list(base.metadata["categorical_fields"])
        base.continuous_fields = list(base.metadata["continuous_fields"])
        base._validate_prepared_contract(config)
        current_sources = {USER_FILE: sha256_file(Path(observed_dir) / USER_FILE),
                           EVENT_FILE: sha256_file(Path(observed_dir) / EVENT_FILE)}
        if not allow_source_drift and current_sources != base.metadata.get("source_files"):
            raise ValueError("Observed source files changed after preparation")
        base.user_roles = authenticate_roles(base.metadata, config, users["user_id"].astype(str))
        base.events = events
        base.encoded_categories = base._encode_categories(events)
        base.continuous = base._encode_continuous(events)
        base.elapsed_hours = base._elapsed_hours(events)
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
            "elapsed_hours": torch.from_numpy(self.base.elapsed_hours[indices]).float(),
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
        "elapsed_hours": pad_sequence(
            [sample["elapsed_hours"] for sample in samples], batch_first=True, padding_value=0.0
        ),
    }
    for key in ("cutoff", "timestamp", "cutoff_kind", "history_event_count"):
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]
    return batch
