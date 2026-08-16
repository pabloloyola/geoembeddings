"""Observed-only context-session contrastive training support."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from zoneinfo import ZoneInfo

from .context_pair_preflight import validate_context_pair_manifest
from .data import EventWindowDataset


CONTEXT_CONTRASTIVE_SCHEMA = "geoembeddings-context-session-contrastive/1.0"


@dataclass(frozen=True)
class ContextTriplet:
    """One frozen positive anchor and its fixed same-user negative pairs."""

    user_id: str
    positive_pair_id: str
    anchor_group_id: str
    positive_group_id: str
    negative_pair_ids: tuple[str, ...]
    anchor_timestamp: str
    positive_timestamp: str
    negative_timestamps: tuple[str, ...]


def _selection_key(seed: int, epoch: int, triplet: ContextTriplet) -> str:
    material = "\0".join(
        [
            str(seed),
            str(epoch),
            triplet.user_id,
            triplet.positive_pair_id,
            *triplet.negative_pair_ids,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_epoch_triplets(
    triplets: Iterable[ContextTriplet],
    *,
    max_positive_anchors_per_user: int,
    seed: int,
    epoch: int,
) -> list[ContextTriplet]:
    """Select a deterministic, per-user-capped subset from the frozen manifest."""
    if max_positive_anchors_per_user < 1:
        raise ValueError("max_positive_anchors_per_user must be positive")
    by_user: dict[str, list[ContextTriplet]] = defaultdict(list)
    for triplet in triplets:
        by_user[triplet.user_id].append(triplet)
    selected: list[ContextTriplet] = []
    for user_id in sorted(by_user):
        ranked = sorted(
            by_user[user_id],
            key=lambda triplet: _selection_key(seed, epoch, triplet),
        )
        selected.extend(ranked[:max_positive_anchors_per_user])
    return sorted(selected, key=lambda triplet: (triplet.user_id, triplet.anchor_timestamp, triplet.positive_pair_id))


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("context contrastive timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC").isoformat()


def _local_day(value: Any, timezone_name: str) -> Any:
    try:
        timezone_value = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"unknown context local-day timezone: {timezone_name!r}") from exc
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("context local-day timestamps must be timezone-aware")
    return timestamp.tz_convert(timezone_value).date()


class FrozenContextTripletDataset(Dataset[dict[str, Any]]):
    """Encode frozen observed prefix triplets without reading protected truth."""

    def __init__(
        self,
        base_dataset: EventWindowDataset,
        manifest_path: str | Path,
        *,
        negative_pairs_per_anchor: int,
        max_sequence_length: int,
        expected_session_gap_hours: float = 6.0,
        expected_intervening_groups: int = 1,
        expected_local_day_timezone: str = "Asia/Tokyo",
        expected_same_local_day: bool = True,
    ) -> None:
        if negative_pairs_per_anchor < 1:
            raise ValueError("negative_pairs_per_anchor must be positive")
        self.base_dataset = base_dataset
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"frozen context pair manifest is missing: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_context_pair_manifest(manifest)
        self.manifest = manifest
        self._authenticate_manifest(
            manifest,
            expected_session_gap_hours=expected_session_gap_hours,
            expected_intervening_groups=expected_intervening_groups,
            expected_local_day_timezone=expected_local_day_timezone,
            expected_same_local_day=expected_same_local_day,
            expected_train_cutoff=base_dataset.metadata["train_end"],
        )
        if dict(manifest["preparation_authentication"].get("observed_source_hashes", {})) != dict(
            base_dataset.metadata.get("source_files", {})
        ):
            raise ValueError("context contrastive manifest does not match prepared observed sources")
        pairs = manifest["pairs"]
        positives = [pair for pair in pairs if pair["relation"] == "positive"]
        negatives = [pair for pair in pairs if pair["relation"] == "negative"]
        positive_by_anchor: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        negative_by_anchor: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pair in positives:
            positive_by_anchor[(str(pair["user_id"]), str(pair["anchor_group_id"]))].append(pair)
        for pair in negatives:
            negative_by_anchor[(str(pair["user_id"]), str(pair["anchor_group_id"]))].append(pair)

        self.triplets: list[ContextTriplet] = []
        for key in sorted(positive_by_anchor):
            positive_candidates = sorted(positive_by_anchor[key], key=lambda pair: str(pair["pair_id"]))
            if len(positive_candidates) != 1:
                raise ValueError("frozen manifest must contain exactly one positive per anchor")
            positive = positive_candidates[0]
            negative_candidates = sorted(
                negative_by_anchor.get(key, []), key=lambda pair: str(pair["pair_id"])
            )
            if len(negative_candidates) < negative_pairs_per_anchor:
                continue
            selected_negatives = negative_candidates[:negative_pairs_per_anchor]
            for negative in selected_negatives:
                if str(negative["user_id"]) != str(positive["user_id"]):
                    raise ValueError("context contrastive negative crosses users")
                if str(negative["anchor_group_id"]) != str(positive["anchor_group_id"]):
                    raise ValueError("context contrastive negative does not share the positive anchor")
                if str(negative["paired_session_id"]) == str(negative["anchor_session_id"]):
                    raise ValueError("context contrastive negative stays within the observed session")
            self.triplets.append(
                ContextTriplet(
                    user_id=str(positive["user_id"]),
                    positive_pair_id=str(positive["pair_id"]),
                    anchor_group_id=str(positive["anchor_group_id"]),
                    positive_group_id=str(positive["paired_group_id"]),
                    negative_pair_ids=tuple(str(pair["pair_id"]) for pair in selected_negatives),
                    anchor_timestamp=_canonical_timestamp(positive["anchor_timestamp"]),
                    positive_timestamp=_canonical_timestamp(positive["paired_timestamp"]),
                    negative_timestamps=tuple(
                        _canonical_timestamp(pair["paired_timestamp"]) for pair in selected_negatives
                    ),
                )
            )

        coverage = manifest.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError("frozen context manifest is missing authenticated coverage")
        positive_anchor_count = len(positive_by_anchor)
        joint_anchor_count = len(self.triplets)
        valid_anchor_count = int(coverage.get("valid_anchor_count", 0))
        if positive_anchor_count != int(coverage.get("positive_anchor_count", positive_anchor_count)):
            raise ValueError("frozen context manifest positive-anchor coverage is inconsistent")
        reported_positive_coverage = float(coverage.get("positive_anchor_coverage", 0.0))
        joint_coverage = joint_anchor_count / max(1, valid_anchor_count)
        self.joint_coverage_report = {
            "schema_version": CONTEXT_CONTRASTIVE_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": _sha256_bytes(self.manifest_path.read_bytes()),
            "negative_pairs_required_per_anchor": negative_pairs_per_anchor,
            "positive_anchor_count": positive_anchor_count,
            "joint_anchor_count": joint_anchor_count,
            "valid_anchor_count": valid_anchor_count,
            "reported_positive_anchor_coverage": reported_positive_coverage,
            "joint_anchor_coverage": joint_coverage,
            "joint_to_positive_anchor_ratio": joint_anchor_count / max(1, positive_anchor_count),
            "users_with_positive": len({triplet.user_id for triplet in self.triplets}),
            "truth_files_opened": False,
        }
        self._event_timestamps = pd.to_datetime(base_dataset.events["timestamp"], utc=True)
        self._user_indices: dict[str, list[int]] = {}
        for user_id, indices in base_dataset.events.groupby("user_id", sort=True).indices.items():
            ordered = sorted(
                (int(index) for index in indices),
                key=lambda index: (self._event_timestamps.iloc[index], index),
            )
            self._user_indices[str(user_id)] = ordered
        self.max_sequence_length = int(max_sequence_length)
        self._encoded_items: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _authenticate_manifest(
        manifest: dict[str, Any],
        *,
        expected_session_gap_hours: float,
        expected_intervening_groups: int,
        expected_local_day_timezone: str,
        expected_same_local_day: bool,
        expected_train_cutoff: str,
    ) -> None:
        auth = manifest["source_authentication"]
        preparation = manifest["preparation_authentication"]
        if auth.get("truth_files_opened") is not False:
            raise ValueError("context contrastive training requires a truth-free manifest")
        expected_sources = dict(preparation.get("observed_source_hashes", {}))
        actual_sources = dict(auth.get("observed_file_hashes", {}))
        if not expected_sources or expected_sources != actual_sources:
            raise ValueError("context contrastive manifest observed-source authentication failed")
        pair_config = manifest["pair_configuration"]
        if float(pair_config.get("session_gap_hours")) != float(expected_session_gap_hours):
            raise ValueError("context contrastive manifest is not frozen at the 6-hour session gap")
        if int(pair_config.get("min_intervening_groups_for_positive")) != int(expected_intervening_groups):
            raise ValueError("context contrastive manifest has the wrong intervening-group rule")
        if str(pair_config.get("positive_local_day_timezone")) != expected_local_day_timezone:
            raise ValueError("context contrastive manifest has the wrong local-day timezone")
        if bool(pair_config.get("positive_same_local_day")) is not expected_same_local_day:
            raise ValueError("context contrastive manifest has the wrong local-day policy")

        for pair in manifest["pairs"]:
            anchor = pd.Timestamp(pair["anchor_timestamp"])
            paired = pd.Timestamp(pair["paired_timestamp"])
            train_cutoff = pd.Timestamp(expected_train_cutoff)
            if anchor > train_cutoff or paired > train_cutoff:
                raise ValueError("context contrastive manifest contains a post-cutoff pair")
            if pair["relation"] == "positive":
                if _local_day(anchor, expected_local_day_timezone) != _local_day(
                    paired, expected_local_day_timezone
                ):
                    raise ValueError("context contrastive manifest contains a cross-local-day positive")

    def _prefix(self, user_id: str, timestamp: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (user_id, timestamp)
        if key in self._encoded_items:
            return self._encoded_items[key]
        cutoff = pd.Timestamp(timestamp)
        indices = [
            index for index in self._user_indices.get(user_id, [])
            if self._event_timestamps.iloc[index] < cutoff
        ]
        if not indices:
            raise ValueError("context contrastive pair has an empty observed prefix")
        indices = indices[-self.max_sequence_length :]
        categorical = torch.from_numpy(self.base_dataset.encoded_categories[indices]).long()
        continuous = torch.from_numpy(self.base_dataset.continuous[indices]).float()
        self._encoded_items[key] = (categorical, continuous)
        return categorical, continuous

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        triplet = self.triplets[index]
        anchor_categorical, anchor_continuous = self._prefix(triplet.user_id, triplet.anchor_timestamp)
        positive_categorical, positive_continuous = self._prefix(triplet.user_id, triplet.positive_timestamp)
        negative = [self._prefix(triplet.user_id, timestamp) for timestamp in triplet.negative_timestamps]
        return {
            "user_id": triplet.user_id,
            "positive_pair_id": triplet.positive_pair_id,
            "anchor_categorical": anchor_categorical,
            "anchor_continuous": anchor_continuous,
            "positive_categorical": positive_categorical,
            "positive_continuous": positive_continuous,
            "negative_categorical": [item[0] for item in negative],
            "negative_continuous": [item[1] for item in negative],
        }

    def selected(self, triplets: list[ContextTriplet]) -> "FrozenContextTripletDataset":
        result = object.__new__(FrozenContextTripletDataset)
        result.__dict__.update(self.__dict__)
        result.triplets = list(triplets)
        result._encoded_items = {}
        return result


def _pad(values: list[torch.Tensor], *, floating: bool) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence(values, batch_first=True, padding_value=0.0 if floating else 0)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    return padded, lengths


def collate_context_triplets(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    if not samples:
        raise ValueError("cannot collate an empty context triplet batch")
    batch: dict[str, Any] = {
        "user_id": [sample["user_id"] for sample in samples],
        "positive_pair_id": [sample["positive_pair_id"] for sample in samples],
    }
    for role in ("anchor", "positive"):
        categorical, lengths = _pad(
            [sample[f"{role}_categorical"] for sample in samples], floating=False
        )
        continuous, _ = _pad(
            [sample[f"{role}_continuous"] for sample in samples], floating=True
        )
        batch[f"{role}_categorical"] = categorical
        batch[f"{role}_continuous"] = continuous
        batch[f"{role}_lengths"] = lengths
    negative_count = len(samples[0]["negative_categorical"])
    if any(len(sample["negative_categorical"]) != negative_count for sample in samples):
        raise ValueError("context triplet batch has inconsistent negative counts")
    negative_categories: list[torch.Tensor] = []
    negative_continuous: list[torch.Tensor] = []
    for sample in samples:
        negative_categories.extend(sample["negative_categorical"])
        negative_continuous.extend(sample["negative_continuous"])
    categories, _ = _pad(negative_categories, floating=False)
    continuous, _ = _pad(negative_continuous, floating=True)
    batch["negative_categorical"] = categories.reshape(
        len(samples), negative_count, categories.shape[1], categories.shape[2]
    )
    batch["negative_continuous"] = continuous.reshape(
        len(samples), negative_count, continuous.shape[1], continuous.shape[2]
    )
    batch["negative_lengths"] = torch.tensor(
        [[len(value) for value in sample["negative_categorical"]] for sample in samples],
        dtype=torch.long,
    )
    return batch


class ContextProjectionHead(nn.Module):
    """The identical candidate/control projection head."""

    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0:
            raise ValueError("context projection dimensions must be positive")
        self.projection = nn.Linear(input_dim, projection_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.projection(context)


def context_infonce_loss(
    anchor_context: torch.Tensor,
    positive_context: torch.Tensor,
    negative_context: torch.Tensor,
    projection_head: ContextProjectionHead,
    *,
    temperature: float,
    detach_context: bool,
) -> torch.Tensor:
    """Compute context-only InfoNCE with optional exact context detachment."""
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    if anchor_context.ndim != 2 or positive_context.shape != anchor_context.shape:
        raise ValueError("anchor and positive contexts must have shape [batch, dimension]")
    if negative_context.ndim != 3 or negative_context.shape[0] != anchor_context.shape[0]:
        raise ValueError("negative contexts must have shape [batch, negatives, dimension]")
    if detach_context:
        anchor_context = anchor_context.detach()
        positive_context = positive_context.detach()
        negative_context = negative_context.detach()
    anchor = F.normalize(projection_head(anchor_context), dim=-1)
    positive = F.normalize(projection_head(positive_context), dim=-1)
    negative = F.normalize(projection_head(negative_context), dim=-1)
    positive_logits = (anchor * positive).sum(dim=-1, keepdim=True)
    negative_logits = (anchor.unsqueeze(1) * negative).sum(dim=-1)
    logits = torch.cat((positive_logits, negative_logits), dim=1) / float(temperature)
    return F.cross_entropy(logits, torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
