"""Observed-only atomic online representation updates (T4.4 / R13)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .baseline import _encode_categories, _encode_continuous
from .io import read_json, sha256_file
from .model import build_model
from .prepare import UNK_TOKEN
from .representation_schema import COMPONENT_NAMES
from .training import _checkpoint_categorical_fields, resolve_device

TOLERANCE_ATOL = 1e-5
TOLERANCE_RTOL = 1e-4


@dataclass(frozen=True)
class OnlineRepresentation:
    user_id: str
    event_count: int
    components: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        frozen = []
        for name, value in self.components:
            array = np.array(value, copy=True)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError("representation components must be finite vectors")
            array.setflags(write=False)
            frozen.append((name, array))
        object.__setattr__(self, "components", tuple(frozen))


@dataclass(frozen=True)
class UpdateResult:
    representations: tuple[OnlineRepresentation, ...]
    input_events: int
    accepted_events: int
    duplicate_events: int
    maximum_absolute_error: float
    maximum_relative_error: float


@dataclass(frozen=True)
class WorkloadBoundary:
    name: str
    calls: tuple[tuple[str, ...], ...]
    requested_batch_size: int
    exclusions: tuple[str, ...] = ()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def event_fingerprint(row: Mapping[str, Any], field_order: Sequence[str]) -> str:
    def clean(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value
    missing = [name for name in field_order if name not in row]
    if missing:
        raise ValueError(f"missing required event fields: {missing}")
    return canonical_hash([[name, clean(row[name])] for name in field_order])


class AtomicOnlineState:
    """Transactional histories bound to one authenticated representation."""

    def __init__(self, *, field_order: Sequence[str], component_names: Sequence[str],
                 compute: Callable[[pd.DataFrame], Mapping[str, np.ndarray]], maximum_history: int,
                 identity: Mapping[str, Any]) -> None:
        self.field_order = tuple(field_order)
        self.component_names = tuple(component_names)
        self.maximum_history = int(maximum_history)
        self.identity = copy.deepcopy(dict(identity))
        self._compute = compute
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._fingerprints: dict[str, dict[str, str]] = {}
        self._outputs: dict[str, OnlineRepresentation] = {}

    @property
    def outputs(self) -> Mapping[str, OnlineRepresentation]:
        return dict(self._outputs)

    def append(self, rows: Sequence[Mapping[str, Any]], *, oracle: Callable[[pd.DataFrame], Mapping[str, np.ndarray]] | None = None) -> UpdateResult:
        if not rows:
            raise ValueError("atomic append requires at least one event")
        histories = copy.deepcopy(self._histories)
        fingerprints = copy.deepcopy(self._fingerprints)
        affected: set[str] = set(); duplicates = 0; accepted = 0
        candidates: list[tuple[str, pd.Timestamp, str, dict[str, Any]]] = []
        for raw in rows:
            row = dict(raw)
            user = str(row.get("user_id", ""))
            if not user:
                raise ValueError("event user_id is required")
            try:
                timestamp = pd.to_datetime(row.get("timestamp"), utc=True)
            except Exception as exc:
                raise ValueError("invalid event timestamp") from exc
            if pd.isna(timestamp):
                raise ValueError("invalid event timestamp")
            row["timestamp"] = timestamp
            fingerprint = event_fingerprint(row, self.field_order)
            candidates.append((user, timestamp, fingerprint, row))
        # Canonicalization occurs before any state mutation.
        seen_call: set[tuple[str, str]] = set()
        for user, timestamp, fingerprint, row in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            key = (user, fingerprint)
            known = fingerprints.setdefault(user, {})
            if key in seen_call or fingerprint in known:
                if fingerprint in known and known[fingerprint] != canonical_hash([[f, str(row[f])] for f in self.field_order]):
                    raise ValueError("event fingerprint collision")
                duplicates += 1; seen_call.add(key); continue
            seen_call.add(key)
            previous = histories.get(user, [])
            if previous:
                last = previous[-1]
                last_key = (pd.Timestamp(last["timestamp"]), str(last["_fingerprint"]))
                if (timestamp, fingerprint) <= last_key:
                    raise ValueError("out-of-order or late event rejects the atomic batch")
            stored = dict(row); stored["_fingerprint"] = fingerprint
            histories.setdefault(user, []).append(stored)
            known[fingerprint] = canonical_hash([[f, str(row[f])] for f in self.field_order])
            affected.add(user); accepted += 1

        produced: list[OnlineRepresentation] = []; max_abs = max_rel = 0.0
        for user in sorted(affected):
            frame = pd.DataFrame([{k: v for k, v in row.items() if k != "_fingerprint"}
                                  for row in histories[user][-self.maximum_history:]])
            incremental = self._compute(frame)
            recomputed = (oracle or self._compute)(frame.copy(deep=True))
            if tuple(incremental) != self.component_names or tuple(recomputed) != self.component_names:
                raise ValueError("component schema mismatch")
            components = []
            for name in self.component_names:
                left = np.asarray(incremental[name]); right = np.asarray(recomputed[name])
                if left.shape != right.shape or left.dtype != right.dtype or not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise ValueError("oracle identity, shape, dtype, or finiteness mismatch")
                difference = np.abs(left - right)
                relative = difference / np.maximum(np.abs(right), TOLERANCE_ATOL)
                max_abs = max(max_abs, float(difference.max(initial=0)))
                max_rel = max(max_rel, float(relative.max(initial=0)))
                if not np.allclose(left, right, atol=TOLERANCE_ATOL, rtol=TOLERANCE_RTOL):
                    raise ValueError("incremental/full recomputation correctness mismatch")
                components.append((name, left))
            produced.append(OnlineRepresentation(user, len(histories[user]), tuple(components)))
        # Commit only after every user and oracle check succeeds.
        self._histories, self._fingerprints = histories, fingerprints
        self._outputs = {**self._outputs, **{item.user_id: item for item in produced}}
        return UpdateResult(tuple(produced), len(rows), accepted, duplicates, max_abs, max_rel)


def baseline_computer(prepared_dir: str | Path, config: Mapping[str, Any]) -> tuple[Callable[[pd.DataFrame], Mapping[str, np.ndarray]], dict[str, Any]]:
    prepared = Path(prepared_dir); metadata = read_json(prepared / "prepared_metadata.json")
    vocabularies = read_json(prepared / "vocabularies.json")
    fields = list(metadata["categorical_fields"]); continuous = list(metadata["continuous_fields"])
    for name in fields:
        if UNK_TOKEN not in vocabularies[name]: raise ValueError(f"vocabulary {name} lacks unknown token")
    def compute(events: pd.DataFrame) -> Mapping[str, np.ndarray]:
        categorical = _encode_categories(events, fields, vocabularies)
        values = _encode_continuous(events, continuous, metadata); parts = []
        if not np.isfinite(values).all(): raise ValueError("non-finite continuous event feature")
        for index, name in enumerate(fields):
            counts = np.bincount(categorical[:, index], minlength=len(vocabularies[name])).astype(np.float32)
            counts[0] = 0; parts.append(counts / max(counts.sum(), 1.0))
        parts.extend((values.mean(0).astype(np.float32), values.std(0).astype(np.float32)))
        return {"combined": np.concatenate(parts).astype(np.float32)}
    identity = {"kind": "baseline", "role": "diagnostic_control", "prepared_metadata_sha256": sha256_file(prepared/"prepared_metadata.json")}
    return compute, identity


def learned_computer(prepared_dir: str | Path, checkpoint_path: str | Path, device_name: str = "cpu") -> tuple[Callable[[pd.DataFrame], Mapping[str, np.ndarray]], tuple[str, ...], dict[str, Any]]:
    prepared = Path(prepared_dir); checkpoint_path = Path(checkpoint_path); metadata = read_json(prepared/"prepared_metadata.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("categorical_fields") != metadata["categorical_fields"] or checkpoint.get("continuous_fields") != metadata["continuous_fields"]:
        raise ValueError("checkpoint field order mismatch")
    device = resolve_device(device_name); model = build_model(checkpoint["vocabularies"], len(checkpoint["continuous_fields"]), checkpoint["config"], categorical_fields=_checkpoint_categorical_fields(checkpoint, metadata["categorical_fields"])).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    names = tuple(checkpoint.get("representation_schema", {}).get("component_names", COMPONENT_NAMES))
    def compute(events: pd.DataFrame) -> Mapping[str, np.ndarray]:
        categorical = _encode_categories(events, metadata["categorical_fields"], checkpoint["vocabularies"])
        continuous = _encode_continuous(events, metadata["continuous_fields"], metadata)
        if not np.isfinite(continuous).all(): raise ValueError("non-finite continuous event feature")
        lengths = torch.tensor([len(events)], dtype=torch.long)  # CPU control metadata.
        with torch.no_grad():
            encoded = model.encode_components(torch.from_numpy(categorical)[None].to(device), torch.from_numpy(continuous)[None].to(device), lengths, augment=False)
        return {name: getattr(encoded, name)[0].detach().cpu().numpy() for name in names}
    identity = {"kind": "learned", "role": "diagnostic_control", "checkpoint_sha256": sha256_file(checkpoint_path), "model_variant": checkpoint.get("model_variant", "single_vector"), "device": str(device)}
    return compute, names, identity
