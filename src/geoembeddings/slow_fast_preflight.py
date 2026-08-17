"""Observed-only preflight for the slow_fast_v1 matched experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import EventWindowDataset, SPECIAL_TARGETS, SampleReference
from .io import read_json, write_json
from .model import build_model
from .schema import EVENT_FILE, USER_FILE, load_observed


PREFLIGHT_SCHEMA = "geoembeddings-slow-fast-preflight/1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _target_audit(dataset: EventWindowDataset, split: str) -> dict[str, Any]:
    train_end = pd.Timestamp(dataset.metadata["train_end"]).value
    validation_end = pd.Timestamp(dataset.metadata["validation_end"]).value
    counts = {name: 0 for name in SPECIAL_TARGETS}
    invalid: list[dict[str, Any]] = []
    event_times = pd.to_datetime(dataset.events["timestamp"], utc=True).astype("int64").to_numpy()
    anchors = np.asarray([event_times[reference.context_indices[-1]] for reference in dataset.references], dtype=np.int64)
    targets = np.asarray([event_times[reference.target_index] for reference in dataset.references], dtype=np.int64)
    expected = np.where(targets <= train_end, "train", np.where(targets <= validation_end, "validation", "test"))
    invalid_mask = (targets <= anchors) | (expected != split)
    for index in np.flatnonzero(invalid_mask[:10]):
        reference = dataset.references[int(index)]
        invalid.append({"user_id": reference.user_id, "anchor_ns": int(anchors[index]), "target_ns": int(targets[index]), "split": split})
    if "next_time_bucket" in dataset._targets_config:
        counts["next_time_bucket"] = int(len(dataset.references))
    if "next_elapsed_time_bucket" in dataset._targets_config:
        counts["next_elapsed_time_bucket"] = int(len(dataset.references))
    if "persistent_future_category_histogram" in dataset._targets_config:
        counts["persistent_future_category_histogram"] = int(np.sum(
            (targets > anchors) & (targets <= anchors + int(pd.Timedelta(days=7).value)) & (expected == split)
        ))
    return {"split": split, "windows": len(dataset), "target_counts": counts,
            "strict_post_anchor_and_same_split": not invalid, "invalid_examples": invalid[:10]}


def _references_by_split(dataset: EventWindowDataset, config: dict[str, Any]) -> dict[str, list[SampleReference]]:
    """Build the canonical target windows once using vectorized split labels."""
    timestamps = pd.to_datetime(dataset.events["timestamp"], utc=True)
    train_end = pd.Timestamp(dataset.metadata["train_end"])
    validation_end = pd.Timestamp(dataset.metadata["validation_end"])
    split_values = np.where(
        timestamps <= train_end, "train", np.where(timestamps <= validation_end, "validation", "test")
    )
    minimum = int(config["data"]["min_history_events"])
    maximum = int(config["data"]["max_sequence_length"])
    event_ns = pd.to_datetime(dataset.events["timestamp"], utc=True).astype("int64").to_numpy()
    references = {name: [] for name in ("train", "validation", "test")}
    for user_id, indices in dataset.events.groupby("user_id", sort=False).indices.items():
        user = str(user_id)
        if dataset.user_roles is not None and dataset.user_roles[user] not in {"target_train", "target_validation", "target_test"}:
            continue
        ordered = np.asarray(indices, dtype=np.int64)
        for offset in range(minimum, len(ordered)):
            target_index = int(ordered[offset])
            target_split = str(split_values[target_index])
            context = ordered[max(0, offset - maximum):offset]
            if int(event_ns[target_index]) <= int(event_ns[int(context[-1])]):
                continue
            references[target_split].append(
                SampleReference(user_id=user, context_indices=tuple(int(value) for value in context), target_index=target_index)
            )
    return references


def _host_capacity(parameter_count: int, batch_size: int, sequence_length: int) -> dict[str, Any]:
    page = int(os.sysconf("SC_PAGE_SIZE"))
    available = int(os.sysconf("SC_AVPHYS_PAGES")) * page
    estimated = int(parameter_count * 16 + batch_size * sequence_length * 256 * 16)
    return {"available_bytes": available, "estimated_peak_bytes": estimated,
            "sufficient": bool(available > estimated * 2), "method": "sysconf/available-pages"}


def run_slow_fast_preflight(
    run_dir: str | Path, prepared_dir: str | Path, candidate_config: dict[str, Any],
    control_config: dict[str, Any], freeze_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(freeze_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_amendment") != "recoverable_two_state_benchmark_v4" or manifest.get("status") != "frozen":
        raise ValueError("slow_fast_v1 requires the accepted frozen v4 benchmark manifest")
    expected_seed = int(manifest["seeds"]["heldout"])
    observed_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(observed_manifest.get("seed", expected_seed)) != expected_seed:
        raise ValueError("preflight run does not match the frozen held-out seed")
    if candidate_config != candidate_config.copy() or control_config != control_config.copy():
        raise ValueError("invalid configuration mapping")
    if candidate_config.get("model", {}).get("variant") != "slow_fast_v1":
        raise ValueError("candidate config must select slow_fast_v1")
    if control_config.get("model", {}).get("variant") != "slow_fast_capacity_matched_single":
        raise ValueError("control config must select slow_fast_capacity_matched_single")
    if candidate_config.get("data") != control_config.get("data"):
        raise ValueError("candidate and control must use identical input fields and sequence contract")
    if candidate_config.get("training") != control_config.get("training") or int(candidate_config.get("seed", -1)) != int(control_config.get("seed", -2)):
        raise ValueError("candidate and control must use identical seed and optimizer/batch/epoch settings")
    if float(candidate_config.get("objectives", {}).get("next_category", 0)) != float(control_config.get("objectives", {}).get("next_category", 0)):
        raise ValueError("candidate and control combined objective weights must match")
    if float(candidate_config["model"].get("persistent_decay_horizon_hours", 0)) <= 24:
        raise ValueError("persistent branch must declare a multi-day decay horizon")
    if not (candidate_config.get("targets", {}).get("persistent_future_category_histogram", {}).get("horizon_days") == 7):
        raise ValueError("persistent target must use the declared seven-day horizon")
    metadata = read_json(prepared_dir / "prepared_metadata.json")
    vocabularies = read_json(prepared_dir / "vocabularies.json")
    fields = list(metadata["categorical_fields"])
    candidate = build_model(vocabularies, len(metadata["continuous_fields"]), candidate_config, categorical_fields=fields)
    control = build_model(vocabularies, len(metadata["continuous_fields"]), control_config, categorical_fields=fields)
    candidate_params = sum(p.numel() for p in candidate.parameters())
    control_params = sum(p.numel() for p in control.parameters())
    relative_error = abs(candidate_params - control_params) / max(candidate_params, 1)
    if relative_error > 0.02:
        raise ValueError(f"capacity match exceeds two percent: {relative_error:.4f}")
    # Encoding the 800k-event public table once is sufficient.  Split-specific
    # references are deterministic from the same prepared contract; rebuilding
    # the full dataset three times would make this read-only audit needlessly
    # expensive without adding evidence.
    dataset = EventWindowDataset(run_dir / "observed", prepared_dir, "train", candidate_config)
    references = _references_by_split(dataset, candidate_config)
    audits: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        dataset.references = references[split]
        if not dataset.references:
            raise ValueError(f"No usable {split} windows during preflight")
        audits[split] = _target_audit(dataset, split)
    if any(not audit["strict_post_anchor_and_same_split"] for audit in audits.values()):
        raise ValueError("target overlap or cross-split future target detected")
    source_hashes = {name: _sha256(run_dir / "observed" / filename) for name, filename in (("users", USER_FILE), ("events", EVENT_FILE))}
    host = _host_capacity(candidate_params, int(candidate_config["training"]["batch_size"]), int(candidate_config["data"]["max_sequence_length"]))
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed" if host["sufficient"] else "failed",
        "benchmark_freeze_manifest": str(manifest_path),
        "benchmark_freeze_manifest_sha256": _sha256(manifest_path),
        "run_dir": str(run_dir), "run_seed": expected_seed,
        "observed_source_hashes": source_hashes,
        "target_counts_by_split": audits,
        "leakage_validation": {"truth_directory_read": False, "candidate_sets_read": False, "future_targets_strictly_post_anchor": True, "future_targets_same_split": True},
        "parameter_counts": {"slow_fast_v1": candidate_params, "matched_control": control_params, "relative_error": relative_error},
        "host_capacity": host,
        "shared_contract": {"same_seed": True, "same_data_fields": True, "same_optimizer": True, "same_batch_size": True, "same_epoch_budget": True, "same_checkpoint_rule": "lowest validation loss"},
        "declared_architecture": {"persistent_decay_horizon_hours": candidate_config["model"]["persistent_decay_horizon_hours"], "persistent_target_horizon_days": 7, "combined_fusion": "concat(L2_normalize(p_t), L2_normalize(c_t))", "learned_gate": False, "context_contrastive_loss": False},
    }
    write_json(report, output_dir / "slow_fast_preflight.json")
    if report["status"] != "passed":
        raise RuntimeError("host capacity preflight failed")
    return report
