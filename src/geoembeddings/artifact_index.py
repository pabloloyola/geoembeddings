"""Build an immutable inventory and comparability audit for experiment evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml

from .contract import OBSERVED_FILES
from .io import read_json, sha256_file, write_json
from .layout import DatasetLayout, ExperimentLayout

SCHEMA_VERSION = "geoembeddings-evidence-index/1.0"
INSPECTION_SCHEMA_VERSION = "geoembeddings-evidence-inspection/1.0"


def inspect_evidence_indexes(
    index_dir: str | Path = "docs/artifacts", *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Verify all immutable evidence indexes without retrieving or changing artifacts."""
    repository = Path(repository_root or Path.cwd()).expanduser().resolve()
    directory = Path(index_dir).expanduser()
    if not directory.is_absolute():
        directory = repository / directory
    index_paths = sorted(directory.glob("*.json"))
    reports = [_inspect_evidence_index(path, repository) for path in index_paths]
    artifacts = [artifact for report in reports for artifact in report["artifacts"]]
    counts = {
        key: sum(artifact["availability"] == key for artifact in artifacts)
        for key in ("present_local", "locally_absent", "intentionally_external", "historically_lost")
    }
    counts["total"] = len(artifacts)
    counts["content_verified"] = sum(artifact["content_verified"] is True for artifact in artifacts)
    counts["content_mismatch"] = sum(artifact["content_verified"] is False for artifact in artifacts)
    return {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "read_only": True,
        "index_directory": normalize_identifier(directory, base=repository),
        "index_count": len(reports),
        "summary": counts,
        "ci_status": "mismatch" if counts["content_mismatch"] else "ok",
        "indexes": reports,
        "limitations": [
            "Absent artifacts are evidence-availability states, not failed scientific results.",
            "No artifact was downloaded and no evidence index was modified.",
        ],
    }


def _inspect_evidence_index(path: Path, repository: Path) -> dict[str, Any]:
    index = read_json(path)
    lost = "lost" in str(index.get("evidence_status", "")).lower() or bool(index.get("loss_audit"))
    artifacts = []
    seen: set[tuple[str, str | None]] = set()
    for entry in _artifact_entries(index):
        identifier = entry.get("identifier", entry.get("path"))
        expected_hash = entry.get("sha256")
        key = (str(identifier), expected_hash if isinstance(expected_hash, str) else None)
        if key in seen:
            continue
        seen.add(key)
        artifacts.append(_inspect_artifact(entry, repository, lost=lost))
    commands = _string_list(index.get("commands"))
    identity = {
        "task_id": index.get("task_id"),
        "index_location": index.get("index_location", normalize_identifier(path, base=repository)),
        "source_commit": index.get("provenance", {}).get("source_commit"),
        "historical_roots": index.get("storage", {}).get("unavailable_historical_roots", []),
    }
    return {
        "index": normalize_identifier(path, base=repository),
        "task_id": index.get("task_id"),
        "evidence_status": index.get("evidence_status", "indexed"),
        "artifacts": artifacts,
        "index_alone_sufficient_for_documentation_claims": False,
        "index_sufficiency": (
            "disposition_only" if lost else "identity_inventory_only"
        ),
        "index_sufficiency_reason": (
            "The index can document historical loss, but cannot substitute for unavailable bytes."
            if lost else "The index records identity and provenance; claims must use authenticated artifact contents and their stated scope."
        ),
        "rerun_commands_for_new_lineage": commands,
        "rerun_guidance": (
            "Run the recorded commands when available, using new run, experiment, and index names; otherwise follow docs/COMMAND_REFERENCE.md."
        ),
        "historical_identity_must_never_be_reused": identity,
    }


def _artifact_entries(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        identifier = value.get("identifier", value.get("path"))
        if isinstance(identifier, str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from _artifact_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_entries(child)


def _inspect_artifact(entry: dict[str, Any], repository: Path, *, lost: bool) -> dict[str, Any]:
    identifier = str(entry.get("identifier", entry.get("path")))
    parsed = urlsplit(identifier)
    external = bool(parsed.scheme and parsed.scheme != "file")
    local_path = None if external else Path(parsed.path if parsed.scheme == "file" else identifier).expanduser()
    if local_path is not None and not local_path.is_absolute():
        local_path = repository / local_path
    present = bool(local_path and local_path.is_file())
    actual_bytes = local_path.stat().st_size if present and local_path is not None else None
    actual_hash = sha256_file(local_path) if present and local_path is not None else None
    expected_bytes = entry.get("bytes") if isinstance(entry.get("bytes"), int) else None
    expected_hash = entry.get("sha256")
    byte_match = (actual_bytes == expected_bytes) if present and expected_bytes is not None else None
    hash_match = (actual_hash == expected_hash) if present else None
    status = str(entry.get("status", "")).lower()
    availability = (
        "present_local" if present else
        "intentionally_external" if external or status in {"external", "remote", "remote-only"} else
        "historically_lost" if lost or status in {"lost", "unavailable"} else
        "locally_absent"
    )
    return {
        "id": entry.get("id"),
        "identifier": identifier,
        "availability": availability,
        "present_locally": present,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "byte_count_matches": byte_match,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "sha256_matches": hash_match,
        "content_verified": (hash_match and byte_match is not False) if present else None,
    }


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def stable_values_hash(values: Iterable[str]) -> str:
    """Hash a set-like collection independently of input ordering."""
    normalized = sorted({str(value) for value in values})
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def normalize_identifier(value: str | Path, *, base: str | Path | None = None) -> str:
    """Return a stable POSIX identifier without requiring or modifying its target."""
    raw = str(value)
    parsed = urlsplit(raw)
    if parsed.scheme and parsed.scheme != "file":
        path = "/".join(part for part in parsed.path.split("/") if part)
        path = f"/{path}" if parsed.path.startswith("/") else path
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, parsed.fragment))
    path = Path(parsed.path if parsed.scheme == "file" else raw).expanduser().resolve(strict=False)
    if base is not None:
        try:
            path = path.relative_to(Path(base).expanduser().resolve(strict=False))
        except ValueError:
            pass
    return path.as_posix()


def build_artifact_index(
    run_dir: str | Path,
    experiment_dir: str | Path,
    output: str | Path,
    *,
    task_id: str = "T0.1a/T0.2",
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = DatasetLayout.from_path(run_dir)
    experiment = ExperimentLayout.from_path(experiment_dir)
    manifest = dataset.validate()
    repository = Path(repository_root or Path.cwd()).resolve()
    metadata = read_json(experiment.prepared_metadata)
    _validate_observed_sources(dataset, metadata)
    if manifest.get("dataset_contract") != metadata.get("dataset_contract"):
        raise ValueError(
            "Preparation contract mismatch: "
            f"manifest={manifest.get('dataset_contract')!r}, prepared={metadata.get('dataset_contract')!r}"
        )
    if manifest.get("users") is not None and int(manifest["users"]) != int(metadata["rows"]["users"]):
        raise ValueError(
            f"Cohort size mismatch: manifest={manifest['users']!r}, prepared={metadata['rows']['users']!r}"
        )

    baseline = _embedding_identity(experiment.baseline_embeddings, "baseline")
    learned = _embedding_identity(experiment.embeddings, "learned")
    diagnostics = _compare_identities(baseline, learned)
    baseline_dense = _embedding_identity(experiment.dense_baseline_embeddings, "baseline dense", dense=True)
    learned_dense = _embedding_identity(experiment.dense_embeddings, "learned dense", dense=True)
    diagnostics.extend(_compare_identities(baseline_dense, learned_dense, label="dense "))
    for kind in ("baseline", "learned"):
        for path in sorted(experiment.robustness_view_dir(kind).glob("*.npz")):
            _embedding_identity(path, f"{kind} robustness view")
    _validate_report_identity(experiment, metadata, diagnostics)
    if diagnostics:
        raise ValueError("Artifact comparability audit failed:\n- " + "\n- ".join(diagnostics))

    simulation_config = _read_yaml(dataset.resolved_config)
    embedding_config = _read_yaml(experiment.resolved_config)
    metadata_hash = sha256_file(experiment.prepared_metadata)
    source_hashes = _canonical_source_hashes(metadata["source_files"])
    shared = [
        ("dataset_manifest", dataset.manifest_path),
        ("resolved_simulation_config", dataset.resolved_config),
        ("deep_validation_report", dataset.deep_validation_report),
        ("prepared_metadata", experiment.prepared_metadata),
        ("resolved_embedding_config", experiment.resolved_config),
        ("vocabularies", experiment.vocabularies),
        ("command_log", experiment.command_log),
    ]
    groups = {
        "shared": shared,
        "baseline": [
            ("baseline_cutoff_export", experiment.baseline_embeddings),
            ("baseline_dense_export", experiment.dense_baseline_embeddings),
            ("baseline_evaluation_report", experiment.baseline_evaluation),
            ("baseline_episode_response_report", experiment.baseline_episode_response),
            ("baseline_robustness_report", experiment.robustness_report("baseline")),
            ("baseline_transfer_report", experiment.transfer_evaluation("baseline")),
            ("baseline_temporal_routine_report", experiment.temporal_routine_evaluation("baseline")),
            ("baseline_reliability_report", experiment.reliability_evaluation("baseline")),
        ],
        "learned": [
            ("learned_checkpoint", experiment.checkpoint),
            ("learned_training_report", experiment.training_report),
            ("learned_cutoff_export", experiment.embeddings),
            ("learned_dense_export", experiment.dense_embeddings),
            ("learned_evaluation_report", experiment.evaluation),
            ("learned_episode_response_report", experiment.episode_response),
            ("learned_robustness_report", experiment.robustness_report("learned")),
            ("learned_transfer_report", experiment.transfer_evaluation("learned")),
            ("learned_temporal_routine_report", experiment.temporal_routine_evaluation("learned")),
            ("learned_reliability_report", experiment.reliability_evaluation("learned")),
        ],
        "robustness_views": [
            (f"{kind}_robustness_view_{path.stem}", path)
            for kind in ("baseline", "learned")
            for path in sorted(experiment.robustness_view_dir(kind).glob("*.npz"))
        ],
        "comparison": [
            ("embedding_comparison_json", experiment.comparison_json),
            ("embedding_comparison_markdown", experiment.comparison_markdown),
        ],
        "benchmarks": [
            ("offline_benchmark", experiment.offline_benchmark),
            ("online_benchmark", experiment.online_benchmark),
        ],
    }
    missing = [str(path) for entries in groups.values() for _, path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required artifacts are missing: " + ", ".join(missing))

    index = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "index_location": normalize_identifier(output, base=repository),
        "provenance": {
            "source_commit": _source_commit(repository),
            "simulator_manifest_sha256": sha256_file(dataset.manifest_path),
            "resolved_seeds": {
                "simulation": _nested_seed(simulation_config),
                "training": _nested_seed(embedding_config),
                "evaluation": embedding_config.get("evaluation", {}).get("robustness", {}).get("seed"),
            },
            "cohort_size": int(metadata["rows"]["users"]),
            "cutoffs": {"train_end": metadata["train_end"], "validation_end": metadata["validation_end"]},
            "observed_source_hashes": source_hashes,
            "preparation_contract_version": metadata["dataset_contract"],
            "categorical_field_order": metadata["categorical_fields"],
            "continuous_field_order": metadata["continuous_fields"],
            "preparation_metadata_sha256": metadata_hash,
        },
        "evidence_identity": {
            "baseline": {**_public_identity(baseline), "observed_source_hashes": source_hashes, "preparation_metadata_sha256": metadata_hash},
            "learned": {**_public_identity(learned), "observed_source_hashes": source_hashes, "preparation_metadata_sha256": metadata_hash},
        },
        "required_artifacts": {
            group: [_artifact(identifier, path, repository) for identifier, path in entries]
            for group, entries in groups.items()
        },
        "comparability_audit": {
            "baseline_and_learned_source_hashes_match": True,
            "cutoffs_match": True,
            "categorical_field_order_matches": True,
            "users_match": True,
            "preparation_contract_matches": True,
            "dense_users_and_timestamps_match": True,
            "robustness_specifications_and_masks_match": True,
            "all_indexed_npz_values_finite": True,
            "result": "passed",
            "blocking_reasons": [],
        },
        "dataset_manifest": {"declared_seed": manifest.get("seed")},
    }
    write_json(index, output)
    return index


def _embedding_identity(path: Path, label: str, *, dense: bool = False) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        label_field = "timestamp" if dense else "cutoff"
        required = {"user_id", label_field, "embedding"}
        if not required.issubset(payload.files):
            raise ValueError(f"{label} export {path} lacks arrays {sorted(required - set(payload.files))}")
        users = payload["user_id"].astype(str)
        labels = payload[label_field].astype(str)
        vectors = payload["embedding"]
        if not np.isfinite(vectors).all():
            raise ValueError(f"{label} export {path} contains non-finite embedding values")
        keys = sorted(zip(users.tolist(), labels.tolist()))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{label} export {path} contains duplicate user/{label_field} rows")
    return {"users": sorted(set(users)), "cutoffs": sorted(set(labels)), "keys": keys, "user_set_sha256": stable_values_hash(users)}


def _compare_identities(baseline: dict[str, Any], learned: dict[str, Any], *, label: str = "") -> list[str]:
    errors = []
    for field in ("users", "cutoffs"):
        if baseline[field] != learned[field]:
            left, right = set(baseline[field]), set(learned[field])
            errors.append(f"baseline/learned {label}{field} mismatch: baseline_only={sorted(left-right)}, learned_only={sorted(right-left)}")
    if baseline["keys"] != learned["keys"]:
        errors.append(f"baseline/learned {label}user/cutoff rows mismatch")
    return errors


def _validate_observed_sources(dataset: DatasetLayout, metadata: dict[str, Any]) -> None:
    declared = _canonical_source_hashes(metadata.get("source_files", {}))
    for filename in OBSERVED_FILES.values():
        actual = sha256_file(dataset.observed / filename)
        if declared.get(filename) != actual:
            raise ValueError(f"Observed source hash mismatch for {filename}: prepared={declared.get(filename)!r}, actual={actual!r}")


def _validate_report_identity(experiment: ExperimentLayout, metadata: dict[str, Any], errors: list[str]) -> None:
    expected_hash = sha256_file(experiment.prepared_metadata)
    expected_sources = _canonical_source_hashes(metadata["source_files"])
    for kind, path in (("baseline", experiment.baseline_episode_response), ("learned", experiment.episode_response)):
        if not path.is_file():
            continue
        contract = read_json(path).get("metric_contract", {})
        if contract.get("prepared_metadata_sha256") != expected_hash:
            errors.append(f"{kind} episode preparation identity mismatch: report={contract.get('prepared_metadata_sha256')!r}, expected={expected_hash!r}")
        if _canonical_source_hashes(contract.get("source_hashes", {})) != expected_sources:
            errors.append(f"{kind} episode observed-source hashes mismatch")
    for kind in ("baseline", "learned"):
        path = experiment.robustness_report(kind)
        if path.is_file():
            report = read_json(path)
            contract = report.get("metric_contract", report)
            if _canonical_source_hashes(contract.get("source_hashes", {})) != expected_sources:
                errors.append(f"{kind} robustness observed-source hashes mismatch")
            fields = contract.get("field_order", {}).get("categorical")
            if fields != metadata["categorical_fields"]:
                errors.append(f"{kind} robustness categorical field order mismatch: report={fields!r}, prepared={metadata['categorical_fields']!r}")
    reports = [read_json(experiment.robustness_report(kind)) for kind in ("baseline", "learned") if experiment.robustness_report(kind).is_file()]
    if len(reports) == 2:
        left = reports[0].get("metric_contract", reports[0])
        right = reports[1].get("metric_contract", reports[1])
        for field in ("source_hashes", "specification_hash", "view_ids", "mask_hashes", "field_order"):
            if left.get(field) != right.get(field):
                errors.append(f"baseline/learned robustness {field} mismatch")


def _canonical_source_hashes(values: dict[str, str]) -> dict[str, str]:
    aliases = {"users": OBSERVED_FILES["users"], "events": OBSERVED_FILES["events"]}
    return {aliases.get(key, key): value for key, value in values.items()}


def _public_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "keys"}


def _artifact(identifier: str, path: Path, repository: Path) -> dict[str, Any]:
    return {"id": identifier, "identifier": normalize_identifier(path, base=repository),
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "status": "present"}


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _nested_seed(config: dict[str, Any]) -> Any:
    return config.get("seed", config.get("run", {}).get("seed"))


def _source_commit(repository: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
