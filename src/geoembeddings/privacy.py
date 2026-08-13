"""Fail-closed input authentication for protected privacy evaluations.

This module deliberately does not know how to read protected labels.  A privacy
evaluator must call :func:`authenticate_privacy_inputs` first and may pass the
returned immutable identities to the label-loading phase.  Keeping this gate in
a truth-independent module makes it possible to test that malformed evidence
cannot create a report (or even cause a protected file to be opened).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .artifact_index import SCHEMA_VERSION as EVIDENCE_INDEX_SCHEMA_VERSION
from .io import read_json, sha256_file
from .representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION, load_embedding_export
from .runtime_metadata import RuntimeMetadata


FACTORIZATION_INDEX_SCHEMA_VERSION = "geoembeddings-factorization-evidence-index/1.0"
PRIVACY_INPUT_SCHEMA_VERSION = "geoembeddings-privacy-input/1.0"
SELECTION_ROLE = "diagnostic_control"
BASELINE_CHECKPOINT_IDENTITY = "not_applicable"


def _canonical_hash(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrivacyInput:
    """Declared, non-protected files and identities for one diagnostic control."""

    name: str
    kind: Literal["statistical_baseline", "learned"]
    export_path: Path
    prepared_metadata_path: Path
    utility_report_path: Path
    selection_role: str
    parameter_count: int
    eligible_users: tuple[str, ...]
    utility_report_users: tuple[str, ...]
    checkpoint_path: Path | None = None
    checkpoint_identity: str | None = None
    model_variant: str | None = None


@dataclass(frozen=True)
class PrivacyInputIdentity:
    """Authenticated identity safe to copy into a privacy report."""

    schema_version: str
    name: str
    kind: str
    selection_role: str
    export_path: str
    export_bytes: int
    export_sha256: str
    export_schema: str
    model_variant: str
    component_order: tuple[str, ...]
    component_dimensions: tuple[int, ...]
    export_keys_sha256: str
    cutoffs: tuple[str, ...]
    checkpoint_identity: str
    preparation_metadata_sha256: str
    preparation_definition_sha256: str
    dataset_contract: str
    categorical_fields: tuple[str, ...]
    continuous_fields: tuple[str, ...]
    observed_source_hashes: tuple[tuple[str, str], ...]
    parameter_count: int
    eligible_users_sha256: str
    utility_report_users_sha256: str


@dataclass(frozen=True)
class AuthenticatedPrivacyInputs:
    """Evidence decision and the fully authenticated control identities."""

    evidence_index_sha256: str
    evidence_index_schema: str
    evidence_task_id: str
    t2_7_decision: str
    inputs: tuple[PrivacyInputIdentity, ...]
    runtime_metadata: RuntimeMetadata | None = None


def _indexed_artifact(index: dict[str, Any], path: Path) -> dict[str, Any]:
    artifacts = index.get("artifacts")
    if isinstance(artifacts, dict):
        candidates = (str(path), str(path.resolve()))
        for candidate in candidates:
            if candidate in artifacts:
                return artifacts[candidate]
    # The general evidence index stores artifacts in named lists.
    required = index.get("required_artifacts", {})
    for entries in required.values() if isinstance(required, dict) else ():
        for artifact in entries:
            identifier = artifact.get("identifier")
            if identifier in {str(path), str(path.resolve())}:
                return artifact
    raise ValueError(f"Export is absent from evidence index: {path}")


def _require_indexed_bytes(index: dict[str, Any], path: Path) -> None:
    artifact = _indexed_artifact(index, path)
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if int(artifact.get("bytes", actual_bytes)) != actual_bytes:
        raise ValueError(f"Indexed byte count mismatch for {path}")
    if artifact.get("sha256") != actual_hash:
        raise ValueError(f"Indexed SHA-256 mismatch for {path}")


def _source_hashes(metadata: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sources = metadata.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Preparation metadata has no observed-source hashes")
    result = tuple(sorted((str(name), str(digest)) for name, digest in sources.items()))
    if any(len(digest) != 64 for _, digest in result):
        raise ValueError("Preparation metadata contains a malformed observed-source hash")
    return result


def _as_scalar(arrays: dict[str, np.ndarray], name: str) -> str:
    if name not in arrays or np.asarray(arrays[name]).shape != ():
        raise ValueError(f"Export metadata {name!r} is missing or non-scalar")
    return str(np.asarray(arrays[name]).item())


def _authenticate_one(spec: PrivacyInput, index: dict[str, Any], matched: dict[str, Any]) -> PrivacyInputIdentity:
    if spec.selection_role != SELECTION_ROLE:
        raise ValueError(f"{spec.name} selection_role must be immutable {SELECTION_ROLE!r}")
    if isinstance(spec.parameter_count, bool) or not isinstance(spec.parameter_count, int) or spec.parameter_count < 0:
        raise ValueError(f"{spec.name} parameter_count must be a non-negative integer")
    for path in (spec.export_path, spec.prepared_metadata_path, spec.utility_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_indexed_bytes(index, spec.export_path)
    _require_indexed_bytes(index, spec.prepared_metadata_path)
    _require_indexed_bytes(index, spec.utility_report_path)

    metadata = read_json(spec.prepared_metadata_path)
    metadata_hash = sha256_file(spec.prepared_metadata_path)
    definition = {key: metadata.get(key) for key in ("train_end", "validation_end", "categorical_fields", "continuous_fields")}
    definition_hash = _canonical_hash(definition)
    categorical = tuple(metadata.get("categorical_fields", ()))
    continuous = tuple(metadata.get("continuous_fields", ()))
    if not categorical or not continuous or len(set(categorical)) != len(categorical) or len(set(continuous)) != len(continuous):
        raise ValueError(f"{spec.name} preparation field order is missing or ambiguous")
    sources = _source_hashes(metadata)
    if metadata.get("dataset_contract") is None:
        raise ValueError(f"{spec.name} preparation metadata lacks dataset contract")

    export = load_embedding_export(spec.export_path)
    arrays = export.arrays
    if export.schema_version != EXPORT_SCHEMA_VERSION:
        raise ValueError("Privacy audits require the versioned component export schema")
    order = tuple(str(value) for value in arrays["component_names"])
    dimensions = tuple(int(value) for value in arrays["component_dimensions"])
    if order != COMPONENT_NAMES:
        raise ValueError("Export component order does not match the canonical schema")
    if tuple(str(value) for value in arrays["categorical_fields"]) != categorical or tuple(str(value) for value in arrays["continuous_fields"]) != continuous:
        raise ValueError("Export/preparation ordered field mismatch")
    if _as_scalar(arrays, "preparation_hash") != metadata_hash:
        raise ValueError("Export preparation-metadata hash mismatch")
    export_sources = tuple(sorted(zip(arrays["source_file_names"].astype(str), arrays["source_hashes"].astype(str))))
    if export_sources != sources:
        raise ValueError("Export/preparation observed-source hash mismatch")
    if _as_scalar(arrays, "train_end") != str(metadata["train_end"]) or _as_scalar(arrays, "validation_end") != str(metadata["validation_end"]):
        raise ValueError("Export/preparation cutoff-definition mismatch")
    users = arrays["user_id"].astype(str)
    cutoffs = arrays["cutoff"].astype(str)
    keys = sorted(zip(users.tolist(), cutoffs.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("Export contains duplicate user/cutoff keys")
    eligible = tuple(sorted(spec.eligible_users))
    utility_users = tuple(sorted(spec.utility_report_users))
    if len(eligible) != len(set(eligible)) or len(utility_users) != len(set(utility_users)):
        raise ValueError("Population identities contain duplicate users")
    if sorted(set(users)) != list(eligible):
        raise ValueError("Exact export user eligibility identity mismatch")

    model_variant = _as_scalar(arrays, "model_variant")
    if spec.kind == "learned":
        if spec.checkpoint_path is None or not spec.checkpoint_path.is_file():
            raise FileNotFoundError(spec.checkpoint_path or "missing learned checkpoint")
        _require_indexed_bytes(index, spec.checkpoint_path)
        checkpoint_hash = sha256_file(spec.checkpoint_path)
        if spec.checkpoint_identity != checkpoint_hash:
            raise ValueError("Learned checkpoint SHA-256 identity mismatch")
        checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_variant = checkpoint.get("model_variant") or checkpoint.get("component_schema", {}).get("model_variant")
        if checkpoint_variant != model_variant or spec.model_variant != model_variant:
            raise ValueError("Learned model variant mismatch")
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Learned checkpoint lacks model_state for parameter authentication")
        actual_parameter_count = sum(int(value.numel()) for value in model_state.values())
        if spec.parameter_count != actual_parameter_count:
            raise ValueError("Learned parameter count mismatch")
        checkpoint_identity = checkpoint_hash
    else:
        if spec.checkpoint_path is not None or spec.checkpoint_identity != BASELINE_CHECKPOINT_IDENTITY:
            raise ValueError("Statistical baseline checkpoint identity must be explicit not_applicable")
        if model_variant != "statistical_baseline" or spec.model_variant != model_variant:
            raise ValueError("Statistical baseline model variant mismatch")
        if spec.parameter_count != 0:
            raise ValueError("Statistical baseline parameter count must be zero")
        checkpoint_identity = BASELINE_CHECKPOINT_IDENTITY

    matched_sources = tuple(sorted((str(k), str(v)) for k, v in matched.get("source_files", {}).items()))
    if matched_sources and matched_sources != sources:
        raise ValueError("Evidence-index observed-source identity mismatch")
    if matched.get("preparation_definition") not in (None, definition):
        raise ValueError("Evidence-index preparation definition mismatch")
    if matched.get("export_keys_sha256") not in (None, _canonical_hash(keys)):
        raise ValueError("Evidence-index export key identity mismatch")
    if matched.get("user_mask_sha256") not in (None, _canonical_hash(list(eligible))):
        raise ValueError("Evidence-index eligible-user identity mismatch")
    if matched.get("cutoffs") not in (None, sorted(set(cutoffs))):
        raise ValueError("Evidence-index cutoff identity mismatch")

    utility = read_json(spec.utility_report_path)
    declared_utility_users = utility.get("population_identity", {}).get("users")
    declared_utility_hash = utility.get("population_identity", {}).get("user_set_sha256")
    expected_utility_hash = _canonical_hash(list(utility_users))
    if declared_utility_users is not None and tuple(sorted(map(str, declared_utility_users))) != utility_users:
        raise ValueError("Utility-report population identity mismatch")
    if declared_utility_hash != expected_utility_hash:
        raise ValueError("Utility-report population hash mismatch")

    return PrivacyInputIdentity(
        PRIVACY_INPUT_SCHEMA_VERSION, spec.name, spec.kind, spec.selection_role,
        str(spec.export_path), spec.export_path.stat().st_size, sha256_file(spec.export_path),
        export.schema_version, model_variant, order, dimensions, _canonical_hash(keys),
        tuple(sorted(set(cutoffs))), checkpoint_identity, metadata_hash, definition_hash,
        str(metadata["dataset_contract"]), categorical, continuous, sources,
        spec.parameter_count, _canonical_hash(list(eligible)), expected_utility_hash,
    )


def authenticate_privacy_inputs(
    evidence_index_path: str | Path,
    inputs: tuple[PrivacyInput, ...] | list[PrivacyInput],
    *,
    runtime_metadata: RuntimeMetadata | None = None,
) -> AuthenticatedPrivacyInputs:
    """Authenticate every public input before a caller opens protected labels.

    No output path or protected-label path is accepted by this function.  Thus a
    failure cannot partially create either privacy output or accidentally open
    truth merely while attempting authentication.
    """
    path = Path(evidence_index_path)
    index = read_json(path)
    schema = index.get("schema_version")
    if schema not in {FACTORIZATION_INDEX_SCHEMA_VERSION, EVIDENCE_INDEX_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported evidence-index schema: {schema!r}")
    if index.get("task_id") not in {"T2.7", "T2.4-T2.7"}:
        raise ValueError("Privacy inputs require the T2.7 evidence identity")
    if index.get("decision") != "do not advance":
        raise ValueError("T2.7 decision mismatch; expected 'do not advance'")
    matched = index.get("matched_identity")
    if not isinstance(matched, dict):
        raise ValueError("Evidence index lacks the matched T2.7 identity")
    if not inputs:
        raise ValueError("At least one privacy input is required")
    names = [item.name for item in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Privacy input names must be unique")
    identities = tuple(_authenticate_one(item, index, matched) for item in inputs)
    first = identities[0]
    for identity in identities[1:]:
        for field in ("preparation_metadata_sha256", "preparation_definition_sha256", "dataset_contract", "categorical_fields", "continuous_fields", "observed_source_hashes", "export_keys_sha256", "cutoffs", "eligible_users_sha256", "utility_report_users_sha256"):
            if getattr(identity, field) != getattr(first, field):
                raise ValueError(f"Privacy controls have mismatched {field}")
    return AuthenticatedPrivacyInputs(
        sha256_file(path), str(schema), str(index["task_id"]), str(index["decision"]),
        identities, runtime_metadata,
    )
