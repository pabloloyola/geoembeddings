"""Shared, versioned file contract between simulation and embedding code."""

from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass
from typing import Any


DATASET_CONTRACT_NAME = "geoembeddings-dataset"
DATASET_CONTRACT_VERSION = "1.0"
SIMULATION_IDENTITY_MANIFEST_SCHEMA = "geoembeddings-simulation-identity/1.0"
SIMULATION_IDENTITY_HASH_ALGORITHM = "sha256-canonical-sorted-identifiers/1.0"

IDENTITY_ENTITY_NAMES = ("users", "regions", "pois", "episodes", "choices", "trajectories")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_MANIFEST_SCHEMA = "geoembeddings-pair-manifest/1.0"
PAIR_INTEGRITY_SCHEMA = "geoembeddings-pair-integrity/1.0"
COUNTERFACTUAL_COMPARISON_SCHEMA = "geoembeddings-counterfactual-comparison/1.0"
PAIR_INTERVENTIONS = ("identity", "observation", "exposure", "opportunity", "temporary-trip", "sustained-preference", "schedule-shift")


@dataclass(frozen=True)
class PairRunIdentity:
    run_dir: str
    simulator_version: str
    dataset_contract: dict[str, str]
    manifest_sha256: str
    config_sha256: str
    source_hashes: dict[str, str]
    identity_schema: str
    entity_hashes: dict[str, str]


@dataclass(frozen=True)
class PairManifest:
    """Typed protected declaration for matching two simulator runs."""

    schema_version: str
    reference: PairRunIdentity
    intervention: PairRunIdentity
    intervention_type: str
    intervention_parameters: dict[str, Any]
    invariant_entity_classes: tuple[str, ...]
    allowed_to_change_fields: tuple[str, ...]
    matching_keys: dict[str, tuple[str, ...]]
    stream_lineage: dict[str, Any]
    creation_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # Normalize tuples to their JSON array representation at the contract boundary.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value: Any) -> "PairManifest":
        validate_pair_manifest(value)
        return cls(
            schema_version=value["schema_version"],
            reference=PairRunIdentity(**value["reference"]),
            intervention=PairRunIdentity(**value["intervention"]),
            intervention_type=value["intervention_type"],
            intervention_parameters=value["intervention_parameters"],
            invariant_entity_classes=tuple(value["invariant_entity_classes"]),
            allowed_to_change_fields=tuple(value["allowed_to_change_fields"]),
            matching_keys={key: tuple(fields) for key, fields in value["matching_keys"].items()},
            stream_lineage=value["stream_lineage"],
            creation_provenance=value["creation_provenance"],
        )


def validate_pair_manifest(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != PAIR_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported pair-manifest schema: {getattr(value, 'get', lambda *_: None)('schema_version')!r}")
    for side in ("reference", "intervention"):
        run = value.get(side)
        if not isinstance(run, dict):
            raise ValueError(f"pair manifest is missing {side} run identity")
        required = ("run_dir", "simulator_version", "dataset_contract", "manifest_sha256", "config_sha256", "source_hashes", "identity_schema", "entity_hashes")
        if any(not run.get(field) for field in required):
            raise ValueError(f"pair manifest {side} identity has missing hashes or provenance")
        hashes = [run["manifest_sha256"], run["config_sha256"], *run["source_hashes"].values(), *run["entity_hashes"].values()]
        if not hashes or any(not _SHA256_RE.fullmatch(str(item)) for item in hashes):
            raise ValueError(f"pair manifest {side} contains a missing or malformed hash")
    if value.get("reference", {}).get("dataset_contract") != value.get("intervention", {}).get("dataset_contract"):
        raise ValueError("paired runs have incompatible dataset contracts")
    if value.get("intervention_type") not in PAIR_INTERVENTIONS:
        raise ValueError(f"unsupported intervention type: {value.get('intervention_type')!r}")
    invariant = value.get("invariant_entity_classes")
    changed = value.get("allowed_to_change_fields")
    if not isinstance(invariant, list) or not invariant or len(invariant) != len(set(invariant)):
        raise ValueError("invariant entity classes must be unique and non-empty")
    if not isinstance(changed, list) or len(changed) != len(set(changed)):
        raise ValueError("allowed-to-change fields must be unique")
    if set(invariant) & set(changed):
        raise ValueError("invariant and allowed-to-change declarations overlap")
    keys = value.get("matching_keys")
    if not isinstance(keys, dict) or set(keys) != set(IDENTITY_ENTITY_NAMES):
        raise ValueError("matching keys must unambiguously cover every identity entity class")
    if any(not isinstance(fields, list) or not fields or len(fields) != len(set(fields)) for fields in keys.values()):
        raise ValueError("ambiguous matching keys are empty or duplicated")
    if len({tuple(fields) for fields in keys.values()}) != len(keys):
        raise ValueError("ambiguous matching keys are reused across entity classes")
    if not isinstance(value.get("stream_lineage"), dict) or not value["stream_lineage"]:
        raise ValueError("stream lineage is required")
    if not isinstance(value.get("creation_provenance"), dict) or not value["creation_provenance"].get("created_at"):
        raise ValueError("creation provenance is required")


def validate_identity_manifest(section: Any, *, stream_names: tuple[str, ...]) -> None:
    """Reject incomplete or unsupported run-level identity/stream provenance."""
    if not isinstance(section, dict):
        raise ValueError("manifest identity section is missing")
    if section.get("schema_version") != SIMULATION_IDENTITY_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported simulation identity schema: {section.get('schema_version')!r}")
    if section.get("hash_algorithm") != SIMULATION_IDENTITY_HASH_ALGORITHM:
        raise ValueError(f"Unsupported identity hash algorithm: {section.get('hash_algorithm')!r}")
    if not isinstance(section.get("identity_generation_version"), str) or not section["identity_generation_version"]:
        raise ValueError("identity_generation_version is required")
    streams = section.get("random_streams")
    if not isinstance(streams, dict) or not isinstance(streams.get("algorithm"), str):
        raise ValueError("complete random-stream provenance is required")
    if not isinstance(streams.get("root_seed"), int) or set(streams.get("seeds", {})) != set(stream_names):
        raise ValueError("resolved seeds for every random stream are required")
    if any(not isinstance(seed, int) for seed in streams["seeds"].values()):
        raise ValueError("random-stream seeds must be integers")
    entities = section.get("entities")
    if not isinstance(entities, dict) or set(entities) != set(IDENTITY_ENTITY_NAMES):
        raise ValueError("identity entity declarations are incomplete")
    for name, declaration in entities.items():
        if not isinstance(declaration, dict) or not isinstance(declaration.get("count"), int) or declaration["count"] < 0:
            raise ValueError(f"invalid identity count for {name}")
        if not _SHA256_RE.fullmatch(str(declaration.get("identity_sha256", ""))):
            raise ValueError(f"malformed identity hash for {name}")

RELIABILITY_REPORT_SCHEMA = "geoembeddings-reliability-report/1.0"
OFFLINE_BENCHMARK_SCHEMA = "geoembeddings-offline-benchmark/1.0"

OBSERVED_FILES = {
    "users": "users_observed.csv.gz",
    "events": "observed_events.csv.gz",
}

TRUTH_FILES = {
    "user_latents": "user_latents.csv.gz",
    "episodes": "episodes_truth.csv.gz",
    "candidate_sets": "candidate_sets.csv.gz",
    "choices": "choices_truth.csv.gz",
    "trajectories": "trajectories_truth.csv.gz",
    "observation_process": "observation_process.csv.gz",
}
