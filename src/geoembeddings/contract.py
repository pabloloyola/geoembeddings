"""Shared, versioned file contract between simulation and embedding code."""

from __future__ import annotations

import re
from typing import Any


DATASET_CONTRACT_NAME = "geoembeddings-dataset"
DATASET_CONTRACT_VERSION = "1.0"
SIMULATION_IDENTITY_MANIFEST_SCHEMA = "geoembeddings-simulation-identity/1.0"
SIMULATION_IDENTITY_HASH_ALGORITHM = "sha256-canonical-sorted-identifiers/1.0"

IDENTITY_ENTITY_NAMES = ("users", "regions", "pois", "episodes", "choices", "trajectories")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
