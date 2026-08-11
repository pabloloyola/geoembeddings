"""Build the protected, versioned declaration joining two simulator runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .contract import (IDENTITY_ENTITY_NAMES, PAIR_MANIFEST_SCHEMA, PairManifest,
                       PairRunIdentity, validate_pair_manifest)
from .layout import DatasetLayout, PairLayout


MATCHING_KEYS = {
    "users": ("user_id",),
    "regions": ("region_id",),
    "pois": ("region_id", "category", "object_slot"),
    "episodes": ("user_id", "calendar_date"),
    "choices": ("episode_id", "primary_poi_choice"),
    "trajectories": ("episode_id", "activity_occurrence", "scheduled_true_time"),
}
ALLOWED_FIELDS = {
    "identity": (),
    "observation": ("observed.*", "truth.observation_process.*"),
    "exposure": ("truth.candidate_sets.utility_exposure", "truth.candidate_sets.utility_total", "truth.candidate_sets.is_chosen", "truth.choices.chosen_poi_id", "truth.trajectories.true_region_id", "truth.trajectories.true_latitude", "truth.trajectories.true_longitude", "observed.events.*"),
    "opportunity": ("truth.candidate_sets.*", "truth.choices.*", "truth.trajectories.*", "observed.*"),
}
INVARIANTS = {
    "identity": IDENTITY_ENTITY_NAMES,
    "observation": IDENTITY_ENTITY_NAMES,
    "exposure": ("users", "regions", "pois", "episodes"),
    "opportunity": ("users", "regions", "pois", "episodes"),
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pair source artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(layout: DatasetLayout, manifest: dict[str, Any]) -> PairRunIdentity:
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Run has no identity manifest: {layout.root}")
    config_hash = manifest.get("config_sha256")
    if not isinstance(config_hash, str) or not config_hash:
        raise ValueError(f"Run is missing config_sha256: {layout.root}")
    sources = {}
    for directory in (layout.observed, layout.truth):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                sources[str(path.relative_to(layout.root))] = _sha256(path)
    return PairRunIdentity(
        run_dir=str(layout.root), simulator_version=str(manifest.get("simulator_version", "")),
        dataset_contract=manifest["dataset_contract"], manifest_sha256=_sha256(layout.manifest_path),
        config_sha256=config_hash, source_hashes=sources, identity_schema=identity["schema_version"],
        entity_hashes={name: identity["entities"][name]["identity_sha256"] for name in IDENTITY_ENTITY_NAMES},
    )


def _intervention_type(reference: dict[str, Any], intervention: dict[str, Any]) -> str:
    declaration = intervention.get("intervention")
    if isinstance(declaration, dict) and declaration.get("type") in {"exposure", "opportunity", "observation"}:
        return str(declaration["type"])
    changed_streams = [name for name, seed in reference["identity"]["random_streams"]["seeds"].items()
                       if intervention["identity"]["random_streams"]["seeds"].get(name) != seed]
    if not changed_streams and reference.get("scenario") == intervention.get("scenario"):
        return "identity"
    if changed_streams == ["observation"] and reference.get("scenario") == intervention.get("scenario"):
        return "observation"
    scenario = str(intervention.get("scenario", ""))
    if "exposure" in scenario:
        return "exposure"
    if "opportunity" in scenario:
        return "opportunity"
    raise ValueError("Cannot infer one unambiguous supported intervention from run lineage")


def create_pair_manifest(reference_run_dir: str | Path, intervention_run_dir: str | Path,
                         output: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    reference_layout = DatasetLayout.from_path(reference_run_dir)
    intervention_layout = DatasetLayout.from_path(intervention_run_dir)
    if reference_layout.root == intervention_layout.root:
        raise ValueError("reference and intervention runs must be distinct")
    reference_manifest = reference_layout.validate(require_truth=True)
    intervention_manifest = intervention_layout.validate(require_truth=True)
    if reference_manifest.get("dataset_contract") != intervention_manifest.get("dataset_contract"):
        raise ValueError("paired runs have incompatible dataset contracts")
    kind = _intervention_type(reference_manifest, intervention_manifest)
    reference_identity = _identity(reference_layout, reference_manifest)
    intervention_identity = _identity(intervention_layout, intervention_manifest)
    for entity in INVARIANTS[kind]:
        if reference_identity.entity_hashes[entity] != intervention_identity.entity_hashes[entity]:
            raise ValueError(f"identity-incompatible invariant entity class: {entity}")
    streams = {side: manifest["identity"]["random_streams"] for side, manifest in
               (("reference", reference_manifest), ("intervention", intervention_manifest))}
    changed_streams = sorted(name for name in streams["reference"]["seeds"]
                             if streams["reference"]["seeds"][name] != streams["intervention"]["seeds"][name])
    definition = intervention_manifest.get("intervention") or {}
    configured_invariants = tuple(definition.get("invariant_entities", INVARIANTS[kind]))
    configured_fields = tuple(definition.get("permitted_changes", ALLOWED_FIELDS[kind]))
    pair = PairManifest(
        schema_version=PAIR_MANIFEST_SCHEMA, reference=reference_identity,
        intervention=intervention_identity, intervention_type=kind,
        intervention_parameters={"reference_scenario": reference_manifest.get("scenario"),
                                 "intervention_scenario": intervention_manifest.get("scenario"),
                                 "changed_streams": changed_streams,
                                 "config_overrides": definition.get("config_overrides", {}),
                                 "affected_random_streams": definition.get("affected_random_streams", []),
                                 "expected_behavioral_diagnostics": definition.get("behavioral_diagnostics", [])},
        invariant_entity_classes=configured_invariants,
        allowed_to_change_fields=configured_fields, matching_keys=MATCHING_KEYS,
        stream_lineage=streams,
        creation_provenance={"created_at": datetime.now(timezone.utc).isoformat(),
                             "geoembeddings_version": __version__, "python": platform.python_version(),
                             "command": "geoembed pair-manifest", "pid": os.getpid()},
    ).to_dict()
    validate_pair_manifest(pair)
    destination = PairLayout.from_manifest_path(output).manifest
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing pair manifest: {destination}")
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("--overwrite target must be an existing regular pair_manifest.json")
        validate_pair_manifest(json.loads(destination.read_text(encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return pair
