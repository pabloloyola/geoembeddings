"""Field-level integrity validation for protected matched simulator runs."""

from __future__ import annotations

import csv
import fnmatch
import gzip
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .contract import (PAIR_INTEGRITY_SCHEMA, PairManifest,
                       validate_pair_manifest)
from .io import sha256_file, write_json
from .layout import DatasetLayout, PairLayout


@dataclass(frozen=True)
class TableSpec:
    name: str
    relative_path: str
    keys: tuple[str, ...]


TABLES = (
    TableSpec("observed.users", "observed/users_observed.csv.gz", ("user_id",)),
    # Events deliberately use their complete public row as a multiset key. There is
    # no simulator event ID, and silently pairing equal-position rows would make
    # validation depend on CSV order.
    TableSpec("observed.events", "observed/observed_events.csv.gz", ()),
    TableSpec("observed.poi_catalog", "observed/poi_catalog.csv.gz", ("poi_id",)),
    TableSpec("observed.recommendation_requests", "observed/recommendation_requests.csv.gz", ("request_id",)),
    TableSpec("observed.impressions", "observed/impressions.csv.gz", ("request_id", "poi_id")),
    TableSpec("observed.interactions", "observed/interactions.csv.gz", ("interaction_id",)),
    TableSpec("truth.user_latents", "truth/user_latents.csv.gz", ("user_id",)),
    TableSpec("truth.episodes", "truth/episodes_truth.csv.gz", ("episode_id",)),
    TableSpec("truth.candidate_sets", "truth/candidate_sets.csv.gz", ("decision_id", "candidate_poi_id")),
    TableSpec("truth.choices", "truth/choices_truth.csv.gz", ("decision_id",)),
    TableSpec("truth.trajectories", "truth/trajectories_truth.csv.gz", ("trajectory_id",)),
    TableSpec("truth.observation_process", "truth/observation_process.csv.gz", ("user_id", "source_service")),
)
CHANGE_TABLE = TableSpec("truth.change_points", "truth/change_points_truth.csv.gz", ("user_id",))
TEMPORARY_SCHEDULE_TABLES = (
    TableSpec("truth.temporary_schedule_shift", "truth/temporary_schedule_shift_truth.csv.gz", ("user_id",)),
    TableSpec("truth.temporary_schedule_shift_events", "truth/temporary_schedule_shift_events.csv.gz", ()),
)
SAMPLE_LIMIT = 10


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV fields: {path}")
        return list(reader.fieldnames), list(reader)


def _key(row: dict[str, str], fields: tuple[str, ...], schema: list[str]) -> tuple[str, ...]:
    selected = fields or tuple(schema)
    return tuple(row[field] for field in selected)


def _allowed(path: str, patterns: Iterable[str]) -> str | None:
    return next((pattern for pattern in patterns if fnmatch.fnmatchcase(path, pattern)), None)


def _string_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, str]]]:
    schema = list(rows[0]) if rows else []
    return schema, [{field: json.dumps(row[field], sort_keys=True, separators=(",", ":"))
                     for field in schema} for row in rows]


def _rebuild_world(config: dict[str, Any], world_seed: int) -> list[dict[str, Any]]:
    # No standalone POI table exists, so reconstruct every object from the
    # authenticated config and named stream rather than checking only sampled POIs.
    from . import simulator
    simulator.activate_config(config)
    return simulator.make_world(random.Random(world_seed))


def compare_rows(name: str, reference_schema: list[str], reference_rows: list[dict[str, str]],
                 intervention_schema: list[str], intervention_rows: list[dict[str, str]],
                 keys: tuple[str, ...], allowed_patterns: Iterable[str]) -> dict[str, Any]:
    """Compare one declared table at key and field granularity."""
    schema_match = reference_schema == intervention_schema
    missing_key_fields = [field for field in keys if field not in reference_schema or field not in intervention_schema]
    if missing_key_fields:
        raise ValueError(f"{name} is missing declared key fields: {missing_key_fields}")
    ref_counts = Counter(_key(row, keys, reference_schema) for row in reference_rows)
    int_counts = Counter(_key(row, keys, intervention_schema) for row in intervention_rows)
    duplicate_ref = [list(key) for key, count in ref_counts.items() if count > 1]
    duplicate_int = [list(key) for key, count in int_counts.items() if count > 1]
    # A complete-row multiset permits repeated identical events and compares counts.
    enforce_unique = bool(keys)
    ref_map = {_key(row, keys, reference_schema): row for row in reference_rows}
    int_map = {_key(row, keys, intervention_schema): row for row in intervention_rows}
    missing = sorted(ref_counts.keys() - int_counts.keys())
    extra = sorted(int_counts.keys() - ref_counts.keys())
    count_mismatch = sorted(key for key in ref_counts.keys() & int_counts.keys() if ref_counts[key] != int_counts[key])
    key_change_pattern = _allowed(f"{name}.*", allowed_patterns)
    mismatches: list[dict[str, Any]] = []
    allowed_counts: Counter[str] = Counter()
    disallowed_count = 0
    if schema_match and (not enforce_unique or (not duplicate_ref and not duplicate_int)):
        for key in sorted(ref_map.keys() & int_map.keys()):
            for field in reference_schema:
                before, after = ref_map[key][field], int_map[key][field]
                if before == after:
                    continue
                field_path = f"{name}.{field}"
                pattern = _allowed(field_path, allowed_patterns)
                if pattern:
                    allowed_counts[pattern] += 1
                else:
                    disallowed_count += 1
                if len(mismatches) < SAMPLE_LIMIT:
                    mismatches.append({"key": list(key), "field": field_path,
                                       "reference": before, "intervention": after,
                                       "allowed_by": pattern})
    key_changes_allowed = key_change_pattern is not None
    if key_changes_allowed:
        allowed_counts[key_change_pattern] += len(missing) + len(extra) + len(count_mismatch)
    passed = (schema_match and not missing_key_fields and (not enforce_unique or not duplicate_ref and not duplicate_int)
              and (key_changes_allowed or not missing and not extra and not count_mismatch) and disallowed_count == 0)
    return {
        "passed": passed, "reference_rows": len(reference_rows), "intervention_rows": len(intervention_rows),
        "reference_schema": reference_schema, "intervention_schema": intervention_schema,
        "schema_match": schema_match, "key_fields": list(keys or tuple(reference_schema)),
        "reference_unique_keys": len(ref_counts), "intervention_unique_keys": len(int_counts),
        "missing_keys": {"count": len(missing), "samples": [list(key) for key in missing[:SAMPLE_LIMIT]]},
        "extra_keys": {"count": len(extra), "samples": [list(key) for key in extra[:SAMPLE_LIMIT]]},
        "duplicate_keys": {"reference_count": len(duplicate_ref), "reference_samples": duplicate_ref[:SAMPLE_LIMIT],
                           "intervention_count": len(duplicate_int), "intervention_samples": duplicate_int[:SAMPLE_LIMIT]},
        "row_count_mismatches": {"count": len(count_mismatch), "samples": [list(key) for key in count_mismatch[:SAMPLE_LIMIT]]},
        "field_mismatches": {"disallowed_count": disallowed_count, "samples": mismatches},
        "allowed_changes": {pattern: count for pattern, count in sorted(allowed_counts.items())},
    }


def _verify_run(side: str, declared: Any) -> tuple[DatasetLayout, dict[str, Any], dict[str, str]]:
    layout = DatasetLayout.from_path(declared.run_dir)
    manifest = layout.validate(require_truth=True)
    actual = {"manifest.json": sha256_file(layout.manifest_path),
              "config.resolved.yaml": sha256_file(layout.resolved_config)}
    if actual["manifest.json"] != declared.manifest_sha256:
        raise ValueError(f"{side} manifest hash is stale")
    if actual["config.resolved.yaml"] != declared.config_sha256:
        raise ValueError(f"{side} resolved-config hash is stale")
    for relative, expected in declared.source_hashes.items():
        path = layout.root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{side} source hash is stale: {relative}")
        actual[relative] = expected
    identity = manifest["identity"]
    for entity, expected in declared.entity_hashes.items():
        if identity["entities"][entity]["identity_sha256"] != expected:
            raise ValueError(f"{side} entity hash is stale: {entity}")
    return layout, manifest, actual


def validate_pair(pair_manifest_path: str | Path) -> dict[str, Any]:
    pair_layout = PairLayout.from_manifest_path(pair_manifest_path)
    if not pair_layout.manifest.is_file():
        raise FileNotFoundError(f"Missing pair manifest: {pair_layout.manifest}")
    raw = json.loads(pair_layout.manifest.read_text(encoding="utf-8"))
    validate_pair_manifest(raw)
    pair = PairManifest.from_dict(raw)
    reference, ref_manifest, ref_hashes = _verify_run("reference", pair.reference)
    intervention, int_manifest, int_hashes = _verify_run("intervention", pair.intervention)
    if pair.stream_lineage != {"reference": ref_manifest["identity"]["random_streams"],
                               "intervention": int_manifest["identity"]["random_streams"]}:
        raise ValueError("pair stream lineage is stale")
    allowed = pair.allowed_to_change_fields
    table_specs = TABLES
    if pair.intervention_type in {"temporary-trip", "sustained-preference"}:
        table_specs += (CHANGE_TABLE,)
    if pair.intervention_type == "temporary_schedule_shift_v1":
        table_specs += TEMPORARY_SCHEDULE_TABLES
    results: dict[str, Any] = {}
    for spec in table_specs:
        ref_schema, ref_rows = _read_table(reference.root / spec.relative_path)
        int_schema, int_rows = _read_table(intervention.root / spec.relative_path)
        results[spec.name] = compare_rows(spec.name, ref_schema, ref_rows, int_schema, int_rows,
                                          spec.keys, allowed)

    ref_config = yaml.safe_load(reference.resolved_config.read_text(encoding="utf-8"))
    int_config = yaml.safe_load(intervention.resolved_config.read_text(encoding="utf-8"))
    ref_region_schema, ref_regions = _string_rows(ref_config["world"]["regions"])
    int_region_schema, int_regions = _string_rows(int_config["world"]["regions"])
    results["truth.world_regions"] = compare_rows(
        "truth.world_regions", ref_region_schema, ref_regions, int_region_schema,
        int_regions, ("id",), allowed)
    ref_poi_schema, ref_pois = _string_rows(_rebuild_world(
        ref_config, ref_manifest["identity"]["random_streams"]["seeds"]["world"]))
    int_poi_schema, int_pois = _string_rows(_rebuild_world(
        int_config, int_manifest["identity"]["random_streams"]["seeds"]["world"]))
    results["truth.world_pois"] = compare_rows(
        "truth.world_pois", ref_poi_schema, ref_pois, int_poi_schema, int_pois,
        ("poi_id",), allowed)
    invariant_results = {
        entity: {"passed": pair.reference.entity_hashes[entity] == pair.intervention.entity_hashes[entity],
                 "reference_sha256": pair.reference.entity_hashes[entity],
                 "intervention_sha256": pair.intervention.entity_hashes[entity],
                 "reference_count": ref_manifest["identity"]["entities"][entity]["count"],
                 "intervention_count": int_manifest["identity"]["entities"][entity]["count"]}
        for entity in pair.invariant_entity_classes
    }
    allowed_totals = {
        pattern: sum(item.get("allowed_changes", {}).get(pattern, 0) for item in results.values())
        for pattern in allowed
    }
    passed = all(item["passed"] for item in results.values()) and all(item["passed"] for item in invariant_results.values())
    report = {
        "schema_version": PAIR_INTEGRITY_SCHEMA, "status": "passed" if passed else "failed",
        "pair_manifest_sha256": sha256_file(pair_layout.manifest), "intervention_type": pair.intervention_type,
        "input_hashes": {"reference": ref_hashes, "intervention": int_hashes},
        "entity_invariants": invariant_results, "table_results": results,
        "allowed_change_results": {pattern: {"passed": True, "difference_count": count}
                                   for pattern, count in allowed_totals.items()},
        "coverage": {name: {"reference_rows": value["reference_rows"], "intervention_rows": value["intervention_rows"]}
                     for name, value in results.items()},
        "limitations": ["This validates controlled simulator artifacts only; it does not establish external causal validity.",
                        "World objects are reconstructed from authenticated configuration and named-stream lineage because no standalone POI truth table is emitted."],
    }
    write_json(report, pair_layout.integrity_report)
    if not passed:
        raise RuntimeError(f"Pair integrity failed; inspect {pair_layout.integrity_report}")
    return report


def require_passing_pair_integrity(pair_manifest_path: str | Path) -> dict[str, Any]:
    """Hard gate to call before any future paired representation evaluation."""
    layout = PairLayout.from_manifest_path(pair_manifest_path)
    if not layout.integrity_report.is_file():
        raise FileNotFoundError(f"Missing required pair integrity report: {layout.integrity_report}")
    report = json.loads(layout.integrity_report.read_text(encoding="utf-8"))
    if report.get("schema_version") != PAIR_INTEGRITY_SCHEMA or report.get("status") != "passed":
        raise ValueError("A passing supported pair integrity report is required before paired evaluation")
    entity_results = report.get("entity_invariants")
    table_results = report.get("table_results")
    allowed_results = report.get("allowed_change_results")
    prerequisite_results = ([*entity_results.values(), *table_results.values(), *allowed_results.values()]
                            if all(isinstance(value, dict) for value in
                                   (entity_results, table_results, allowed_results)) else [])
    if (not isinstance(entity_results, dict) or not entity_results
            or not isinstance(table_results, dict) or set(table_results) not in (
                {spec.name for spec in TABLES} | {"truth.world_regions", "truth.world_pois"},
                {spec.name for spec in TABLES + (CHANGE_TABLE,)} | {"truth.world_regions", "truth.world_pois"},
                {spec.name for spec in TABLES + TEMPORARY_SCHEDULE_TABLES} | {"truth.world_regions", "truth.world_pois"})
            or not isinstance(allowed_results, dict)
            or not prerequisite_results or not all(isinstance(item, dict) and item.get("passed") is True
                                                   for item in prerequisite_results)):
        raise ValueError("Pair integrity report has incomplete or failing prerequisite results")
    if report.get("pair_manifest_sha256") != sha256_file(layout.manifest):
        raise ValueError("Pair integrity report is stale for the current pair manifest")
    # Re-authenticate every declared run input, not merely the pair declaration.
    pair = PairManifest.from_dict(json.loads(layout.manifest.read_text(encoding="utf-8")))
    _verify_run("reference", pair.reference)
    _verify_run("intervention", pair.intervention)
    return report
