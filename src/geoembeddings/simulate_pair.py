"""Create immutable, configuration-driven matched simulator interventions."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from . import simulator
from .layout import DatasetLayout, PairLayout
from .pair_integrity import validate_pair
from .pair_manifest import create_pair_manifest


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _diagnostics(layout: DatasetLayout) -> dict[str, float]:
    candidates = _rows(layout.truth / "candidate_sets.csv.gz")
    choices = _rows(layout.truth / "choices_truth.csv.gz")
    events = _rows(layout.observed / "observed_events.csv.gz")
    observation = _rows(layout.truth / "observation_process.csv.gz")
    chosen = [row for row in candidates if row["is_chosen"] == "1"]
    location = [row for row in observation if row["source_service"] == "location"]
    result = {
        "exposed_chosen_rate": sum(float(row["exposed"]) for row in chosen) / max(1, len(chosen)),
        "mean_candidate_count": sum(float(row["candidate_count"]) for row in choices) / max(1, len(choices)),
        "observed_event_count": float(len(events)),
        "mean_location_gps_sd_m": sum(float(row["gps_sd_m"]) for row in location) / max(1, len(location)),
        "mean_routine_choice_hour": sum(int(row["timestamp"][11:13]) + int(row["timestamp"][14:16]) / 60
                                         for row in choices if row["choice_context"] == "routine") /
                                    max(1, sum(row["choice_context"] == "routine" for row in choices)),
    }
    change_path = layout.truth / "change_points_truth.csv.gz"
    if change_path.is_file():
        point = _rows(change_path)[0]
        after_ids = {row["decision_id"] for row in choices if row["timestamp"] >= point["change_start_time"] and row["chosen_category"] == point["target_category"]}
        values = [float(row["utility_preference"]) for row in candidates if row["decision_id"] in after_ids]
        result["changed_category_preference_utility"] = sum(values) / max(1, len(values))
    return result


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"intervention override has invalid path: {dotted}")
        cursor = child
    if parts[-1] not in cursor:
        raise ValueError(f"intervention override has unknown field: {dotted}")
    cursor[parts[-1]] = value


def simulate_pair(config_path: str | Path, reference_run_dir: str | Path,
                  intervention_run_dir: str | Path, pair_dir: str | Path, *,
                  intervention: str, users: int | None = None, days: int | None = None,
                  seed: int | None = None) -> dict[str, Any]:
    """Generate, structurally validate, declare, and field-validate one pair."""
    config_path = Path(config_path).expanduser().resolve()
    reference = DatasetLayout.from_path(reference_run_dir)
    changed = DatasetLayout.from_path(intervention_run_dir)
    pair = PairLayout(Path(pair_dir).expanduser().resolve())
    targets = (reference.root, changed.root, pair.root)
    if len(set(targets)) != len(targets):
        raise ValueError("reference, intervention, and pair roots must be distinct")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing paired artifacts: {existing}")

    base = simulator.load_config(config_path)
    if intervention not in base["interventions"]:
        raise ValueError(f"unknown configured intervention: {intervention!r}")
    definition = copy.deepcopy(base["interventions"][intervention])
    reference_config = copy.deepcopy(base)
    intervention_config = copy.deepcopy(base)
    for dotted, value in definition["config_overrides"].items():
        _set_path(intervention_config, dotted, value)
    resolved_seed = int(seed if seed is not None else base["run"]["seed"])
    for config, output in ((reference_config, reference.root), (intervention_config, changed.root)):
        config["run"].update(users=int(users if users is not None else base["run"]["users"]),
                             days=int(days if days is not None else base["run"]["days"]),
                             seed=resolved_seed, output=str(output))
    overrides = dict(intervention_config["run"].get("random_streams", {}))
    for stream in definition.get("reseed_streams", []):
        overrides[stream] = simulator.derive_stream_seed(resolved_seed, stream) + int(definition["stream_seed_offset"])
    if overrides:
        intervention_config["run"]["random_streams"] = overrides
    intervention_config["run"]["intervention"] = {
        "type": intervention,
        "config_overrides": definition["config_overrides"],
        "affected_random_streams": definition["affected_random_streams"],
        "invariant_entities": definition["invariant_entities"],
        "permitted_changes": definition["permitted_changes"],
        "behavioral_diagnostics": definition["behavioral_diagnostics"],
        "declaration_version": definition.get("declaration_version"),
        "change": definition.get("change"),
        "schedule_shift": definition.get("schedule_shift"),
    }
    if intervention in {"temporary-trip", "sustained-preference"}:
        reference_config["run"]["intervention"] = copy.deepcopy(intervention_config["run"]["intervention"])
        reference_config["run"]["intervention"]["change"] = dict(definition["change"], preference_delta=0.0)

    pair.root.mkdir(parents=True)
    for label, config, layout in (("reference", reference_config, reference),
                                  ("intervention", intervention_config, changed)):
        snapshot = pair.root / f"{label}_input.yaml"
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        args = argparse.Namespace(config=str(snapshot), output=str(layout.root), overwrite=False,
            seed=resolved_seed, users=config["run"]["users"], days=config["run"]["days"],
            start_date=config["run"]["start_date"], scenario=config["run"]["scenario"],
            full_kanto=config["run"]["full_kanto"])
        simulator.activate_config(config)
        simulator.simulate(args)
        from .simulation_validation import validate
        report = validate(layout.root)
        from .io import write_json
        write_json(report, layout.root / "deep_validation_report.json")
        structural = [check for check in report["checks"] if check.get("layer") == "integrity"]
        if not structural or not all(check["passed"] for check in structural):
            raise RuntimeError(f"{label} structural validation failed")
    create_pair_manifest(reference.root, changed.root, pair.manifest)
    integrity = validate_pair(pair.manifest)
    reference_metrics, intervention_metrics = _diagnostics(reference), _diagnostics(changed)
    diagnostics = {}
    for metric, direction in definition["behavioral_diagnostics"].items():
        before, after = reference_metrics[metric], intervention_metrics[metric]
        passed = after > before if direction == "increase" else after < before
        diagnostics[metric] = {"reference": before, "intervention": after,
                               "delta": after - before, "expected_direction": direction,
                               "passed": passed}
    behavioral = {"schema_version": "geoembeddings-pair-behavioral-diagnostics/1.0",
                  "status": "passed" if all(item["passed"] for item in diagnostics.values()) else "failed",
                  "intervention": intervention, "seed": resolved_seed,
                  "experimental_assumption": "Synthetic intervention constants are experimental assumptions, not claims about Tokyo or Kanto.",
                  "diagnostics": diagnostics}
    behavioral_path = pair.root / "behavioral_diagnostics.json"
    behavioral_path.write_text(json.dumps(behavioral, indent=2) + "\n", encoding="utf-8")
    if behavioral["status"] != "passed":
        raise RuntimeError(f"paired behavioral diagnostics failed; inspect {behavioral_path}")
    return {"intervention": intervention, "reference_run_dir": str(reference.root),
            "intervention_run_dir": str(changed.root), "pair_manifest": str(pair.manifest),
            "pair_integrity": str(pair.integrity_report), "behavioral_diagnostics": str(behavioral_path),
            "status": integrity["status"]}
