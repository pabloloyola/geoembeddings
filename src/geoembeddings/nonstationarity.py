"""Protected, matched R11 adaptation and forgetting audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json
from .layout import PairLayout

AUDIT_SCHEMA = "geoembeddings-nonstationarity-audit/1.0"
PAIR_SCHEMA = "geoembeddings-change-evaluation/2.0"
IDENTITY_FIELDS = ("users", "cutoffs", "preparation_contract", "source_lineage",
                   "component_schema", "relative_day_definition", "censoring_rules")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite input at {label}")
    return result


def _ci(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "lower_95": None, "upper_95": None}
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return {"n": 1, "mean": mean, "lower_95": mean, "upper_95": mean}
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return {"n": len(array), "mean": mean, "lower_95": mean - half, "upper_95": mean + half}


def threshold_time(points: list[tuple[int, float]], threshold: float, *, direction: str,
                   origin_day: int = 0) -> dict[str, Any]:
    """Return first threshold crossing and explicit interval censoring."""
    threshold = _finite(threshold, "threshold")
    clean = sorted((int(day), _finite(value, f"day {day}")) for day, value in points if int(day) >= origin_day)
    if not clean:
        return {"time_days": None, "crossed": False, "left_censored": False,
                "right_censored": False, "missing_post_change_coverage": True}
    predicate = (lambda value: value >= threshold) if direction == "above" else (
        lambda value: value <= threshold) if direction == "below" else None
    if predicate is None:
        raise ValueError("direction must be 'above' or 'below'")
    for day, value in clean:
        if predicate(value):
            return {"time_days": day - origin_day, "crossed": True,
                    "left_censored": day > origin_day and (day, value) == clean[0],
                    "right_censored": False, "missing_post_change_coverage": False}
    return {"time_days": None, "crossed": False, "left_censored": False,
            "right_censored": True, "missing_post_change_coverage": False}


def _load(path: str | Path, expected: str) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema_version") != PAIR_SCHEMA:
        raise ValueError(f"{expected} report has unsupported schema")
    if report.get("intervention") != expected:
        raise ValueError(f"Expected {expected} report, got {report.get('intervention')!r}")
    if report.get("authentication", {}).get("status") != "passed":
        raise ValueError(f"{expected} report is not authenticated")
    return report, hashlib.sha256(raw).hexdigest()


def _series(component: dict[str, Any]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for point in component.get("curve", []):
        day = int(point["relative_day"])
        values = point.get("matched_user_drift", {})
        if not isinstance(values, dict):
            raise ValueError("matched_user_drift must map user IDs to drift")
        result[day] = {str(user): _finite(value, f"{user}/{day}") for user, value in values.items()}
    return result


def audit_nonstationarity(no_change_report: str | Path, temporary_report: str | Path,
                          sustained_report: str | Path, output_root: str | Path, *,
                          adaptation_threshold: float = 0.1,
                          recovery_threshold: float = 0.05,
                          overwrite: bool = False) -> dict[str, Any]:
    """Compare three authenticated pair reports without producing a winner."""
    reports, hashes = {}, {}
    for name, path in (("no-change", no_change_report), ("temporary-trip", temporary_report),
                       ("sustained-preference", sustained_report)):
        reports[name], hashes[name] = _load(path, name)
    identity = reports["no-change"].get("comparison_identity", {})
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"Comparison identity is incomplete: {missing}")
    for scenario, report in reports.items():
        candidate = report.get("comparison_identity", {})
        for field in IDENTITY_FIELDS:
            if candidate.get(field) != identity[field]:
                raise ValueError(f"Incompatible pair identities: {scenario} differs in {field}")
    representation_names = set(reports["no-change"].get("representations", {}))
    if not representation_names or any(set(r.get("representations", {})) != representation_names for r in reports.values()):
        raise ValueError("Incompatible representation identities")
    output = PairLayout.from_path(output_root)
    destination, markdown = output.nonstationarity_audit_json, output.nonstationarity_audit_markdown
    if destination.exists() or markdown.exists():
        if not overwrite:
            raise FileExistsError("Refusing to overwrite immutable nonstationarity audit")
        if any(p.exists() and (p.is_symlink() or not p.is_file()) for p in (destination, markdown)):
            raise ValueError("--overwrite targets must be regular audit files")
    results: dict[str, Any] = {}
    for representation in sorted(representation_names):
        role = reports["no-change"]["representations"][representation].get("selection_role")
        if role not in {"diagnostic_control", "selected_candidate"}:
            raise ValueError(f"Invalid selection role for {representation}")
        component_names = set(reports["no-change"]["representations"][representation].get("components", {}))
        if any(set(r["representations"][representation].get("components", {})) != component_names for r in reports.values()):
            raise ValueError(f"Incompatible component identity for {representation}")
        components = {}
        for component_name in sorted(component_names):
            curves = {scenario: _series(report["representations"][representation]["components"][component_name])
                      for scenario, report in reports.items()}
            shared_users = set(identity["users"])
            for curve in curves.values():
                for values in curve.values():
                    shared_users &= set(values)
            user_metrics, exclusions = [], []
            for user in sorted(set(identity["users"])):
                if user not in shared_users:
                    exclusions.append({"user_id": user, "reason": "missing_post_change_coverage"})
                    continue
                adjusted = {}
                for scenario in ("temporary-trip", "sustained-preference"):
                    adjusted[scenario] = [(day, values[user] - curves["no-change"].get(day, {}).get(user, np.nan))
                                          for day, values in curves[scenario].items()
                                          if user in curves["no-change"].get(day, {}) and day >= 0]
                adaptation = threshold_time(adjusted["sustained-preference"], adaptation_threshold,
                                            direction="above")
                temporary = adjusted["temporary-trip"]
                duration = int(reports["temporary-trip"]["change_contract"]["temporary_duration_days"])
                recovery = threshold_time(temporary, recovery_threshold, direction="below", origin_day=duration)
                temp_values = [v for d, v in temporary if d >= 0]
                post_values = [v for d, v in temporary if d >= duration]
                sustained_values = [v for d, v in adjusted["sustained-preference"] if d >= 0]
                user_metrics.append({"user_id": user, "time_to_adaptation": adaptation,
                    "recovery_time": recovery,
                    "forgetting_after_temporary_change": (max(temp_values) - post_values[-1]) if temp_values and post_values else None,
                    "permanent_drift_after_sustained_change": sustained_values[-1] if sustained_values else None})
            def values(key: str) -> list[float]:
                return [float(row[key]) for row in user_metrics if row[key] is not None]
            components[component_name] = {
                "matched_control": "no-change drift subtracted at identical user and relative day",
                "thresholds": {"adaptation": adaptation_threshold, "recovery": recovery_threshold},
                "time_to_adaptation": {"summary_days": _ci([float(r["time_to_adaptation"]["time_days"]) for r in user_metrics if r["time_to_adaptation"]["crossed"]]),
                    "right_censored": sum(r["time_to_adaptation"]["right_censored"] for r in user_metrics),
                    "left_censored": sum(r["time_to_adaptation"]["left_censored"] for r in user_metrics)},
                "recovery_time": {"summary_days": _ci([float(r["recovery_time"]["time_days"]) for r in user_metrics if r["recovery_time"]["crossed"]]),
                    "right_censored": sum(r["recovery_time"]["right_censored"] for r in user_metrics),
                    "left_censored": sum(r["recovery_time"]["left_censored"] for r in user_metrics)},
                "forgetting_after_temporary_change": _ci(values("forgetting_after_temporary_change")),
                "permanent_drift_after_sustained_change": _ci(values("permanent_drift_after_sustained_change")),
                "coverage": {"required_users": len(identity["users"]), "included_users": len(user_metrics)},
                "exclusions": exclusions, "per_user": user_metrics}
        results[representation] = {"selection_role": role, "components": components}
    report = {"schema_version": AUDIT_SCHEMA, "requirement": "R11", "comparison_identity": identity,
              "input_report_sha256": hashes, "representations": results,
              "aggregate_winner": None,
              "selection_dependent_conclusion": "available" if any(v["selection_role"] == "selected_candidate" for v in results.values()) else "unavailable: no selected_candidate",
              "limitations": ["Change semantics and thresholds are simulator-defined.",
                              "Diagnostic branch names do not establish component semantics."]}
    write_json(report, destination)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("# R11 nonstationarity audit\n\nNo aggregate winner is reported.\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n", encoding="utf-8")
    return report
