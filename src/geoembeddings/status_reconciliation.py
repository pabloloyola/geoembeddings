"""Validate indexed evidence and derive a non-composite reference decision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import read_json, sha256_file

INDEX_SCHEMA = "geoembeddings-evidence-index/1.0"
DECISION_SCHEMA = "geoembeddings-decision/1.0"
ALLOWED_ACTIONS = ("repair/ablate", "finish_evaluator_gate", "factorize")


@dataclass(frozen=True)
class AxisConclusion:
    name: str
    conclusion: str
    coverage: str
    missingness: str


@dataclass(frozen=True)
class Reconciliation:
    action: str
    axes: tuple[AxisConclusion, ...]


def reconcile_artifact_index(
    artifact_index: str | Path, *, repository_root: str | Path | None = None
) -> Reconciliation:
    """Load, authenticate, and reconcile every indexed JSON scientific report."""
    index_path = Path(artifact_index).resolve()
    index = read_json(index_path)
    root = Path(repository_root).resolve() if repository_root else _repository_root(index_path)
    _validate_index(index)
    artifacts = _artifact_map(index)
    reports: dict[str, dict[str, Any]] = {}
    for artifact_id in _required_report_ids():
        entry = artifacts.get(artifact_id)
        if entry is None:
            raise ValueError(f"Required indexed report is absent: {artifact_id}")
        path = _resolve_local(entry, root)
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ValueError(
                f"Indexed artifact hash mismatch for {artifact_id}: "
                f"index={entry.get('sha256')!r}, actual={actual!r}"
            )
        reports[artifact_id] = read_json(path)

    prepared_entry = artifacts.get("prepared_metadata")
    if prepared_entry is None:
        raise ValueError("Required indexed preparation metadata is absent")
    prepared_path = _resolve_local(prepared_entry, root)
    if sha256_file(prepared_path) != prepared_entry.get("sha256"):
        raise ValueError("Indexed preparation metadata hash mismatch")
    prepared = read_json(prepared_path)
    _validate_report_contracts(index, prepared, reports)
    return derive_status(reports, index.get("coverage_and_missingness", {}))


def derive_status(
    reports: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Any]
) -> Reconciliation:
    """Derive separate scientific axes; missing axes select the evaluator gate."""
    comparison = reports.get("embedding_comparison_json", {})
    required = {
        "persistent probes": "persistent_information",
        "incremental information": "preference_beyond_geography_and_activity",
        "collapse/geometry": "stability_and_distinctiveness",
        "episode response": "episode_response_comparison",
        "robustness": "R6_R7_robustness_comparison",
        "next-event performance/coverage": "common_future_event_probes",
    }
    missing = [label for label, key in required.items() if not comparison.get(key)]
    if missing:
        axes = tuple(
            AxisConclusion(
                label,
                "pending" if label in missing else "available; interpret separately",
                _global_coverage(coverage, comparison),
                f"missing required axis: {label}" if label in missing else _axis_missingness(label, reports),
            )
            for label in required
        )
        return Reconciliation("finish_evaluator_gate", axes)

    persistent = comparison["persistent_information"]
    incremental = comparison["preference_beyond_geography_and_activity"]
    geometry = comparison["stability_and_distinctiveness"]
    persistent_delta = _delta(persistent, "mean_r2")
    incremental_delta = _number(incremental.get("learned_minus_baseline_incremental_r2"))
    geometry_warning = _collapse_warning(geometry)
    action = "repair/ablate" if geometry_warning or persistent_delta < 0 or incremental_delta < 0 else "factorize"

    conclusions = {
        "persistent probes": f"learned-minus-baseline mean R2 {persistent_delta:+.4f}",
        "incremental information": f"learned-minus-baseline incremental R2 {incremental_delta:+.4f}",
        "collapse/geometry": "collapse warning triggered" if geometry_warning else "no configured collapse warning; geometry remains diagnostic",
        "episode response": _delta_summary(comparison["episode_response_comparison"]),
        "robustness": _robustness_summary(comparison["R6_R7_robustness_comparison"]),
        "next-event performance/coverage": _future_summary(comparison["common_future_event_probes"]),
    }
    axes = tuple(
        AxisConclusion(label, conclusions[label], _axis_coverage(label, reports, comparison, coverage), _axis_missingness(label, reports))
        for label in required
    )
    if action not in ALLOWED_ACTIONS:  # defensive assertion for future edits
        raise AssertionError(action)
    return Reconciliation(action, axes)


def render_decision(result: Reconciliation, artifact_index: str) -> str:
    lines = [
        "# T0.2a reference decision", "", "```yaml", f"schema_version: {DECISION_SCHEMA}",
        "task_id: T0.2a", "status: complete", f"artifact_index: {artifact_index}",
        f"selected_action: {result.action}", "aggregate_winner: null", "```", "",
        "## Comparability audit", "", "The indexed comparability audit passed and every consumed report matched its indexed SHA-256 digest.", "",
        "## Per-axis conclusions", "", "| Axis | Conclusion | Coverage | Missing-label/user qualifications |",
        "|---|---|---|---|",
    ]
    for axis in result.axes:
        lines.append(f"| {axis.name} | {axis.conclusion} | {axis.coverage} | {axis.missingness} |")
    lines += ["", "No aggregate score or aggregate winner is calculated.", "", "## Selected action", "", f"Exactly one action is selected: **{result.action}**.", ""]
    return "\n".join(lines)


def _validate_index(index: Mapping[str, Any]) -> None:
    if index.get("schema_version") != INDEX_SCHEMA or index.get("task_id") != "T0.2":
        raise ValueError("Artifact index is not a T0.2 evidence index")
    if index.get("evidence_status", "complete") != "complete":
        raise ValueError("T0.2 evidence index is not complete")
    audit = index.get("comparability_audit", {})
    if audit.get("result") != "passed" or audit.get("blocking_reasons"):
        raise ValueError(f"T0.2 comparability audit did not pass: {audit.get('blocking_reasons', [])}")
    required_flags = (
        "baseline_and_learned_source_hashes_match", "cutoffs_match",
        "categorical_field_order_matches", "continuous_field_order_matches", "users_match",
        "dense_users_and_timestamps_match", "preparation_contract_matches",
        "robustness_specifications_and_masks_match",
    )
    bad = [name for name in required_flags if audit.get(name) is not True]
    if bad:
        raise ValueError(f"Comparability audit lacks required passing checks: {bad}")
    left, right = index.get("evidence_identity", {}).get("baseline", {}), index.get("evidence_identity", {}).get("learned", {})
    for field in ("observed_source_hashes", "cutoffs", "categorical_field_order", "continuous_field_order", "user_set_sha256", "preparation_metadata_sha256"):
        if left.get(field) != right.get(field):
            raise ValueError(f"Baseline/learned {field} mismatch")


def _validate_report_contracts(index: Mapping[str, Any], prepared: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]) -> None:
    provenance = index["provenance"]
    prep = provenance.get("preparation_metadata", provenance)
    expected_cat = prep.get("categorical_field_order", provenance.get("categorical_field_order"))
    expected_cont = prep.get("continuous_field_order", provenance.get("continuous_field_order"))
    if prepared.get("categorical_fields") != expected_cat or prepared.get("continuous_fields") != expected_cont:
        raise ValueError("Preparation metadata field order mismatch")
    expected_cutoffs = provenance.get("cutoffs")
    actual_cutoffs = {"train_end": prepared.get("train_end"), "validation_end": prepared.get("validation_end")}
    if actual_cutoffs != expected_cutoffs:
        raise ValueError("Preparation metadata cutoff mismatch")
    expected_sources = provenance.get("observed_source_hashes")
    if _canonical_sources(prepared.get("source_files", {})) != _canonical_sources(expected_sources or {}):
        raise ValueError("Preparation metadata observed-source hash mismatch")
    for kind in ("baseline", "learned"):
        robustness = reports[f"{kind}_robustness_report"]
        contract = robustness.get("metric_contract", robustness)
        report_sources = contract.get("source_hashes", robustness.get("source_hashes", {}))
        if _canonical_sources(report_sources) != _canonical_sources(expected_sources or {}):
            raise ValueError(f"{kind} robustness source hashes mismatch")
        if contract.get("field_order", robustness.get("field_order")) != reports["baseline_robustness_report"].get("metric_contract", reports["baseline_robustness_report"]).get("field_order", reports["baseline_robustness_report"].get("field_order")):
            raise ValueError(f"{kind} robustness field order mismatch")
    left = reports["baseline_robustness_report"].get("metric_contract", reports["baseline_robustness_report"])
    right = reports["learned_robustness_report"].get("metric_contract", reports["learned_robustness_report"])
    for field in ("specification_hash", "view_ids", "mask_hashes"):
        if not left.get(field) or left.get(field) != right.get(field):
            raise ValueError(f"Required robustness {field} mismatch or absent")


def _required_report_ids() -> tuple[str, ...]:
    return ("baseline_evaluation_report", "learned_evaluation_report", "baseline_episode_response_report", "learned_episode_response_report", "baseline_robustness_report", "learned_robustness_report", "embedding_comparison_json")


def _artifact_map(index: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {entry["id"]: entry for group in index.get("required_artifacts", {}).values() for entry in group}


def _resolve_local(entry: Mapping[str, Any], root: Path) -> Path:
    identifier = entry.get("identifier")
    if entry.get("status") != "present" or not identifier:
        raise ValueError(f"Indexed artifact is not present: {entry.get('id')}")
    if "://" in str(identifier):
        raise ValueError(f"Cannot authenticate non-local indexed report: {identifier}")
    path = Path(str(identifier)); path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"Indexed artifact is unavailable: {path}")
    return path


def _repository_root(index_path: Path) -> Path:
    return index_path.parents[2] if index_path.parent.name == "artifacts" and index_path.parent.parent.name == "docs" else Path.cwd().resolve()


def _number(value: Any) -> float:
    if value is None: raise ValueError("Required numeric metric is missing")
    return float(value)


def _delta(group: Mapping[str, Any], metric: str) -> float:
    return _number(group["learned"][metric]) - _number(group["baseline"][metric])


def _collapse_warning(geometry: Mapping[str, Any]) -> bool:
    learned = geometry["learned"]
    return (_number(learned["test_geometry"]["effective_rank_ratio"]) <= 0 or _number(learned["same_minus_different_train_test_cosine"]) <= 0 or _number(learned["temporal_user_retrieval"]["train_query_test_gallery_top1"]) <= 0)


def _canonical_sources(values: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {"users": "users_observed.csv.gz", "events": "observed_events.csv.gz"}
    return {aliases.get(key, key): value for key, value in values.items()}


def _delta_summary(values: Mapping[str, Any]) -> str:
    return "; ".join(f"{key} Δ={_number(value['learned_minus_baseline']):+.4f}" for key, value in values.items())


def _robustness_summary(values: Mapping[str, Any]) -> str:
    return f"{len(values.get('R6_views', []))} missing-service and {len(values.get('R7_views', []))} perturbation views; axes not composited"


def _future_summary(values: Mapping[str, Any]) -> str:
    targets = [v for k, v in values.items() if k != "protocol" and isinstance(v, Mapping)]
    comparable = sum("learned_minus_baseline_accuracy" in value for value in targets)
    return f"{comparable}/{len(targets)} targets have matched accuracy deltas"


def _global_coverage(coverage: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    shared = comparison.get("comparison_contract", {}).get("shared_users", coverage.get("users_with_observed_events", "unknown"))
    return f"{shared} shared users; {coverage.get('users_without_observed_events', 'unknown')} simulated users lacked observed events"


def _axis_coverage(label: str, reports: Mapping[str, Mapping[str, Any]], comparison: Mapping[str, Any], coverage: Mapping[str, Any]) -> str:
    if label == "next-event performance/coverage":
        values = comparison["common_future_event_probes"]
        known = [v.get("known_label_coverage") for k, v in values.items() if k != "protocol" and isinstance(v, Mapping)]
        return f"{_global_coverage(coverage, comparison)}; known-label coverage={known}"
    if label == "robustness":
        rows = comparison["R6_R7_robustness_comparison"].get("R6_views", []) + comparison["R6_R7_robustness_comparison"].get("R7_views", [])
        return f"{_global_coverage(coverage, comparison)}; per-view matched rows/coverage={[ (r.get('matched_rows'), r.get('coverage')) for r in rows ]}"
    return _global_coverage(coverage, comparison)


def _axis_missingness(label: str, reports: Mapping[str, Mapping[str, Any]]) -> str:
    if label == "episode response":
        return "episode reports retain missing users, labels, and empty response bins"
    if label == "robustness":
        return "reports retain unencodable keys and insufficient-history exclusions"
    if label == "next-event performance/coverage":
        return "unknown target labels are excluded and qualified by known-label coverage"
    return "held-out probe users and missing labels remain qualified in source reports"
