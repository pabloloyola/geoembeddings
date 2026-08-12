from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoembeddings.nonstationarity import audit_nonstationarity, threshold_time


def test_threshold_equality_and_never_crossed() -> None:
    assert threshold_time([(0, .1)], .1, direction="above")["time_days"] == 0
    never = threshold_time([(0, .09), (2, .09)], .1, direction="above")
    assert never["right_censored"] and not never["crossed"]


def test_threshold_left_right_censoring_and_missing_coverage() -> None:
    assert threshold_time([(2, .2)], .1, direction="above")["left_censored"]
    assert threshold_time([(0, .2)], .1, direction="below")["right_censored"]
    assert threshold_time([], .1, direction="above")["missing_post_change_coverage"]


def test_threshold_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        threshold_time([(0, float("nan"))], .1, direction="above")


def _report(path: Path, intervention: str, *, users: list[str] | None = None) -> None:
    users = users or ["u1", "u2"]
    identity = {"users": users, "cutoffs": ["2026-01-01"], "preparation_contract": {"hash": "p"},
        "source_lineage": {"root": "s"}, "component_schema": {"combined": 2},
        "relative_day_definition": "floor((timestamp-change_start)/86400)",
        "censoring_rules": "first observed crossing; final observation right-censors"}
    curve = [{"relative_day": day, "matched_user_drift": {u: value for u in users}}
             for day, value in [(-1, 0), (0, .1), (1, .2), (2, .02)]]
    payload = {"schema_version": "geoembeddings-change-evaluation/2.0", "intervention": intervention,
        "authentication": {"status": "passed"}, "comparison_identity": identity,
        "change_contract": {"temporary_duration_days": 2},
        "representations": {
            "statistical_baseline": {"selection_role": "diagnostic_control", "components": {"combined": {"curve": curve}}},
            "capacity_matched_single": {"selection_role": "diagnostic_control", "components": {"combined": {"curve": curve}}},
            "factorized_diagnostic": {"selection_role": "diagnostic_control", "components": {
                "persistent": {"curve": curve}, "context": {"curve": curve}, "combined": {"curve": curve}}}}}
    path.write_text(json.dumps(payload))


def test_audit_reports_roles_components_coverage_and_censoring(tmp_path: Path) -> None:
    paths = []
    for name in ("no-change", "temporary-trip", "sustained-preference"):
        path = tmp_path / f"{name}.json"; _report(path, name); paths.append(path)
    report = audit_nonstationarity(*paths, tmp_path / "output")
    assert report["aggregate_winner"] is None
    assert report["selection_dependent_conclusion"].startswith("unavailable")
    component = report["representations"]["factorized_diagnostic"]["components"]["persistent"]
    assert component["coverage"]["included_users"] == 2
    assert (tmp_path / "output/audits/nonstationarity.md").is_file()


def test_audit_rejects_incompatible_pair_identity(tmp_path: Path) -> None:
    paths = []
    for name in ("no-change", "temporary-trip", "sustained-preference"):
        path = tmp_path / f"{name}.json"; _report(path, name, users=["other"] if name == "temporary-trip" else None); paths.append(path)
    with pytest.raises(ValueError, match="users"):
        audit_nonstationarity(*paths, tmp_path / "output")
