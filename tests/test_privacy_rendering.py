import json
from pathlib import Path

import pytest

import geoembeddings.privacy as privacy_module
from geoembeddings.layout import PairLayout
from geoembeddings.privacy import render_privacy_markdown, write_privacy_audit


def _report():
    return {
        "schema_version": "geoembeddings-privacy-audit/1.0",
        "threat_model": {}, "inputs": {}, "lineage": {}, "splits": {},
        "membership_population": {
            "statistical_baseline": {"status": "not_applicable", "reason": "no_learned_target_parameters"},
            "learned": {"supported": False, "status": "unavailable", "reason": "no_nonmember_population"},
        },
        "sensitive_attributes": {}, "attacks": {},
        "membership_metrics": {}, "sensitive_probe_metrics": {},
        "utility_privacy_axes": {}, "coverage": {}, "exclusions": [],
        "selection": {"selection_dependent_privacy_conclusion": {
            "status": "unavailable", "reason": "no_selected_candidate"}},
        "limitations": [], "command": "geoembed audit-privacy",
        "timestamps": {"created_at": "2026-08-13T00:00:00Z"},
        "runtime_metadata": {},
    }


def test_privacy_paths_and_authoritative_rendering(tmp_path: Path) -> None:
    json_path, markdown_path = write_privacy_audit(_report(), tmp_path)
    layout = PairLayout.from_path(tmp_path)
    assert (json_path, markdown_path) == (layout.privacy_audit_json, layout.privacy_audit_markdown)
    loaded = json.loads(json_path.read_text())
    assert json.dumps(loaded, indent=2, sort_keys=True) in markdown_path.read_text()
    with pytest.raises(FileExistsError):
        write_privacy_audit(_report(), tmp_path)


def test_privacy_report_rejects_prohibited_or_misleading_results() -> None:
    report = _report(); report["aggregate_winner"] = None
    with pytest.raises(ValueError, match="aggregate"):
        render_privacy_markdown(report)
    report = _report(); report["membership_population"]["learned"]["status"] = "failed"
    with pytest.raises(ValueError, match="unavailable"):
        render_privacy_markdown(report)


def test_privacy_pair_write_removes_partial_output_on_publish_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace = privacy_module.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(privacy_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated"):
        write_privacy_audit(_report(), tmp_path)
    layout = PairLayout.from_path(tmp_path)
    assert not layout.privacy_audit_json.exists()
    assert not layout.privacy_audit_markdown.exists()
