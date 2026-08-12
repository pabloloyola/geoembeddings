import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_evidence_links", ROOT / "scripts/check_evidence_links.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def registry() -> dict:
    return {
        "completion_is_scientific_success": False,
        "requirements": [{"id": f"R{i}", "status_change_evidence": "external study"} for i in range(1, 14)],
        "artifacts": [{"path": "artifact.json", "sha256": "0" * 64, "task_id": "T4.5", "availability": "local", "accepted": True}],
        "claims": [{
            "artifact": "artifact.json", "task_id": "T4.5", "requirement_ids": ["R1"],
            "cohort_size": 50, "seed": 7, "source_identity": "commit", "preparation_identity": "prep hash",
            "scientific_scope": "synthetic diagnostic", "evidence_kind": "simulator_only", "statement": "metric was 0.5",
        }],
    }


def validate(tmp_path: Path, value: dict) -> list[str]:
    (tmp_path / "TASKS.md").write_text(
        "- [x] **T0.2 — Historical evidence.**\n- [ ] **T4.5 — Evidence.**\n"
    )
    (tmp_path / "artifact.json").write_text("bytes")
    doc = tmp_path / "doc.md"
    doc.write_text("```evidence-registry\n" + json.dumps(value) + "\n```\n")
    return MODULE.validate_document(doc, tmp_path)


def test_missing_artifact_and_incorrect_hash_are_rejected(tmp_path: Path) -> None:
    value = registry()
    errors = validate(tmp_path, value)
    assert any("SHA-256 mismatch" in error for error in errors)
    value["artifacts"][0]["path"] = "missing.json"
    assert any("missing artifact" in error for error in validate(tmp_path, value))


def test_absent_claim_metadata_is_rejected(tmp_path: Path) -> None:
    value = registry()
    for field in ("cohort_size", "seed", "scientific_scope"):
        value["claims"][0].pop(field)
    errors = validate(tmp_path, value)
    assert any("cohort_size" in error and "seed" in error and "scientific_scope" in error for error in errors)


def test_duplicate_and_unknown_requirement_ids_are_rejected(tmp_path: Path) -> None:
    value = registry()
    value["requirements"].append({"id": "R1"})
    value["requirements"][1]["id"] = "R14"
    errors = validate(tmp_path, value)
    assert any("duplicate requirement IDs: R1" in error for error in errors)
    assert any("unknown requirement IDs: R14" in error for error in errors)


def test_simulator_result_cannot_claim_external_validation(tmp_path: Path) -> None:
    value = registry()
    value["claims"][0]["statement"] = "This externally validates the representation."
    assert any("simulator-only result" in error for error in validate(tmp_path, value))


def test_historical_t02_is_distinct_from_accepted_t04(tmp_path: Path) -> None:
    value = registry()
    value["artifacts"] = [
        {"path": "lost.bin", "task_id": "T0.2", "availability": "unavailable_historical", "accepted": False},
        {"path": "artifact.json", "task_id": "T4.5", "availability": "local", "accepted": True, "sha256": "0" * 64},
    ]
    assert not any("lost.bin" in error for error in validate(tmp_path, value))
