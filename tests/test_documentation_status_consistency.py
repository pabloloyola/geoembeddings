import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_status_consistency", ROOT / "scripts/check_status_consistency.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
inconsistent_declarations = MODULE.inconsistent_declarations


def test_stale_completed_ranking_tasks_are_rejected() -> None:
    tasks = "- [x] **T3.5 — Frozen ranker.**\n- [x] **T3.7 — Transfer slices.**\n"
    document = (
        "T3.5 is the next implementation task.\n\n"
        "Unseen-POI evaluation (T3.7) is still unimplemented.\n"
    )
    failures = inconsistent_declarations(tasks, document)
    assert any(failure.startswith("T3.5:") for failure in failures)
    assert any(failure.startswith("T3.7:") for failure in failures)


def test_current_status_and_handoff_match_checked_tasks() -> None:
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    for relative_path in ("docs/CURRENT_STATUS.md", "docs/AGENT_HANDOFF.md"):
        document = (ROOT / relative_path).read_text(encoding="utf-8")
        assert inconsistent_declarations(tasks, document) == []
