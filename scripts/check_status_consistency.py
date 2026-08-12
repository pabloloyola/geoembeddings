#!/usr/bin/env python3
"""Reject status documents that describe completed TASKS.md items as future work."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TASK_PATTERN = re.compile(r"^- \[[xX]\] \*\*(T\d+(?:\.\d+)?[a-z]?)\b", re.MULTILINE)
TASK_REFERENCE = re.compile(r"\bT\d+(?:\.\d+)?[a-z]?\b")
FUTURE_DECLARATION = re.compile(
    r"\b(?:pending|unimplemented)\b|\bnext\b.{0,100}\b(?:task|implementation|option)\b",
    re.IGNORECASE | re.DOTALL,
)
DEFAULT_STATUS_DOCUMENTS = ("docs/CURRENT_STATUS.md", "docs/AGENT_HANDOFF.md")


def completed_task_ids(tasks_text: str) -> set[str]:
    """Return identifiers whose canonical TASKS.md checkbox is checked."""
    return set(TASK_PATTERN.findall(tasks_text))


def inconsistent_declarations(tasks_text: str, document_text: str) -> list[str]:
    """Find paragraphs that declare a checked task pending, unimplemented, or next."""
    completed = completed_task_ids(tasks_text)
    failures: list[str] = []
    table_rows = [line for line in document_text.splitlines() if line.startswith("|")]
    prose = "\n".join(
        line for line in document_text.splitlines() if not line.startswith("|")
    )
    declarations = table_rows + re.split(r"\n\s*\n", prose)
    for paragraph in declarations:
        if not FUTURE_DECLARATION.search(paragraph):
            continue
        stale_ids = sorted(completed.intersection(TASK_REFERENCE.findall(paragraph)))
        if stale_ids:
            excerpt = " ".join(paragraph.split())[:180]
            failures.append(f"{', '.join(stale_ids)}: {excerpt}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("documents", nargs="*", default=list(DEFAULT_STATUS_DOCUMENTS))
    args = parser.parse_args()
    tasks_text = (args.repository_root / "TASKS.md").read_text(encoding="utf-8")
    failures: list[str] = []
    for relative_path in args.documents:
        path = args.repository_root / relative_path
        for declaration in inconsistent_declarations(tasks_text, path.read_text(encoding="utf-8")):
            failures.append(f"{relative_path}: {declaration}")
    if failures:
        parser.exit(1, "stale task status declarations:\n- " + "\n- ".join(failures) + "\n")
    print(
        f"status consistency passed: {len(completed_task_ids(tasks_text))} completed tasks; "
        f"{len(args.documents)} documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
