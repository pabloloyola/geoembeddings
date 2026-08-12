#!/usr/bin/env python3
"""Statically validate an external-validity evidence registry in Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REGISTRY = re.compile(r"```evidence-registry\s*\n(.*?)\n```", re.DOTALL)
REQUIREMENT_ID = re.compile(r"R(?:[1-9]|1[0-3])\Z")
TASK_ID = re.compile(r"T\d+(?:\.\d+)?[a-z]?\Z")
TASK_HEADING = re.compile(r"^- \[[ xX]\] \*\*(T\d+(?:\.\d+)?[a-z]?)\b", re.MULTILINE)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PROHIBITED_EXTERNAL_CLAIM = re.compile(
    r"(?:external(?:ly)? valid(?:ation|ity)?|real-world valid(?:ation|ity)?|"
    r"validat(?:es|ed) (?:in|for) (?:the )?real world)", re.IGNORECASE
)
REQUIRED_CLAIM_FIELDS = {
    "artifact",
    "cohort_size",
    "seed",
    "source_identity",
    "preparation_identity",
    "scientific_scope",
    "evidence_kind",
    "statement",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(text: str) -> dict[str, Any]:
    matches = REGISTRY.findall(text)
    if len(matches) != 1:
        raise ValueError("document must contain exactly one evidence-registry block")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError("evidence registry must be a JSON object")
    return value


def validate_document(document: Path, repository_root: Path) -> list[str]:
    """Return deterministic validation failures without importing project code."""
    try:
        registry = load_registry(document.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]

    errors: list[str] = []
    tasks_path = repository_root / "TASKS.md"
    known_tasks = set(TASK_HEADING.findall(tasks_path.read_text(encoding="utf-8")))
    requirement_records = registry.get("requirements", [])
    ids = [record.get("id") for record in requirement_records if isinstance(record, dict)]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate requirement IDs: {', '.join(duplicates)}")
    unknown = sorted({str(item) for item in ids if not isinstance(item, str) or not REQUIREMENT_ID.fullmatch(item)})
    if unknown:
        errors.append(f"unknown requirement IDs: {', '.join(unknown)}")
    missing = sorted({f"R{i}" for i in range(1, 14)} - set(ids))
    if missing:
        errors.append(f"missing requirement IDs: {', '.join(missing)}")

    artifacts: dict[str, dict[str, Any]] = {}
    for artifact in registry.get("artifacts", []):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            errors.append("artifact entry must have a string path")
            continue
        relative = artifact["path"]
        artifacts[relative] = artifact
        path = repository_root / relative
        availability = artifact.get("availability")
        if availability == "unavailable_historical":
            if artifact.get("task_id") != "T0.2":
                errors.append(f"{relative}: unavailable historical evidence must be T0.2")
            if artifact.get("accepted") is not False:
                errors.append(f"{relative}: unavailable historical evidence cannot be accepted")
        else:
            if not path.is_file():
                errors.append(f"missing artifact: {relative}")
                continue
            recorded = artifact.get("sha256")
            if not isinstance(recorded, str) or not SHA256.fullmatch(recorded):
                errors.append(f"{relative}: invalid SHA-256 identifier")
            elif _sha256(path) != recorded:
                errors.append(f"{relative}: SHA-256 mismatch")
        task_id = artifact.get("task_id")
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id) or task_id not in known_tasks:
            errors.append(f"{relative}: unknown task ID {task_id!r}")

    for index, claim in enumerate(registry.get("claims", []), start=1):
        label = f"claim {index}"
        if not isinstance(claim, dict):
            errors.append(f"{label}: must be an object")
            continue
        absent = sorted(REQUIRED_CLAIM_FIELDS - claim.keys())
        if absent:
            errors.append(f"{label}: missing claim metadata: {', '.join(absent)}")
        artifact = claim.get("artifact")
        if artifact not in artifacts:
            errors.append(f"{label}: references unregistered artifact {artifact!r}")
        if claim.get("cohort_size") in (None, ""):
            errors.append(f"{label}: absent cohort metadata")
        if claim.get("seed") in (None, ""):
            errors.append(f"{label}: absent seed metadata")
        if not claim.get("scientific_scope"):
            errors.append(f"{label}: absent scientific scope metadata")
        requirements = claim.get("requirement_ids", [])
        for requirement in requirements:
            if not isinstance(requirement, str) or not REQUIREMENT_ID.fullmatch(requirement):
                errors.append(f"{label}: unknown requirement ID {requirement!r}")
        task_id = claim.get("task_id")
        if task_id not in known_tasks:
            errors.append(f"{label}: unknown task ID {task_id!r}")
        statement = str(claim.get("statement", ""))
        if claim.get("evidence_kind") == "simulator_only" and PROHIBITED_EXTERNAL_CLAIM.search(statement):
            errors.append(f"{label}: simulator-only result is described as external validation")

    if registry.get("completion_is_scientific_success") is not False:
        errors.append("task completion must be explicitly distinct from scientific success")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    document = args.document if args.document.is_absolute() else args.repository_root / args.document
    errors = validate_document(document, args.repository_root)
    if errors:
        parser.exit(1, "evidence validation failed:\n- " + "\n- ".join(errors) + "\n")
    print(f"evidence validation passed: {document.relative_to(args.repository_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
