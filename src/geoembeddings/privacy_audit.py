"""Root-based orchestration for the diagnostic-control R12 privacy audit."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .io import read_json, sha256_file
from .layout import DatasetLayout, ExperimentLayout, PrivacyEvidenceLayout, UtilityReportLayout
from .privacy import (BASELINE_CHECKPOINT_IDENTITY, PRIVACY_AUDIT_SCHEMA_VERSION,
                      PrivacyInput, authenticate_privacy_inputs, write_privacy_audit)
from .privacy_evaluation import load_privacy_config
from .representation_schema import load_embedding_export
from .runtime_metadata import collect_runtime_metadata


def _parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Learned checkpoint lacks model_state: {path}")
    return sum(int(value.numel()) for value in state.values())


def _declaration(name: str, experiment: ExperimentLayout,
                 utility: UtilityReportLayout) -> PrivacyInput:
    baseline = name == "statistical_baseline"
    export_path = experiment.baseline_embeddings if baseline else experiment.embeddings
    export = load_embedding_export(export_path)
    users = tuple(sorted(set(export.arrays["user_id"].astype(str).tolist())))
    model_variant = str(np.asarray(export.arrays["model_variant"]).item())
    utility_path = utility.report(name)
    utility_value = read_json(utility_path)
    population = utility_value.get("population_identity", {})
    utility_users = tuple(sorted(map(str, population.get("users", users))))
    checkpoint = None if baseline else experiment.checkpoint
    checkpoint_identity = BASELINE_CHECKPOINT_IDENTITY if baseline else sha256_file(checkpoint)
    return PrivacyInput(
        name=name, kind="statistical_baseline" if baseline else "learned",
        export_path=export_path, prepared_metadata_path=experiment.prepared_metadata,
        utility_report_path=utility_path, selection_role="diagnostic_control",
        parameter_count=0 if baseline else _parameter_count(checkpoint),
        eligible_users=users, utility_report_users=utility_users,
        checkpoint_path=checkpoint, checkpoint_identity=checkpoint_identity,
        model_variant=model_variant,
    )


def audit_privacy(*, run_dir: str | Path, experiments: Mapping[str, str | Path],
                  evidence_dir: str | Path, utility_report_dir: str | Path,
                  config_path: str | Path, output_dir: str | Path,
                  overwrite: bool = False) -> dict[str, Any]:
    """Authenticate diagnostic controls, then publish a bounded R12 report.

    Membership remains scientifically unavailable when the canonical roots do
    not provide authenticated user-level target-training participation. Export
    coverage is deliberately never repurposed as a membership label.
    """
    started = time.monotonic()
    config = load_privacy_config(config_path)  # Validate before any protected access.
    run = DatasetLayout.from_path(run_dir)
    run.validate(require_truth=False)
    if not experiments or "statistical_baseline" not in experiments:
        raise ValueError("audit-privacy requires a statistical_baseline diagnostic control")
    layouts = {name: ExperimentLayout.from_path(root) for name, root in experiments.items()}
    utility = UtilityReportLayout.from_path(utility_report_dir)
    declarations = [_declaration(name, layout, utility) for name, layout in sorted(layouts.items())]
    evidence = PrivacyEvidenceLayout.from_path(evidence_dir)
    authenticated = authenticate_privacy_inputs(evidence.evidence_index, declarations)
    source_hashes = dict(authenticated.inputs[0].observed_source_hashes)
    for filename, expected in source_hashes.items():
        source = run.observed / filename
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Dataset observed-source authentication failed: {filename}")

    # Authentication has succeeded. This lineage has no canonical, authenticated
    # participation artifact, so opening protected labels cannot produce a valid
    # membership experiment and is intentionally avoided.
    membership: dict[str, Any] = {}
    membership_metrics: dict[str, Any] = {}
    for identity in authenticated.inputs:
        if identity.kind == "statistical_baseline":
            membership[identity.name] = {"status": "not_applicable", "reason": "no_learned_target_parameters"}
        else:
            membership[identity.name] = {
                "supported": False, "status": "unavailable",
                "reason": "authenticated_training_membership_labels_unavailable",
            }
            membership_metrics[identity.name] = {
                "status": "unavailable", "reason": "authenticated_training_membership_labels_unavailable"
            }
    sensitive = {
        item.name: {"status": "unavailable", "reason": "authenticated_privacy_population_unavailable",
                    "derivation_version": item.derivation.version}
        for item in config.sensitive_attributes
    }
    utility_axes = {}
    for declaration in declarations:
        value = read_json(declaration.utility_report_path)
        utility_axes[declaration.name] = {
            "report_path": str(declaration.utility_report_path),
            "report_sha256": sha256_file(declaration.utility_report_path),
            "metrics": value.get("utility_metrics", value.get("metrics", {})),
            "coverage": value.get("coverage", {}),
        }
    runtime = collect_runtime_metadata(duration_seconds=time.monotonic() - started,
                                       seed=config.audit_seed)
    identities = [asdict(item) for item in authenticated.inputs]
    report = {
        "schema_version": PRIVACY_AUDIT_SCHEMA_VERSION,
        "threat_model": {"schema_version": config.schema_version,
                         "membership": config.membership,
                         "primary_endpoints": config.reporting["primary_endpoints"]},
        "inputs": {"dataset_root": str(run.root), "evidence_root": str(evidence.root),
                   "utility_report_root": str(utility.root), "controls": identities,
                   "evidence_index_sha256": authenticated.evidence_index_sha256},
        "lineage": {"dataset_contract": identities[0]["dataset_contract"],
                    "observed_source_hashes": identities[0]["observed_source_hashes"],
                    "preparation_definition_sha256": identities[0]["preparation_definition_sha256"]},
        "splits": {"status": "unavailable", "reason": "authenticated_training_membership_labels_unavailable",
                   "split_seed": config.split_seed},
        "membership_population": membership,
        "sensitive_attributes": sensitive,
        "attacks": {"status": "not_run", "reason": "authenticated_privacy_population_unavailable",
                    "configuration": config.attacks},
        "membership_metrics": membership_metrics,
        "sensitive_probe_metrics": sensitive,
        "utility_privacy_axes": utility_axes,
        "coverage": {"authenticated_controls": len(identities), "membership_evaluated_controls": 0},
        "exclusions": [{"reason": "authenticated_training_membership_labels_unavailable",
                        "count": sum(item["kind"] == "learned" for item in identities)}],
        "selection": {"roles": {item["name"]: item["selection_role"] for item in identities},
                      "selection_dependent_privacy_conclusion": {
                          "status": "unavailable", "reason": "no_selected_candidate"}},
        "limitations": ["Diagnostic controls do not establish deployment privacy.",
                        "Component names do not establish factorized semantics.",
                        "Unavailable membership evidence is not evidence of privacy."],
        "command": "geoembed audit-privacy",
        "timestamps": {"created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
        "runtime_metadata": runtime.to_dict(),
    }
    json_path, markdown_path = write_privacy_audit(report, output_dir, overwrite=overwrite)
    return {"privacy_json": str(json_path), "privacy_markdown": str(markdown_path),
            "supported_diagnostic_conclusion": "diagnostic-control audit only; selection-dependent privacy conclusion unavailable: no_selected_candidate"}
