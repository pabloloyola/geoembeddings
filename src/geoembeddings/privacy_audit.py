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
                      SUPPORTED_PROTECTED_ATTRIBUTES, PrivacyInput,
                      authenticate_privacy_inputs, construct_privacy_population,
                      construct_sensitive_probe_population,
                      load_protected_labels, run_protected_attribute_attacks,
                      run_privacy_attacks_from_config,
                      write_privacy_audit)
from .privacy_evaluation import load_privacy_config
from .representation_schema import load_embedding_export
from .runtime_metadata import collect_runtime_metadata
from .user_roles import (PROTOCOL_SCHEMA as USER_ROLE_SCHEMA, assign_users,
                         assignment_hash, role_summary)


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


def _authenticated_membership(experiment: ExperimentLayout, declaration: PrivacyInput,
                              evidence_index: Mapping[str, Any]) -> dict[str, bool] | None:
    """Recover membership only from an indexed, checkpoint-bound role protocol."""
    path = experiment.training_participation
    if not path.is_file():
        return None
    artifacts = evidence_index.get("artifacts", {})
    indexed = next((artifacts[key] for key in (str(path), str(path.resolve()))
                    if key in artifacts), None) if isinstance(artifacts, Mapping) else None
    if not isinstance(indexed, Mapping) or indexed.get("sha256") != sha256_file(path):
        return None
    value = read_json(path)
    protocol = value.get("user_role_protocol")
    if (value.get("schema_version") != "geoembeddings-training-participation/1.0"
            or not isinstance(protocol, Mapping)
            or protocol.get("schema_version") != USER_ROLE_SCHEMA):
        return None
    if value.get("checkpoint_identity", {}).get("sha256") != declaration.checkpoint_identity:
        return None
    if value.get("preparation_identity", {}).get("prepared_metadata_sha256") != sha256_file(declaration.prepared_metadata_path):
        return None
    assignments = assign_users(declaration.eligible_users, protocol)
    expected = {
        "schema_version": USER_ROLE_SCHEMA, "seed": int(protocol["seed"]),
        "fractions": {name: float(protocol["fractions"][name])
                      for name in ("target_train", "target_validation", "target_test")},
        "assignment_sha256": assignment_hash(assignments), "roles": role_summary(assignments),
    }
    if dict(protocol) != expected:
        return None
    # Validation users selected the checkpoint and are therefore neither clean
    # members nor clean non-members under the frozen threat model.
    return {user: role == "target_train" for user, role in assignments.items()
            if role != "target_validation"}


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

    # Authentication has succeeded. Membership and sensitive-label applicability
    # are independent: export coverage is never repurposed as participation.
    membership: dict[str, Any] = {}
    membership_metrics: dict[str, Any] = {}
    evidence_value = read_json(evidence.evidence_index)
    membership_labels: dict[str, dict[str, bool]] = {}
    for identity in authenticated.inputs:
        if identity.kind == "statistical_baseline":
            membership[identity.name] = {"status": "not_applicable", "reason": "no_learned_target_parameters"}
        else:
            declaration = next(item for item in declarations if item.name == identity.name)
            labels = _authenticated_membership(layouts[identity.name], declaration, evidence_value)
            if labels is None:
                membership[identity.name] = {"supported": False, "status": "unavailable",
                                             "reason": "authenticated_training_membership_labels_unavailable"}
                membership_metrics[identity.name] = {"status": "unavailable",
                                                     "reason": "authenticated_training_membership_labels_unavailable"}
            else:
                membership_labels[identity.name] = labels
    exports = {declaration.name: load_embedding_export(declaration.export_path)
               for declaration in declarations}
    common_users = sorted(set.intersection(*(
        set(export.arrays["user_id"].astype(str)) for export in exports.values())))
    provenance_by_user: dict[str, dict[str, float]] = {}
    for user in common_users:
        rows = np.flatnonzero(exports[declarations[0].name].arrays["user_id"].astype(str) == user)
        history = exports[declarations[0].name].arrays.get("history_event_count", np.zeros(len(rows)))
        history_value = float(np.max(history[rows])) if len(history) == len(exports[declarations[0].name].arrays["user_id"]) else 0.0
        provenance_by_user[user] = {
            "history_event_count": history_value,
            "cutoff_availability_count": float(len(rows)),
            # Service coverage is not embedded in the authenticated export. Keep
            # the public covariate explicit rather than deriving it from truth.
            "service_coverage_count": 0.0,
        }
    membership_populations: dict[str, Any] = {}
    boundaries = {item.name: item.bin_boundaries for item in config.matching_variables}
    for name, labels in membership_labels.items():
        eligible = {user: value for user, value in labels.items() if user in common_users}
        population = construct_privacy_population(
            exports[name], target_model_lineage=next(item.checkpoint_identity for item in authenticated.inputs if item.name == name),
            membership_by_user=eligible,
            provenance_by_user={user: provenance_by_user[user] for user in eligible},
            cutoff_order=config.cutoff_order, component_order=config.component_order,
            matching_boundaries=boundaries, audit_seed=config.audit_seed,
            split_seed=config.split_seed, split_fractions=(config.split.train_fraction,
                config.split.validation_fraction, config.split.test_fraction),
            minimum_total=config.support.minimum_total,
            minimum_per_class=config.support.minimum_per_class,
            minimum_per_stratum=config.support.minimum_per_stratum,
        )
        membership_populations[name] = population
        membership[name] = {"supported": population.status == "available", "status": population.status,
                            "reason": population.reason, "user_set_hashes": dict(population.user_set_hashes)}
        membership_metrics[name] = run_privacy_attacks_from_config(population, None, config)
    fractions = (config.split.train_fraction, config.split.validation_fraction,
                 config.split.test_fraction)
    lineage = authenticated.evidence_index_sha256
    probe_populations: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        probe_populations[declaration.name] = {}
        for component in config.component_order:
            probe_populations[declaration.name][component] = construct_sensitive_probe_population(
                exports[declaration.name], target_model_lineage=lineage,
                eligible_users=common_users, provenance_by_user=provenance_by_user,
                cutoff_order=config.cutoff_order, component_order=(component,),
                split_seed=config.split_seed, split_fractions=fractions,
            )
    reference_population = next(iter(next(iter(probe_populations.values())).values()))
    split_by_user = {record.user_id: record.split for record in reference_population.records}
    requested = tuple(item.name for item in config.sensitive_attributes)
    supported = tuple(name for name in requested if name in SUPPORTED_PROTECTED_ATTRIBUTES)
    bundle = load_protected_labels(
        authenticated, run.user_latents_truth, split_by_user=split_by_user,
        attributes=supported or ("age_group",), minimum_total=config.support.minimum_total,
        minimum_per_class=config.support.minimum_per_class,
        minimum_cell_support=config.support.minimum_sensitive_label_cell,
    )
    summaries = {summary.name: asdict(summary) for summary in bundle.summaries}
    sensitive: dict[str, Any] = {}
    sensitive_metrics: dict[str, Any] = {}
    for item in config.sensitive_attributes:
        if item.name not in supported:
            summary = {"status": "unavailable", "reason": "unsupported_protected_attribute",
                       "derivation_version": item.derivation.version, "eligible_count": len(common_users),
                       "missing_count": 0, "unsupported_count": len(common_users)}
            sensitive[item.name] = summary; sensitive_metrics[item.name] = summary
            continue
        sensitive[item.name] = summaries[item.name]
        sensitive_metrics[item.name] = {
            declaration.name: {
                component: run_protected_attribute_attacks(population, bundle, item.name, config)
                for component, population in probe_populations[declaration.name].items()
            } for declaration in declarations
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
        "splits": {"membership": ({"status": "available", "user_grouped": True,
                                      "user_set_hashes": dict(next(iter(membership_populations.values())).user_set_hashes)}
                                     if membership_populations else
                                    {"status": "unavailable", "reason": "authenticated_training_membership_labels_unavailable"}),
                   "sensitive_probe": {"status": "available", "split_seed": config.split_seed,
                                       "user_grouped": True, "user_set_hashes": dict(reference_population.user_set_hashes)}},
        "membership_population": membership,
        "sensitive_attributes": sensitive,
        "attacks": {"status": "run_for_supported_sensitive_attributes", "configuration": config.attacks},
        "membership_metrics": membership_metrics,
        "sensitive_probe_metrics": sensitive_metrics,
        "utility_privacy_axes": utility_axes,
        "coverage": {"authenticated_controls": len(identities), "membership_evaluated_controls": sum(
                         result.get("status") == "available" for result in membership_metrics.values()),
                     "sensitive_probe_common_users": len(common_users),
                     "sensitive_probe_supported_attributes": len(supported)},
        "exclusions": ([{"reason": "authenticated_training_membership_labels_unavailable",
                         "count": sum(item["kind"] == "learned" for item in identities) - len(membership_labels)}]
                       if len(membership_labels) < sum(item["kind"] == "learned" for item in identities) else []),
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
