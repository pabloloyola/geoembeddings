from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import geoembeddings.privacy_audit as privacy_audit_module
from geoembeddings.contract import OBSERVED_FILES
from geoembeddings.io import sha256_file, write_json
from geoembeddings.privacy_audit import audit_privacy
from geoembeddings.representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION
from geoembeddings.user_roles import (PROTOCOL_SCHEMA, assign_users,
                                      assignment_hash, role_summary)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_export(path: Path, users: list[str], metadata: Path, source_hash: str,
                  model_variant: str) -> None:
    rows = [(user, cutoff) for user in users
            for cutoff in ("train_end", "validation_end", "test_end")]
    index = np.arange(len(rows), dtype=np.float64)
    combined = np.column_stack((index % 17, index % 11)) / 17
    persistent = combined + (np.asarray([int(user[1:]) % 3 for user, _ in rows])[:, None] / 10)
    context = combined[:, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, user_id=np.asarray([row[0] for row in rows]),
        cutoff=np.asarray([row[1] for row in rows]), embedding=combined,
        component_persistent=persistent, component_context=context,
        component_combined=combined, component_names=np.asarray(COMPONENT_NAMES),
        component_dimensions=np.asarray([2, 2, 2]),
        schema_version=np.asarray(EXPORT_SCHEMA_VERSION), model_variant=np.asarray(model_variant),
        categorical_fields=np.asarray(["service"]), continuous_fields=np.asarray(["hour"]),
        preparation_hash=np.asarray(sha256_file(metadata)),
        source_file_names=np.asarray(["observed_events.csv.gz"]),
        source_hashes=np.asarray([source_hash]), train_end=np.asarray("2026-01-01"),
        validation_end=np.asarray("2026-01-02"),
        export_cutoffs=np.asarray(["train_end", "validation_end", "test_end"]),
        compatibility=np.asarray("fixture"), history_event_count=np.full(len(rows), 20),
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    users = [f"u{i:03d}" for i in range(180)]
    run = tmp_path / "run"
    observed = run / "observed"
    observed.mkdir(parents=True)
    for filename in OBSERVED_FILES.values():
        pd.DataFrame({"fixture": []}).to_csv(observed / filename, index=False, compression="gzip")
    write_json({"dataset_contract": {"name": "geoembeddings-dataset", "version": "2.0"}},
               run / "manifest.json")
    source_hash = sha256_file(observed / "observed_events.csv.gz")

    protocol_config = {"schema_version": PROTOCOL_SCHEMA, "seed": 73, "fractions": {
        "target_train": .45, "target_validation": .10, "target_test": .45}}
    assignments = assign_users(users, protocol_config)
    protocol = {**protocol_config, "assignment_sha256": assignment_hash(assignments),
                "roles": role_summary(assignments)}
    truth = run / "truth"
    truth.mkdir()
    # Cycling values provide each probe split with all three train-fitted tertiles.
    pd.DataFrame({"user_id": users, "price_sensitivity": np.arange(len(users)) % 30}).to_csv(
        truth / "user_latents.csv.gz", index=False, compression="gzip")

    experiments: dict[str, Path] = {}
    utilities = tmp_path / "utility"
    utilities.mkdir()
    artifacts: dict[str, dict[str, object]] = {}
    for name, variant in (("statistical_baseline", "statistical_baseline"),
                          ("capacity_matched_single", "single_vector")):
        root = tmp_path / name
        experiments[name] = root
        metadata = root / "prepared/prepared_metadata.json"
        write_json({"preparation_schema_version": "geoembeddings-preparation/2.0",
                    "dataset_contract": "geoembeddings-dataset/2.0",
                    "train_end": "2026-01-01", "validation_end": "2026-01-02",
                    "categorical_fields": ["service"], "continuous_fields": ["hour"],
                    "source_files": {"observed_events.csv.gz": source_hash},
                    "user_role_protocol": protocol}, metadata)
        export = root / ("statistical_baseline.npz" if name == "statistical_baseline" else "embeddings.npz")
        _write_export(export, users, metadata, source_hash, variant)
        utility = utilities / f"{name}.json"
        write_json({"population_identity": {"users": users,
                                             "user_set_sha256": _canonical_hash(users)},
                    "utility_metrics": {"held_out_accuracy": .5}, "coverage": {"users": len(users)}}, utility)
        indexed = [metadata, export, utility]
        if name != "statistical_baseline":
            checkpoint = root / "model/best_model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": {"weight": torch.ones(2, 2)},
                        "model_variant": variant}, checkpoint)
            participation = root / "model/training_participation.json"
            write_json({"schema_version": "geoembeddings-training-participation/1.0",
                        "user_role_protocol": protocol,
                        "checkpoint_identity": {"sha256": sha256_file(checkpoint)},
                        "preparation_identity": {"prepared_metadata_sha256": sha256_file(metadata),
                                                 "observed_source_hashes": {"observed_events.csv.gz": source_hash}}},
                       participation)
            indexed.extend((checkpoint, participation))
        for path in indexed:
            artifacts[str(path.resolve())] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    keys = sorted((user, cutoff) for user in users
                  for cutoff in ("train_end", "validation_end", "test_end"))
    write_json({"schema_version": "geoembeddings-factorization-evidence-index/1.0",
                "task_id": "T2.7", "decision": "do not advance",
                "matched_identity": {"source_files": {"observed_events.csv.gz": source_hash},
                                     "export_keys_sha256": _canonical_hash(keys),
                                     "user_mask_sha256": _canonical_hash(users),
                                     "cutoffs": ["test_end", "train_end", "validation_end"]},
                "artifacts": artifacts}, evidence / "evidence_index.json")

    config = yaml.safe_load(Path("configs/privacy/diagnostic_v1.yaml").read_text())
    config["features"]["component_order"] = ["combined"]
    config["support"] = {"minimum_total": 30, "minimum_per_class": 8,
                         "minimum_per_stratum": 8, "minimum_sensitive_label_cell": 2}
    config["sensitive_attributes"] = [{"name": "price_sensitivity_group", "source": "evaluator_truth",
        "derivation": {"version": "fixture-fixed-edges/1.0", "method": "fixed_edges",
                       "source_field": "price_sensitivity", "bin_boundaries": [-1, 10, 20, 31],
                       "labels": ["low", "medium", "high"]}}]
    config["attacks"]["linear"]["inverse_regularization_strengths"] = [.1]
    config["attacks"]["nonlinear"].update({"hidden_units": [4], "epochs": 2, "tuning_budget": 1})
    config["bootstrap"].update({"replicates": 5, "seed": 91})
    config_path = tmp_path / "privacy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"run": run, "evidence": evidence, "utility": utilities, "config": config_path,
            "output": tmp_path / "output", **experiments}


def test_audit_privacy_orchestration_runs_authenticated_diagnostic_controls(
        tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    populations = []
    real_attack = privacy_audit_module.run_privacy_attacks_from_config

    def record_population(population, labels, config, **kwargs):
        if kwargs.get("task", "membership") == "membership":
            populations.append(population)
        return real_attack(population, labels, config, **kwargs)

    monkeypatch.setattr(privacy_audit_module, "run_privacy_attacks_from_config", record_population)
    result = audit_privacy(
        run_dir=fixture["run"],
        experiments={name: fixture[name] for name in ("statistical_baseline", "capacity_matched_single")},
        evidence_dir=fixture["evidence"], utility_report_dir=fixture["utility"],
        config_path=fixture["config"], output_dir=fixture["output"],
    )
    report = json.loads(Path(result["privacy_json"]).read_text())
    markdown = Path(result["privacy_markdown"]).read_text()
    rendered = json.loads(markdown.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    assert rendered == report

    membership = report["membership_metrics"]["capacity_matched_single"]
    assert membership["status"] == "available" and membership["attacks"]
    assert populations
    split_users = [{record.user_id for record in populations[0].records if record.split == split}
                   for split in ("train", "validation", "test")]
    assert all(split_users) and all(left.isdisjoint(right) for index, left in enumerate(split_users)
                                    for right in split_users[index + 1:])
    sensitive = report["sensitive_probe_metrics"]["price_sensitivity_group"]
    assert sensitive["capacity_matched_single"]["combined"]["status"] == "available"
    split_hashes = report["splits"]["membership"]["user_set_hashes"]
    assert len({split_hashes[name] for name in ("train", "validation", "test")}) == 3
    for results in (membership, sensitive["capacity_matched_single"]["combined"]):
        boot = next(iter(results["attacks"].values()))[
            "roc_auc" if results.get("task") == "membership" else "macro_f1"]["bootstrap"]
        assert boot["interval"] is not None and boot["replicate_count"] == 5
    assert set(report["utility_privacy_axes"]) == {"statistical_baseline", "capacity_matched_single"}
    assert all(role == "diagnostic_control" for role in report["selection"]["roles"].values())
    assert report["selection"]["selection_dependent_privacy_conclusion"] == {
        "status": "unavailable", "reason": "no_selected_candidate"}
    serialized = json.dumps(report)
    assert "aggregate_winner" not in serialized and "protected_labels" not in serialized
    assert not any(user in serialized for user in ("u000", "u179"))
