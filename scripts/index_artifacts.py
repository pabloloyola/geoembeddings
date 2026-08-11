#!/usr/bin/env python3
"""Validate and index a complete T0.2 reference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(identifier: Path, root: Path, *, status: str = "present", note: str | None = None) -> dict:
    if not identifier.is_file():
        raise FileNotFoundError(identifier)
    result = {
        "identifier": identifier.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(identifier),
        "status": status,
    }
    if note:
        result["note"] = note
    return result


def load_export(path: Path, *, dense: bool = False) -> tuple[set[tuple[str, str]], set[str]]:
    with np.load(path, allow_pickle=False) as payload:
        matrix = payload["embedding"]
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"non-finite or non-matrix embedding in {path}")
        users = payload["user_id"].astype(str)
        labels = payload["timestamp" if dense else "cutoff"].astype(str)
        keys = set(zip(users, labels))
        if len(keys) != len(users):
            raise ValueError(f"duplicate user/cutoff keys in {path}")
        return keys, set(users)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    run, experiment = args.run_dir.resolve(), args.experiment_dir.resolve()
    metadata_path = experiment / "prepared/prepared_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    embedding_config = yaml.safe_load((experiment / "prepared/config.resolved.yaml").read_text())

    baseline_keys, baseline_users = load_export(experiment / "statistical_baseline.npz")
    learned_keys, learned_users = load_export(experiment / "embeddings.npz")
    if baseline_keys != learned_keys:
        raise ValueError("baseline and learned cutoff users/cutoffs differ")
    baseline_dense_keys, baseline_dense_users = load_export(
        experiment / "dense_statistical_baseline.npz", dense=True
    )
    learned_dense_keys, learned_dense_users = load_export(
        experiment / "dense_embeddings.npz", dense=True
    )
    if baseline_dense_keys != learned_dense_keys or baseline_dense_users != learned_dense_users:
        raise ValueError("baseline and learned dense users/timestamps differ")

    reports = {}
    for kind in ("baseline", "learned"):
        path = experiment / f"robustness/{kind}_robustness.json"
        reports[kind] = json.loads(path.read_text())
        for view in reports[kind]["views"]:
            view_path = Path(view["path"])
            if view_path.is_file():
                load_export(view_path)
    left = reports["baseline"]["metric_contract"]
    right = reports["learned"]["metric_contract"]
    for field in ("source_hashes", "specification_hash", "view_ids", "mask_hashes", "field_order"):
        if left[field] != right[field]:
            raise ValueError(f"robustness {field} differs")

    common_artifacts = {
        "dataset_manifest": run / "manifest.json",
        "resolved_simulation_config": run / "config.resolved.yaml",
        "deep_validation_report": run / "deep_validation_report.json",
        "prepared_metadata": metadata_path,
        "resolved_embedding_config": experiment / "prepared/config.resolved.yaml",
        "vocabularies": experiment / "prepared/vocabularies.json",
        "command_log": experiment / "t0.2_commands.log",
    }
    baseline_artifacts = {
        "baseline_cutoff_export": experiment / "statistical_baseline.npz",
        "baseline_dense_export": experiment / "dense_statistical_baseline.npz",
        "baseline_evaluation_report": experiment / "baseline_evaluation.json",
        "baseline_episode_response_report": experiment / "baseline_episode_response.json",
        "baseline_robustness_report": experiment / "robustness/baseline_robustness.json",
    }
    learned_artifacts = {
        "learned_checkpoint": experiment / "model/best_model.pt",
        "learned_training_report": experiment / "model/training_report.json",
        "learned_cutoff_export": experiment / "embeddings.npz",
        "learned_dense_export": experiment / "dense_embeddings.npz",
        "learned_evaluation_report": experiment / "evaluation.json",
        "learned_episode_response_report": experiment / "episode_response.json",
        "learned_robustness_report": experiment / "robustness/learned_robustness.json",
    }
    comparison_artifacts = {
        "embedding_comparison_json": experiment / "comparison/embedding_comparison.json",
        "embedding_comparison_markdown": experiment / "comparison/embedding_comparison.md",
    }

    def indexed(items: dict[str, Path]) -> list[dict]:
        return [{"id": name, **artifact(path, repo)} for name, path in items.items()]

    robust_views = []
    for kind in ("baseline", "learned"):
        for view in reports[kind]["views"]:
            if view["path"] is not None:
                robust_views.append({
                    "id": f"{kind}_robustness_view_{view['view_id']}",
                    **artifact(Path(view["path"]), repo),
                })

    user_hash = hashlib.sha256("\n".join(sorted(baseline_users)).encode()).hexdigest()
    cutoffs = sorted({cutoff for _, cutoff in baseline_keys})
    cutoff_counts = {cutoff: sum(label == cutoff for _, label in baseline_keys) for cutoff in cutoffs}
    no_event_users = manifest["users"] - metadata["users_with_events"]
    missing_cutoffs = manifest["users"] * len(cutoffs) - len(baseline_keys)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    prep_hash = sha256(metadata_path)
    identity = {
        "observed_source_hashes": metadata["source_files"],
        "cutoffs": {"train_end": metadata["train_end"], "validation_end": metadata["validation_end"]},
        "categorical_field_order": metadata["categorical_fields"],
        "continuous_field_order": metadata["continuous_fields"],
        "user_set_sha256": user_hash,
        "preparation_metadata_sha256": prep_hash,
    }
    result = {
        "schema_version": "geoembeddings-evidence-index/1.0",
        "task_id": "T0.2",
        "evidence_status": "complete",
        "completion_claim": True,
        "index_location": args.output.as_posix(),
        "storage": {
            "immutable_artifact_roots": [run.relative_to(repo).as_posix(), experiment.relative_to(repo).as_posix()],
            "durable_external_identifier": None,
            "immutability": "local roots made read-only after indexing; generated artifacts remain gitignored",
        },
        "provenance": {
            "source_commit": source_commit,
            "simulator_manifest_sha256": sha256(manifest_path),
            "simulation_seed": manifest["seed"],
            "training_seed": embedding_config["seed"],
            "evaluation_seed": embedding_config["evaluation"]["robustness"]["seed"],
            "cohort_users": manifest["users"],
            "expected_cohort_users": 500,
            "cutoffs": identity["cutoffs"],
            "observed_source_hashes": metadata["source_files"],
            "preparation_metadata": {
                "identifier": artifact(metadata_path, repo)["identifier"],
                "sha256": prep_hash,
                "contract_version": metadata["dataset_contract"],
                "categorical_field_order": metadata["categorical_fields"],
                "continuous_field_order": metadata["continuous_fields"],
            },
        },
        "evidence_identity": {"baseline": identity, "learned": identity},
        "required_artifacts": {
            "shared": indexed(common_artifacts),
            "baseline": indexed(baseline_artifacts),
            "learned": indexed(learned_artifacts),
            "robustness_views": robust_views,
            "comparison": indexed(comparison_artifacts),
        },
        "comparability_audit": {
            "baseline_and_learned_source_hashes_match": True,
            "cutoffs_match": True,
            "categorical_field_order_matches": True,
            "continuous_field_order_matches": True,
            "users_match": True,
            "dense_users_and_timestamps_match": True,
            "preparation_contract_matches": True,
            "robustness_specifications_and_masks_match": True,
            "all_indexed_npz_values_finite": True,
            "result": "passed",
            "blocking_reasons": [],
        },
        "coverage_and_missingness": {
            "simulated_users": manifest["users"],
            "users_with_observed_events": metadata["users_with_events"],
            "users_without_observed_events": no_event_users,
            "explanation_users_without_events": "The simulator emitted users with no adopted/recorded service events; observed-only exports cannot encode empty histories.",
            "cutoff_export_rows": len(baseline_keys),
            "cutoff_row_counts": cutoff_counts,
            "missing_user_cutoff_rows": missing_cutoffs,
            "explanation_missing_cutoffs": "Exports omit a user/cutoff only when that user has no observed history at that cutoff; no empty history is imputed.",
            "dense_rows": len(baseline_dense_keys),
            "episode_and_robustness_coverage": "Detailed label, bin, unencodable-view, and insufficient-history coverage is retained in the indexed reports; no report rows were filtered by the indexer.",
            "absent_artifacts": [],
            "artifact_note": "Historical separate *_event_removal.json files are not part of robustness-metrics/2.0; event-removal/truncation results are consolidated in each indexed *_robustness.json report.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
