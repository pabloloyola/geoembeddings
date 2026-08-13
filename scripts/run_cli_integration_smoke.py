#!/usr/bin/env python3
"""Run the complete CLI contract on a bounded, disposable synthetic cohort.

This is an integration/CI smoke, not reference-scale scientific evidence.  It
intentionally uses the supported 50-user/seven-day smoke cohort, one CPU training epoch, and
one benchmark iteration.  Every artifact lives below a temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from geoembeddings.contract import OBSERVED_FILES, TRUTH_FILES
from geoembeddings.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _run(*arguments: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "geoembeddings", *arguments]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout[completed.stdout.index("{"):])
    except json.JSONDecodeError as error:
        raise AssertionError(f"command did not emit JSON: {' '.join(command)}\n{completed.stdout}") from error


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as artifact:
        assert artifact.files, path
        for name in artifact.files:
            values = artifact[name]
            if np.issubdtype(values.dtype, np.number):
                assert np.isfinite(values).all(), f"non-finite {path}:{name}"


def _write_configs(directory: Path) -> tuple[Path, Path]:
    embedding = yaml.safe_load((ROOT / "configs/embedding/single_vector.yaml").read_text())
    embedding["model"].update({
        "categorical_embedding_dim": 4, "event_dim": 8, "hidden_dim": 8,
        "user_embedding_dim": 8, "dropout": 0.0, "event_dropout": 0.0,
    })
    embedding["data"]["max_sequence_length"] = 16
    embedding["training"].update({"epochs": 1, "batch_size": 64, "device": "cpu"})
    embedding_path = directory / "embedding-smoke.yaml"
    embedding_path.write_text(yaml.safe_dump(embedding, sort_keys=False), encoding="utf-8")

    privacy = yaml.safe_load((ROOT / "configs/privacy/diagnostic_v1.yaml").read_text())
    # The cohort is deliberately too small for a scientific membership attack.
    # The audit must still authenticate inputs and emit a valid unavailable result.
    privacy["features"]["component_order"] = ["combined"]
    privacy["support"].update({"minimum_total": 100, "minimum_per_class": 20,
                               "minimum_per_stratum": 20})
    privacy["bootstrap"]["replicates"] = 2
    privacy_path = directory / "privacy-smoke.yaml"
    privacy_path.write_text(yaml.safe_dump(privacy, sort_keys=False), encoding="utf-8")
    return embedding_path, privacy_path


def _privacy_inputs(run: Path, experiment: Path, evidence: Path, utility: Path) -> None:
    """Index real smoke artifacts in the narrow format required by audit-privacy."""
    evidence.mkdir()
    utility.mkdir()
    metadata = experiment / "prepared/prepared_metadata.json"
    baseline = experiment / "statistical_baseline.npz"
    learned = experiment / "embeddings.npz"
    checkpoint = experiment / "model/best_model.pt"
    participation = experiment / "model/training_participation.json"
    baseline_eval = _load_json(experiment / "baseline_evaluation.json")
    learned_eval = _load_json(experiment / "evaluation.json")
    with np.load(learned, allow_pickle=False) as export:
        export_users = sorted(set(str(value) for value in export["user_id"].tolist()))
    utility_paths = {
        "statistical_baseline": utility / "statistical_baseline.json",
        "capacity_matched_single": utility / "capacity_matched_single.json",
    }
    for name, report in (("statistical_baseline", baseline_eval),
                         ("capacity_matched_single", learned_eval)):
        utility_paths[name].write_text(json.dumps({
            "population_identity": {"users": export_users,
                                    "user_set_sha256": _canonical_hash(export_users)},
            "utility_metrics": report.get("metrics", {}),
            "coverage": report.get("coverage", {}),
        }, indent=2, sort_keys=True), encoding="utf-8")

    artifacts: dict[str, dict[str, object]] = {}
    indexed = [metadata, baseline, learned, checkpoint, participation, *utility_paths.values()]
    for path in indexed:
        artifacts[str(path.resolve())] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    with np.load(learned, allow_pickle=False) as export:
        keys = sorted((str(user), str(cutoff)) for user, cutoff in
                      zip(export["user_id"].tolist(), export["cutoff"].tolist(), strict=True))
        users = sorted(set(str(value) for value in export["user_id"].tolist()))
        source_files = dict(zip(export["source_file_names"].tolist(),
                                export["source_hashes"].tolist(), strict=True))
    index = {
        "schema_version": "geoembeddings-factorization-evidence-index/1.0",
        # The authenticator requires the frozen T2.7 protocol labels. This local
        # index is deleted with the smoke root and is never an evidence artifact.
        "task_id": "T2.7", "decision": "do not advance",
        "smoke_only": True,
        "matched_identity": {"source_files": source_files,
                             "export_keys_sha256": _canonical_hash(keys),
                             "user_mask_sha256": _canonical_hash(users),
                             "cutoffs": sorted({cutoff for _, cutoff in keys})},
        "artifacts": artifacts,
    }
    (evidence / "evidence_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def run_smoke(workspace: Path) -> None:
    run, experiment = workspace / "run", workspace / "experiment"
    evidence, utility, audit = workspace / "evidence", workspace / "utility", workspace / "audit"
    embedding_config, privacy_config = _write_configs(workspace)
    common = ("--run-dir", str(run), "--experiment-dir", str(experiment),
              "--config", str(embedding_config))

    _run("simulate", "--run-dir", str(run), "--users", "50", "--days", "7",
         "--seed", "20260811")
    manifest = _load_json(run / "manifest.json")
    assert manifest["dataset_contract"] == {"name": "geoembeddings-dataset", "version": "2.0"}
    observed_hashes = {name: sha256_file(run / "observed" / name) for name in OBSERVED_FILES.values()}
    truth_hashes = {name: sha256_file(run / "truth" / name) for name in TRUTH_FILES.values()}

    _run("validate", "--run-dir", str(run))
    validation = _load_json(run / "deep_validation_report.json")
    assert validation["status"] in {"pass", "passed", "passed_with_warnings"}
    assert truth_hashes == {name: sha256_file(run / "truth" / name) for name in truth_hashes}

    _run("prepare", *common)
    metadata_path = experiment / "prepared/prepared_metadata.json"
    metadata = _load_json(metadata_path)
    assert metadata["source_files"]
    assert all(observed_hashes[name] == digest for name, digest in metadata["source_files"].items())
    assert metadata["categorical_fields"] and metadata["continuous_fields"]
    preparation_hash = sha256_file(metadata_path)

    _run("baseline", *common)
    _run("train", *common)
    _run("export", *common)
    _run("export-dense", *common, "--kind", "learned", "--event-stride", "4")
    for path in (experiment / "statistical_baseline.npz", experiment / "embeddings.npz",
                 experiment / "dense_embeddings.npz"):
        _assert_finite_npz(path)
        with np.load(path, allow_pickle=False) as artifact:
            assert str(artifact["preparation_hash"].item()) == preparation_hash
            export_hashes = dict(zip(artifact["source_file_names"].tolist(),
                                     artifact["source_hashes"].tolist(), strict=True))
            assert export_hashes
            assert all(observed_hashes[name] == digest for name, digest in export_hashes.items())

    _run("evaluate", *common, "--kind", "baseline")
    _run("evaluate", *common, "--kind", "learned")
    _run("compare", "--run-dir", str(run), "--experiment-dir", str(experiment),
         "--config", str(embedding_config))
    _run("rank", "--run-dir", str(run), "--experiment-dir", str(experiment),
         "--model", "nearest", "--k", "1", "5")
    _run("benchmark", *common, "--warmup", "0", "--iterations", "1")

    _privacy_inputs(run, experiment, evidence, utility)
    _run("audit-privacy", "--run-dir", str(run),
         "--experiment-dir", f"statistical_baseline={experiment}",
         "--experiment-dir", f"capacity_matched_single={experiment}",
         "--evidence-dir", str(evidence), "--utility-report-dir", str(utility),
         "--config", str(privacy_config), "--output-dir", str(audit))
    privacy = _load_json(audit / "audits/privacy.json")
    assert privacy["schema_version"] == "geoembeddings-privacy-audit/1.0"
    assert all(value["status"] == "unavailable" for value in privacy["membership_metrics"].values())

    canonical = [
        experiment / "model/best_model.pt", experiment / "model/training_report.json",
        experiment / "baseline_evaluation.json", experiment / "evaluation.json",
        experiment / "comparison/embedding_comparison.json", experiment / "ranking/nearest.json",
        experiment / "benchmarks/offline.json", experiment / "benchmarks/online.json",
        audit / "audits/privacy.md",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in canonical)
    assert observed_hashes == {name: sha256_file(run / "observed" / name) for name in observed_hashes}
    assert truth_hashes == {name: sha256_file(run / "truth" / name) for name in truth_hashes}

    # Every stage is immutable by default; a repeat must fail without --overwrite.
    ranking_hash = sha256_file(experiment / "ranking/nearest.json")
    immutable = subprocess.run([
        sys.executable, "-m", "geoembeddings", "rank", "--run-dir", str(run),
        "--experiment-dir", str(experiment), "--model", "nearest", "--k", "1", "5",
    ], cwd=ROOT, text=True, capture_output=True)
    assert immutable.returncode != 0
    assert sha256_file(experiment / "ranking/nearest.json") == ranking_hash
    # Modeling roots must not accept protected/direct sub-roots.
    boundary = subprocess.run([sys.executable, "-m", "geoembeddings", "prepare",
                               "--run-dir", str(run / "truth"), "--experiment-dir", str(workspace / "bad"),
                               "--config", str(embedding_config)], cwd=ROOT, text=True, capture_output=True)
    assert boundary.returncode != 0, "prepare unexpectedly accepted a protected truth sub-root"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="Keep artifacts here instead of using a temporary root")
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        run_smoke(args.work_dir.resolve())
        print(args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="geoembed-cli-smoke-") as temporary:
            run_smoke(Path(temporary))
            print("complete (temporary artifacts removed)")


if __name__ == "__main__":
    main()
