#!/usr/bin/env python3
"""Run the immutable slow-fast v1 control stages with durable stage status.

This runner intentionally stops after control training.  It prepares the fresh
control experiment, authenticates the frozen preflight contract, trains for the
configured budget, and validates the final checkpoint.  Exports and evaluation
are separate commands so a stopped training run cannot accidentally feed them a
partial checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE_COMMIT = "1ceeb1ffb18ff6fa35d9ffd804a6ed6c9a160304"
FROZEN_MANIFEST_SHA256 = "e5f4b29a180b6440fedbd595e6cedba4b935afa1948cc1c04f954b5986540a3c"
CANDIDATE_CONFIG_SHA256 = "0063f9604beb5587fd8d2b6de5402012a9b480a3e03dd2d4aeb6d8c4f0c14ccf"
CONTROL_CONFIG_SHA256 = "5772f46b0d0adff13576f0692e4ad47cdda829cf510122864827a9c91881d2d0"
OBSERVED_FILES = ("observed_events.csv.gz", "users_observed.csv.gz")
FROZEN_INPUTS = (
    "src/geoembeddings/data.py",
    "src/geoembeddings/export.py",
    "src/geoembeddings/model.py",
    "src/geoembeddings/prepare.py",
    "src/geoembeddings/slow_fast_preflight.py",
    "src/geoembeddings/training.py",
    "configs/embedding/slow_fast_v1.yaml",
    "configs/embedding/slow_fast_capacity_matched_single.yaml",
    "scripts/slow_fast_v1_experiment.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _git_is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _assert_frozen_inputs_unchanged() -> None:
    for arguments in (
        ["git", "diff", "--quiet", FROZEN_SOURCE_COMMIT, "HEAD", "--", *FROZEN_INPUTS],
        ["git", "diff", "--quiet", "--", *FROZEN_INPUTS],
    ):
        if subprocess.run(arguments, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise ValueError("frozen slow-fast model/config/evaluator inputs have drifted")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")


def _source_hashes(run_dir: Path) -> dict[str, str]:
    observed = run_dir / "observed"
    return {name: _sha256(observed / name) for name in OBSERVED_FILES}


def _assert_candidate(candidate_dir: Path, candidate_config: Path, manifest: Path, run_dir: Path) -> dict[str, Any]:
    prepared = candidate_dir / "prepared"
    checkpoint = candidate_dir / "model" / "best_model.pt"
    report_path = candidate_dir / "model" / "training_report.json"
    marker_path = candidate_dir / "training.complete"
    for path in (prepared / "prepared_metadata.json", checkpoint, report_path, marker_path):
        if not path.is_file():
            raise ValueError(f"candidate artifact missing: {path}")
    config = _load_yaml(candidate_config)
    report = _read_json(report_path)
    marker = _read_json(marker_path)
    metadata = _read_json(prepared / "prepared_metadata.json")
    if config.get("model", {}).get("variant") != "slow_fast_v1":
        raise ValueError("candidate config is not slow_fast_v1")
    if report.get("model_variant") != "slow_fast_v1" or report.get("seed") != config.get("seed"):
        raise ValueError("candidate model or seed lineage mismatch")
    if len(report.get("history", [])) != int(config["training"]["epochs"]):
        raise ValueError("candidate did not complete its configured epoch budget")
    if report.get("configuration_sha256") != _canonical_hash(config):
        raise ValueError("candidate resolved configuration hash mismatch")
    preparation = report.get("preparation_identity", {})
    if preparation.get("source_files") != metadata.get("source_files"):
        raise ValueError("candidate prepared source lineage mismatch")
    if preparation.get("source_files") != _source_hashes(run_dir):
        raise ValueError("candidate/run observed source hashes differ")
    if report.get("runtime_metadata", {}).get("source_commit") != FROZEN_SOURCE_COMMIT:
        raise ValueError("candidate runtime source commit is not the frozen slow-fast commit")
    checkpoint_hash = _sha256(checkpoint)
    state = _load_checkpoint(checkpoint)
    if int(state.get("epoch", 0)) != int(config["training"]["epochs"]):
        raise ValueError("candidate checkpoint is not an epoch-8 final checkpoint")
    if report.get("artifact_lineage", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("candidate report/checkpoint hash mismatch")
    if marker.get("checkpoint_sha256") != checkpoint_hash or marker.get("checkpoint_epoch") != int(state["epoch"]):
        raise ValueError("candidate completion marker/checkpoint mismatch")
    return {
        "model_variant": state.get("model_variant"),
        "seed": state.get("seed"),
        "checkpoint_sha256": checkpoint_hash,
        "prepared_metadata_sha256": _sha256(prepared / "prepared_metadata.json"),
        "source_files": dict(metadata.get("source_files", {})),
        "parameter_counts": state.get("parameter_counts"),
        "runtime_source_commit": report.get("runtime_metadata", {}).get("source_commit"),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


class StageRunner:
    """Run commands after creating every directory used by their redirections."""

    def __init__(self, retry_dir: Path):
        self.retry_dir = retry_dir
        self.active: subprocess.Popen[bytes] | None = None

    def _status_paths(self, status_dir: Path, stage: str) -> tuple[Path, Path]:
        return status_dir / f"{stage}.status.json", status_dir / f"{stage}.exit_status"

    def run(self, stage: str, command: list[str], stage_dir: Path, artifacts: Iterable[Path] = ()) -> int:
        log_dir = stage_dir / "logs"
        status_dir = stage_dir / "status"
        # This is deliberately before opening stdout/stderr.  The retry1 bug
        # opened a path under a derived stage name whose directory did not exist.
        for directory in (stage_dir, log_dir, status_dir, *artifacts):
            directory.mkdir(parents=True, exist_ok=True)
        status_path, exit_path = self._status_paths(status_dir, stage)
        started = _now()
        _atomic_json(status_path, {"stage": stage, "state": "running", "started_at": started, "command": command})
        stdout_path = log_dir / f"{stage}.stdout.log"
        stderr_path = log_dir / f"{stage}.stderr.log"
        process: subprocess.Popen[bytes] | None = None
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=_environment(), stdout=stdout, stderr=stderr)
                self.active = process
                return_code = process.wait()
        except OSError:
            return_code = 127
        finally:
            self.active = None
        _atomic_json(status_path, {
            "stage": stage,
            "state": "passed" if return_code == 0 else "failed",
            "started_at": started,
            "finished_at": _now(),
            "exit_status": return_code,
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
            "command": command,
        })
        exit_path.write_text(f"{return_code}\n", encoding="utf-8")
        if return_code:
            raise RuntimeError(f"{stage} failed with exit status {return_code}; see {stderr_path}")
        return return_code


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_path = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path if not existing else f"{source_path}{os.pathsep}{existing}"
    environment.setdefault("UV_CACHE_DIR", "/tmp/geoembed-uv-cache")
    return environment


def _validate_final_control(control_dir: Path, control_config: Path, manifest: Path, run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    prepared = control_dir / "prepared"
    model = control_dir / "model"
    checkpoint = model / "best_model.pt"
    report_path = model / "training_report.json"
    participation = model / "training_participation.json"
    for path in (prepared / "prepared_metadata.json", checkpoint, report_path, participation):
        if not path.is_file():
            raise ValueError(f"control final artifact missing: {path}")
    config = _load_yaml(control_config)
    report = _read_json(report_path)
    metadata = _read_json(prepared / "prepared_metadata.json")
    state = _load_checkpoint(checkpoint)
    expected_epochs = int(config["training"]["epochs"])
    if config.get("model", {}).get("variant") != "slow_fast_capacity_matched_single":
        raise ValueError("control config is not the capacity-matched slow-fast control")
    if state.get("model_variant") != config["model"]["variant"] or report.get("model_variant") != state.get("model_variant"):
        raise ValueError("control model variant lineage mismatch")
    if int(state.get("seed", -1)) != int(config["seed"]) or report.get("seed") != config["seed"]:
        raise ValueError("control seed lineage mismatch")
    if len(report.get("history", [])) != expected_epochs or int(state.get("epoch", 0)) != expected_epochs:
        raise ValueError("control did not produce an epoch-8 final checkpoint")
    if report.get("configuration_sha256") != _canonical_hash(config):
        raise ValueError("control resolved configuration hash mismatch")
    source_files = dict(metadata.get("source_files", {}))
    if source_files != _source_hashes(run_dir) or source_files != candidate["source_files"]:
        raise ValueError("control observed source hashes do not match candidate")
    prepared_hash = _sha256(prepared / "prepared_metadata.json")
    if prepared_hash != candidate["prepared_metadata_sha256"] or report.get("preparation_identity", {}).get("prepared_metadata_sha256") != prepared_hash:
        raise ValueError("control preparation hash does not match candidate")
    if report.get("preparation_identity", {}).get("source_files") != source_files:
        raise ValueError("control report source lineage mismatch")
    checkpoint_hash = _sha256(checkpoint)
    if report.get("artifact_lineage", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("control report/checkpoint hash mismatch")
    marker = {
        "schema_version": "slow-fast-training-complete/1.0",
        "variant": "control",
        "model_variant": state["model_variant"],
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "execution_commit": _git_commit(),
        "config_raw_sha256": _sha256(control_config),
        "prepared_metadata_sha256": prepared_hash,
        "manifest_sha256": _sha256(manifest),
        "observed_source_hashes": source_files,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(state["epoch"]),
        "parameter_counts": state.get("parameter_counts"),
        "checkpoint_rule": "lowest validation loss after configured final epoch",
    }
    _atomic_json(control_dir / "training.complete", marker)
    return marker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, default=ROOT / "configs/embedding/slow_fast_v1.yaml")
    parser.add_argument("--control-config", type=Path, default=ROOT / "configs/embedding/slow_fast_capacity_matched_single.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/multihorizon-profile-s20260817/recoverable_two_state_benchmark_v4/benchmark_freeze_7d5d1d6/benchmark_freeze_manifest.json")
    parser.add_argument("--frozen-source-commit", default=FROZEN_SOURCE_COMMIT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    control_dir = args.control_dir.resolve()
    retry_dir = control_dir.parent
    if control_dir.exists():
        raise FileExistsError(f"immutable control target already exists: {control_dir}")
    if args.frozen_source_commit != FROZEN_SOURCE_COMMIT:
        raise ValueError("unexpected frozen source commit")
    if not _git_is_ancestor(FROZEN_SOURCE_COMMIT):
        raise ValueError("current execution commit does not descend from frozen slow-fast source")
    _assert_frozen_inputs_unchanged()
    manifest = args.manifest.resolve()
    candidate_config = args.candidate_config.resolve()
    control_config = args.control_config.resolve()
    _assert_file_hash(manifest, FROZEN_MANIFEST_SHA256, "frozen manifest")
    _assert_file_hash(candidate_config, CANDIDATE_CONFIG_SHA256, "candidate config")
    _assert_file_hash(control_config, CONTROL_CONFIG_SHA256, "control config")
    if not run_dir.joinpath("manifest.json").is_file():
        raise FileNotFoundError(f"missing run manifest: {run_dir / 'manifest.json'}")
    candidate = _assert_candidate(candidate_dir, candidate_config, manifest, run_dir)
    observed_manifest = _read_json(run_dir / "manifest.json")
    if int(observed_manifest.get("seed", -1)) != int(candidate["seed"]):
        raise ValueError("run/candidate seed mismatch")
    retry_dir.mkdir(parents=True, exist_ok=False)
    (retry_dir / "status").mkdir()
    lineage = {
        "schema_version": "slow-fast-control-retry-lineage/1.0",
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "execution_commit": _git_commit(),
        "manifest_sha256": _sha256(manifest),
        "candidate_config_sha256": _sha256(candidate_config),
        "control_config_sha256": _sha256(control_config),
        "observed_source_hashes": _source_hashes(run_dir),
        "run_seed": observed_manifest.get("seed"),
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_prepared_metadata_sha256": candidate["prepared_metadata_sha256"],
        "stages": ["control_prepare", "preflight", "control_train", "control_validate"],
        "exports_started": False,
        "evaluation_started": False,
    }
    _atomic_json(retry_dir / "lineage.json", lineage)
    runner = StageRunner(retry_dir)
    runner_status = retry_dir / "status" / "runner.status.json"
    _atomic_json(runner_status, {"state": "running", "started_at": _now(), "lineage": str((retry_dir / "lineage.json").resolve())})
    try:
        runner.run(
            "control_prepare",
            ["uv", "run", "--offline", "geoembed", "prepare", "--run-dir", str(run_dir), "--experiment-dir", str(control_dir), "--config", str(control_config)],
            control_dir,
            (control_dir / "prepared",),
        )
        preflight_code = (
            "import sys,yaml; "
            "from geoembeddings.slow_fast_preflight import run_slow_fast_preflight; "
            "run_slow_fast_preflight(sys.argv[1],sys.argv[2],yaml.safe_load(open(sys.argv[3])),"
            "yaml.safe_load(open(sys.argv[4])),sys.argv[5],sys.argv[6])"
        )
        runner.run(
            "preflight",
            [sys.executable, "-c", preflight_code, str(run_dir), str(candidate_dir / "prepared"), str(candidate_config), str(control_config), str(manifest), str(retry_dir / "preflight")],
            retry_dir / "preflight",
            (retry_dir / "preflight",),
        )
        runner.run(
            "control_train",
            ["uv", "run", "--offline", "geoembed", "train", "--run-dir", str(run_dir), "--experiment-dir", str(control_dir), "--config", str(control_config)],
            control_dir,
            (control_dir / "model",),
        )
        runner.run(
            "control_validate",
            [
                sys.executable, "-c",
                "from scripts.slow_fast_v1_control_runner import validate_control_from_cli; validate_control_from_cli()",
                "--run-dir", str(run_dir), "--candidate-dir", str(candidate_dir),
                "--control-dir", str(control_dir), "--candidate-config", str(candidate_config),
                "--control-config", str(control_config), "--manifest", str(manifest),
            ],
            control_dir,
            (control_dir / "model",),
        )
        _atomic_json(runner_status, {"state": "passed", "finished_at": _now(), "exit_status": 0, "lineage": str((retry_dir / "lineage.json").resolve())})
        (retry_dir / "COMPLETE").write_text("CONTROL_TRAINING_COMPLETE\n", encoding="utf-8")
        return 0
    except BaseException as error:
        _atomic_json(runner_status, {"state": "failed", "finished_at": _now(), "exit_status": 1, "error": str(error), "lineage": str((retry_dir / "lineage.json").resolve())})
        raise


def validate_control_from_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    candidate = _assert_candidate(args.candidate_dir, args.candidate_config, args.manifest, args.run_dir)
    marker = _validate_final_control(args.control_dir, args.control_config, args.manifest, args.run_dir, candidate)
    print(json.dumps({"status": "passed", "checkpoint_epoch": marker["checkpoint_epoch"], "checkpoint_sha256": marker["checkpoint_sha256"]}))


if __name__ == "__main__":
    raise SystemExit(main())
