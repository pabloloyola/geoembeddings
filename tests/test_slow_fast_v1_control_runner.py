from __future__ import annotations

import json
import sys

import pytest

from scripts.slow_fast_v1_control_runner import StageRunner


def test_stage_runner_creates_stage_dirs_before_redirect_and_records_status(tmp_path) -> None:
    stage_dir = tmp_path / "candidate"
    artifact_dir = stage_dir / "dense_artifacts"
    runner = StageRunner(tmp_path / "retry")

    runner.run(
        "candidate_dense_export",
        [sys.executable, "-c", "print('dense export')"],
        stage_dir,
        (artifact_dir,),
    )

    assert (stage_dir / "logs/candidate_dense_export.stdout.log").read_text().strip() == "dense export"
    assert (stage_dir / "logs/candidate_dense_export.stderr.log").read_text() == ""
    status = json.loads((stage_dir / "status/candidate_dense_export.status.json").read_text())
    assert status["state"] == "passed"
    assert status["exit_status"] == 0
    assert (stage_dir / "status/candidate_dense_export.exit_status").read_text() == "0\n"
    assert artifact_dir.is_dir()


def test_stage_runner_records_failed_stage_status(tmp_path) -> None:
    stage_dir = tmp_path / "control"
    runner = StageRunner(tmp_path / "retry")

    with pytest.raises(RuntimeError, match="control_train failed"):
        runner.run(
            "control_train",
            [sys.executable, "-c", "raise SystemExit(7)"],
            stage_dir,
            (stage_dir / "model",),
        )

    status = json.loads((stage_dir / "status/control_train.status.json").read_text())
    assert status["state"] == "failed"
    assert status["exit_status"] == 7
    assert (stage_dir / "status/control_train.exit_status").read_text() == "7\n"
