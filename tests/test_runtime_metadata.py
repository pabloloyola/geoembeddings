from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from geoembeddings.config import load_config
from geoembeddings.io import read_json
from geoembeddings.runtime_metadata import (
    RUNTIME_METADATA_SCHEMA_VERSION,
    RuntimeMetadata,
    collect_runtime_metadata,
)
from geoembeddings.training import train_model


def test_runtime_metadata_is_versioned_deterministic_and_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geoembeddings.runtime_metadata.version", lambda _: "9.8.7")
    monkeypatch.setattr("geoembeddings.runtime_metadata.platform.python_version", lambda: "3.12.1")
    monkeypatch.setattr("geoembeddings.runtime_metadata.platform.platform", lambda: "TestOS-1")
    monkeypatch.setattr("geoembeddings.runtime_metadata.torch.__version__", "2.9.0")
    monkeypatch.setattr("geoembeddings.runtime_metadata.subprocess.run", lambda *a, **k: type("R", (), {"stdout": "abc123\n"})())
    first = collect_runtime_metadata(duration_seconds=1.25, seed=42).to_dict()
    second = collect_runtime_metadata(duration_seconds=1.25, seed=42).to_dict()
    assert first == second
    assert first["schema_version"] == RUNTIME_METADATA_SCHEMA_VERSION
    assert first["package_version"] == "9.8.7" and first["source_commit"] == "abc123"
    assert first["seed"] == 42 and type(first["seed"]) is int
    assert first["device_type"] is None and first["accelerator"] is None
    assert json.loads(json.dumps(first)) == first


@pytest.mark.parametrize("duration", [-1.0, math.inf, -math.inf, math.nan])
def test_runtime_metadata_rejects_invalid_durations(duration: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RuntimeMetadata("v", "3", None, "2", "os", None, None, duration, 1)


def test_runtime_metadata_handles_unavailable_package_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError
    monkeypatch.setattr("geoembeddings.runtime_metadata.version", lambda _: (_ for _ in ()).throw(PackageNotFoundError()))
    monkeypatch.setattr("geoembeddings.runtime_metadata.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    metadata = collect_runtime_metadata(duration_seconds=0, seed=7, device="cpu").to_dict()
    assert metadata["package_version"] is None
    assert metadata["source_commit"] is None
    assert metadata["device_type"] == "cpu"
    assert metadata["accelerator"] is None


def test_cpu_training_report_contains_runtime_metadata_and_preserves_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/embedding/single_vector.yaml")
    config["training"].update({"device": "cpu", "epochs": 1, "batch_size": 256})
    config["model"].update({"categorical_embedding_dim": 4, "event_dim": 8,
                            "hidden_dim": 8, "user_embedding_dim": 8})
    report = train_model(root / "smoke/run/observed", root / "smoke/experiment/prepared",
                         tmp_path / "model", config)
    serialized = read_json(tmp_path / "model/training_report.json")
    assert serialized["runtime_metadata"]["device_type"] == "cpu"
    assert math.isfinite(serialized["runtime_metadata"]["wall_clock_duration_seconds"])
    assert serialized["runtime_metadata"]["seed"] == config["seed"]
    assert serialized["history"] == report["history"]
    assert serialized["categorical_fields"] == report["categorical_fields"]
    assert serialized["best_validation_loss"] == report["best_validation_loss"]
