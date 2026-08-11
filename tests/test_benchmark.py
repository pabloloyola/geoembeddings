from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoembeddings.benchmark import latency_statistics, run_offline_benchmark, validate_offline_benchmark
from test_reliability import _inputs


def test_latency_statistics_are_finite():
    result = latency_statistics([.1, .2, .3])
    assert result["iterations"] == 3 and result["p95_seconds"] >= result["p50_seconds"]
    with pytest.raises(ValueError): latency_statistics([])


def test_cpu_benchmark_schema_statistics_and_protection(tmp_path):
    observed, prepared, export = _inputs(tmp_path); output = tmp_path/"experiment"/"benchmarks"/"offline.json"
    report = run_offline_benchmark(observed, prepared, {"learned": export,
        "baseline": Path(tmp_path/"missing.npz")}, output, {"seed": 3}, warmup=0, iterations=2)
    assert report["schema_version"] == "geoembeddings-offline-benchmark/1.0"
    assert report["runtime_metadata"]["device_type"] == "cpu"
    assert report["representations"]["learned"]["reliability_evaluation"]["iterations"] == 2
    assert report["representations"]["baseline"]["status"] == "artifact_missing"
    assert "truth/" in report["information_boundary"]
    validate_offline_benchmark(report)
    report["representations"]["learned"]["bad"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"): validate_offline_benchmark(report)
    with pytest.raises(FileExistsError):
        run_offline_benchmark(observed, prepared, {"learned": export}, output, {}, warmup=0, iterations=1)


def test_benchmark_rejects_mismatched_representation_population(tmp_path):
    observed, prepared, export = _inputs(tmp_path); other = tmp_path/"other.npz"
    np.savez_compressed(other, user_id=np.array(["different", "different"]),
        cutoff=np.array(["train", "test"]), embedding=np.ones((2, 2)))
    with pytest.raises(ValueError, match="mismatched user/cutoff"):
        run_offline_benchmark(observed, prepared, {"baseline": export, "learned": other},
            tmp_path/"offline.json", {}, warmup=0, iterations=1)
