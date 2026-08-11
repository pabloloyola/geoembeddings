"""Observed-only offline artifact/evaluator benchmark (R13)."""

from __future__ import annotations

import json
import io
import platform
import resource
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .contract import OFFLINE_BENCHMARK_SCHEMA
from .io import sha256_file, write_json
from .reliability import load_reliability_inputs, resampling_statistics, validate_preparation_config
from .runtime_metadata import collect_runtime_metadata


def latency_statistics(samples: list[float]) -> dict[str, float | int]:
    if not samples or not np.isfinite(samples).all() or min(samples) < 0:
        raise ValueError("latency samples must be non-empty, finite, and non-negative")
    ordered = np.sort(samples)
    return {"iterations": len(samples), "mean_seconds": float(np.mean(ordered)),
            "p50_seconds": float(np.quantile(ordered, .5)), "p95_seconds": float(np.quantile(ordered, .95)),
            "minimum_seconds": float(ordered[0]), "maximum_seconds": float(ordered[-1])}


def validate_offline_benchmark(report: dict[str, Any]) -> None:
    required = {"schema_version", "runtime_metadata", "device_software", "source_hashes",
                "preparation_identity", "configuration", "representations", "information_boundary"}
    if report.get("schema_version") != OFFLINE_BENCHMARK_SCHEMA or not required.issubset(report):
        raise ValueError("Invalid offline benchmark schema")
    def walk(value: Any) -> None:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Offline benchmark contains a non-finite statistic")
        if isinstance(value, dict):
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(report)


def _measure(operation: Callable[[], int], *, warmup: int, iterations: int) -> dict[str, Any]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    for _ in range(warmup): operation()
    tracemalloc.start(); before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timings, work = [], 0
    for _ in range(iterations):
        started = time.perf_counter(); work = operation(); timings.append(time.perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = latency_statistics(timings)
    stats.update({"warmup_iterations": warmup, "work_items_per_iteration": work,
                  "throughput_items_per_second": float(work / stats["mean_seconds"]),
                  "python_peak_allocated_bytes": int(peak),
                  "process_peak_rss_bytes": int(after_rss * (1024 if platform.system() != "Darwin" else 1)),
                  "process_peak_rss_delta_bytes": int(max(0, after_rss-before_rss) * (1024 if platform.system() != "Darwin" else 1))})
    return stats


def run_offline_benchmark(observed_dir: str | Path, prepared_dir: str | Path,
                          artifacts: dict[str, Path], output_path: str | Path,
                          config: dict[str, Any], *, warmup: int = 1, iterations: int = 5,
                          overwrite: bool = False) -> dict[str, Any]:
    started = time.perf_counter(); output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_path}")
    metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")); results = {}
    validate_preparation_config(metadata, config)
    reference_keys: set[str] | None = None
    for kind, artifact in artifacts.items():
        if not artifact.is_file():
            results[kind] = {"status": "artifact_missing", "artifact": str(artifact)}; continue
        _, users, cutoffs, values = load_reliability_inputs(observed_dir, prepared_dir, artifact)
        keys = set(np.char.add(np.char.add(users, "\0"), cutoffs))
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Baseline and learned benchmark exports have mismatched user/cutoff identity")
        reference_keys = keys
        def load_operation(path: Path = artifact) -> int:
            with np.load(path, allow_pickle=False) as payload:
                loaded = np.asarray(payload["embedding"])
                if not np.isfinite(loaded).all(): raise ValueError("non-finite benchmark input")
                return len(loaded)
        def export_operation(v: np.ndarray = values, u: np.ndarray = users,
                             c: np.ndarray = cutoffs) -> int:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, user_id=u, cutoff=c, embedding=v)
            return len(v)
        def evaluation_operation(v: np.ndarray = values, u: np.ndarray = users) -> int:
            count = 0
            for user in sorted(set(u)):
                selected = v[u == user]
                if len(selected) >= 2:
                    resampling_statistics(selected, seed=20260811 + count, resamples=25); count += 1
            return count
        results[kind] = {"status": "ok", "artifact": str(artifact), "artifact_size_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact), "workload": {"rows": len(users), "users": len(set(users)),
                "cutoffs": sorted(set(cutoffs)), "embedding_dimension": int(values.shape[1])},
            "export_artifact_read_validation": _measure(load_operation, warmup=warmup, iterations=iterations),
            "offline_export_serialization": _measure(export_operation, warmup=warmup, iterations=iterations),
            "reliability_evaluation": _measure(evaluation_operation, warmup=warmup, iterations=iterations)}
    seed = int(config.get("evaluation", {}).get("reliability", {}).get("seed", config.get("seed", 0)))
    report = {"schema_version": OFFLINE_BENCHMARK_SCHEMA, "benchmark_kind": "offline_frozen_export_and_evaluation",
        "runtime_metadata": collect_runtime_metadata(duration_seconds=time.perf_counter()-started, seed=seed, device="cpu").to_dict(),
        "device_software": {"device": "cpu", "processor": platform.processor() or None,
            "python_implementation": platform.python_implementation(), "numpy_version": np.__version__,
            "pytorch_version": torch.__version__, "torch_threads": torch.get_num_threads()},
        "source_hashes": metadata["source_files"], "preparation_identity": {"prepared_metadata_sha256": sha256_file(metadata_path),
            "train_end": metadata["train_end"], "validation_end": metadata["validation_end"]},
        "configuration": {"warmup_iterations": warmup, "measured_iterations": iterations},
        "representations": results,
        "information_boundary": "Benchmark reads observed source files, preparation metadata, and frozen representation artifacts only; truth/ is neither accepted nor opened.",
        "limitations": "Hardware-specific offline frozen-export read/evaluation timing; it does not measure training or online incremental updates."}
    validate_offline_benchmark(report); write_json(report, output_path); return report
