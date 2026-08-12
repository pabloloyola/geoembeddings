"""Observed-only offline artifact/evaluator benchmark (R13)."""

from __future__ import annotations

import json
import hashlib
import io
import platform
import resource
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .contract import OFFLINE_BENCHMARK_SCHEMA, ONLINE_BENCHMARK_SCHEMA, ONLINE_WORKLOAD_SCHEMA
from .io import read_json, sha256_file, write_json
from .reliability import load_reliability_inputs, resampling_statistics, validate_preparation_config
from .runtime_metadata import collect_runtime_metadata
from .online import (AtomicOnlineState, baseline_computer, canonical_hash,
                     event_fingerprint, learned_computer)
from .schema import load_observed


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


def freeze_online_workload(observed_dir: str | Path, prepared_dir: str | Path,
                           checkpoint_path: str | Path, output_path: str | Path, *,
                           seed: int = 20260812, maximum_users: int = 128) -> dict[str, Any]:
    """Freeze deterministic event selection; an existing manifest is immutable."""
    _, events = load_observed(observed_dir); events = events.copy()
    metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    field_order = list(events.columns)
    rows: dict[str, dict[str, Any]] = {}
    by_user: dict[str, list[str]] = {}
    for record in events.to_dict("records"):
        record["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat()
        fingerprint = event_fingerprint(record, field_order)
        rows[fingerprint] = {key: (None if pd.isna(value) else value) for key, value in record.items()}
        by_user.setdefault(str(record["user_id"]), []).append(fingerprint)
    users = sorted(by_user, key=lambda value: canonical_hash([seed, value]))[:maximum_users]
    cold = [[by_user[user][0]] for user in users if by_user[user]]
    update_pool = [fingerprint for user in users for fingerprint in by_user[user][32:64]]
    steady = [[fingerprint] for fingerprint in update_pool]
    workloads = {"cold_start": cold, "steady_single_event": steady}
    for size in (8, 32, 128):
        workloads[f"frozen_batch_{size}"] = [update_pool[offset:offset+size] for offset in range(0, len(update_pool), size) if len(update_pool[offset:offset+size]) == size]
    core = {"schema_version": ONLINE_WORKLOAD_SCHEMA, "derivation_version": "sha256-seeded-user-order/1.0",
            "seed": seed, "field_order": field_order, "selected_users": users, "events": rows,
            "user_event_fingerprints": {user: by_user[user] for user in users}, "prefill_count": 32,
            "workloads": workloads, "source_hashes": metadata["source_files"],
            "preparation_metadata_sha256": sha256_file(metadata_path),
            "checkpoint_sha256": sha256_file(checkpoint_path) if Path(checkpoint_path).is_file() else None}
    core["workload_sha256"] = canonical_hash(core)
    output_path = Path(output_path)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != core: raise FileExistsError("named online workload is immutable and its inputs changed")
        return existing
    write_json(core, output_path); return core


def validate_online_benchmark(report: dict[str, Any]) -> None:
    if report.get("schema_version") != ONLINE_BENCHMARK_SCHEMA:
        raise ValueError("Invalid online benchmark schema")
    if not {"runtime_metadata", "workload_manifest", "representations", "source_hashes"}.issubset(report):
        raise ValueError("Incomplete online benchmark")
    def walk(value: Any) -> None:
        if isinstance(value, float) and not np.isfinite(value): raise ValueError("online benchmark contains non-finite statistic")
        if isinstance(value, dict):
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(report)


def run_online_benchmark(observed_dir: str | Path, prepared_dir: str | Path,
                         checkpoint_path: str | Path, workload_path: str | Path,
                         output_path: str | Path, config: dict[str, Any], *, warmup: int = 10,
                         iterations: int = 100, overwrite: bool = False, device: str = "cpu") -> dict[str, Any]:
    """Benchmark transactional baseline and learned diagnostic controls."""
    if warmup < 0 or iterations < 1: raise ValueError("warmup must be non-negative and iterations positive")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite: raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_path}")
    workload = freeze_online_workload(observed_dir, prepared_dir, checkpoint_path, workload_path,
                                      seed=int(config.get("benchmark", {}).get("seed", 20260812)))
    metadata = read_json(Path(prepared_dir)/"prepared_metadata.json")
    maximum = int(config["data"]["max_sequence_length"])
    baseline, baseline_identity = baseline_computer(prepared_dir, config)
    learned, learned_names, learned_identity = learned_computer(prepared_dir, checkpoint_path, device)
    computers = {"baseline": (baseline, ("combined",), baseline_identity),
                 "learned": (learned, learned_names, learned_identity)}
    results: dict[str, Any] = {}
    for kind, (computer, names, identity) in computers.items():
        entries = []
        for workload_name, calls in workload["workloads"].items():
            if not calls:
                entries.append({"workload": workload_name, "status": "excluded", "exclusions": ["insufficient source events"]}); continue
            samples = []; accepted_total = checked = 0; max_abs = max_rel = 0.0
            def execute() -> tuple[int, int, float, float]:
                state = AtomicOnlineState(field_order=workload["field_order"], component_names=names,
                    compute=computer, maximum_history=maximum, identity=identity)
                # Frozen prefill is excluded from timing and applies only to update workloads.
                if workload_name != "cold_start":
                    update_users = {str(workload["events"][fp]["user_id"]) for call in calls for fp in call}
                    prefill = [workload["events"][fp] for user in sorted(update_users)
                               for fp in workload["user_event_fingerprints"][user][:32]]
                    if prefill: state.append(prefill, oracle=computer)
                accepted = rows_checked = 0; local_abs = local_rel = 0.0
                for call in calls:
                    update = state.append([workload["events"][fp] for fp in call], oracle=computer)
                    accepted += update.accepted_events; rows_checked += len(update.representations)
                    local_abs = max(local_abs, update.maximum_absolute_error); local_rel = max(local_rel, update.maximum_relative_error)
                return accepted, rows_checked, local_abs, local_rel
            for _ in range(warmup): execute()
            tracemalloc.start(); before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for _ in range(iterations):
                if device.startswith("cuda"): torch.cuda.synchronize()
                started = time.perf_counter(); accepted, row_count, absolute, relative = execute()
                if device.startswith("cuda"): torch.cuda.synchronize()
                samples.append(time.perf_counter()-started); accepted_total += accepted; checked += row_count
                max_abs = max(max_abs, absolute); max_rel = max(max_rel, relative)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            statistics = latency_statistics(samples); elapsed = sum(samples)
            entries.append({"workload": workload_name, "status": "ok", "requested_batch_size": int(workload_name.rsplit("_",1)[-1]) if workload_name.startswith("frozen_batch") else 1,
                "realized_calls_per_iteration": len(calls), "realized_events_per_iteration": sum(map(len, calls)),
                "warmup_iterations": warmup, "measured_iterations": iterations, "latency": statistics,
                "throughput": {"accepted_events_per_second": accepted_total/elapsed, "completed_calls_per_second": len(calls)*iterations/elapsed},
                "peak_memory": {"python_allocation_bytes": int(peak), "process_rss_bytes": int(after*(1024 if platform.system() != "Darwin" else 1)), "device_allocated_bytes": None, "device_exclusion": "unsupported for CPU" if device == "cpu" else "not sampled"},
                "oracle": {"status": "passed", "checked_rows": checked, "checked_components": list(names), "maximum_absolute_error": max_abs, "maximum_relative_error": max_rel}, "exclusions": []})
        # Serialization is measured independently from updates and round-tripped.
        export_state = AtomicOnlineState(field_order=workload["field_order"], component_names=names,
            compute=computer, maximum_history=maximum, identity=identity)
        export_rows = [workload["events"][fingerprints[0]] for fingerprints in
                       workload["user_event_fingerprints"].values() if fingerprints]
        if export_rows: export_state.append(export_rows, oracle=computer)
        arrays = {f"{user}__{name}": value for user, output in sorted(export_state.outputs.items())
                  for name, value in output.components}
        def serialize_memory() -> int:
            buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
            with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as loaded:
                if set(loaded.files) != set(arrays) or any(not np.isfinite(loaded[key]).all() for key in loaded.files):
                    raise ValueError("serialization round-trip mismatch")
            return len(buffer.getvalue())
        def serialize_file() -> int:
            with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
                np.savez_compressed(handle, **arrays); handle.flush()
                with np.load(handle.name, allow_pickle=False) as loaded:
                    if set(loaded.files) != set(arrays): raise ValueError("serialization schema mismatch")
                return Path(handle.name).stat().st_size
        memory = _measure(serialize_memory, warmup=warmup, iterations=iterations)
        file_result = _measure(serialize_file, warmup=warmup, iterations=iterations)
        probe = io.BytesIO(); np.savez_compressed(probe, **arrays)
        entries.extend([
            {"workload": "export_serialization_memory", "status": "ok", "payload_bytes": len(probe.getvalue()), "payload_sha256": hashlib.sha256(probe.getvalue()).hexdigest(), "round_trip": "passed", "latency": memory, "exclusions": []},
            {"workload": "export_serialization_temporary_file", "status": "ok", "artifact_bytes": file_result["work_items_per_iteration"], "round_trip": "passed", "latency": file_result, "exclusions": ["temporary file cleanup excluded"]},
        ])
        results[kind] = {"representation": kind, "selection_role": "diagnostic_control", "identity": identity, "component_schema": list(names), "workloads": entries}
    runtime = collect_runtime_metadata(duration_seconds=0.0, seed=workload["seed"], device=device).to_dict()
    report = {"schema_version": ONLINE_BENCHMARK_SCHEMA, "benchmark_kind": "atomic_incremental_update",
        "runtime_metadata": runtime, "device_software": {"device": device, "cpu_model": platform.processor() or None, "os": platform.platform(), "architecture": platform.machine(), "python": platform.python_version(), "numpy": np.__version__, "pytorch": torch.__version__, "torch_threads": torch.get_num_threads(), "synchronization": "torch.cuda.synchronize" if device.startswith("cuda") else "synchronous CPU"},
        "source_hashes": metadata["source_files"], "preparation_identity": {"metadata_sha256": sha256_file(Path(prepared_dir)/"prepared_metadata.json"), "categorical_fields": metadata["categorical_fields"], "continuous_fields": metadata["continuous_fields"]},
        "checkpoint_identity": {"sha256": sha256_file(checkpoint_path)}, "workload_manifest": {"path": str(Path(workload_path)), "sha256": workload["workload_sha256"], "schema_version": workload["schema_version"]},
        "configuration": {"warmup_iterations": warmup, "measured_iterations": iterations, "dtype": "float32", "device": device, "configuration_sha256": canonical_hash(config)},
        "representations": results, "information_boundary": "observed/, preparation metadata, and authenticated checkpoint only; truth/ is neither accepted nor opened.",
        "exclusions": ["training", "hardware-normalized comparison", "accelerator memory when unsupported"]}
    validate_online_benchmark(report); write_json(report, output_path); return report
