"""Held-out-user, window-bootstrap uncertainty calibration for R10 controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .io import sha256_file, write_json
from .layout import DatasetLayout, ExperimentLayout
from .reliability import calibration_bins, coverage_risk_curve

SCHEMA = "geoembeddings-diagnostic-calibration/1.0"
ROLE = "diagnostic_control"
ALLOWED = {
    "statistical_baseline",
    "capacity_matched_single",
    "factorized_pc",
    "persistent_only",
    "context_only",
    "factorized_no_context_loss",
    "factorized_no_persistent_loss",
}


def _set_identity(users: list[str]) -> dict[str, Any]:
    values = sorted(set(users))
    return {
        "count": len(values),
        "users": values,
        "sha256": hashlib.sha256("\n".join(values).encode()).hexdigest(),
    }


def frozen_user_split(
    users: list[str], *, seed: int, calibration_fraction: float
) -> tuple[list[str], list[str]]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    ordered = sorted(
        set(users), key=lambda u: hashlib.sha256(f"{seed}\0{u}".encode()).digest()
    )
    edge = max(
        1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction)))
    )
    if len(ordered) < 2:
        raise ValueError("at least two common users are required")
    return sorted(ordered[:edge]), sorted(ordered[edge:])


def bootstrap_user(
    values: np.ndarray, *, seed: int, replicates: int, minimum_windows: int
) -> tuple[float, float] | None:
    """Bootstrap prefix windows and score their centroid against a later window."""
    if replicates < 2 or minimum_windows < 2:
        raise ValueError("replicates and minimum_windows must be at least two")
    if len(values) < minimum_windows + 1:
        return None
    history, target = values[:-1], values[-1]
    rng = np.random.default_rng(seed)
    means = history[rng.integers(0, len(history), (replicates, len(history)))].mean(
        axis=1
    )
    uncertainty = float(np.mean(np.sum((means - means.mean(axis=0)) ** 2, axis=1)))
    error = float(np.linalg.norm(history.mean(axis=0) - target))
    return uncertainty, error


def fit_affine(uncertainty: np.ndarray, error: np.ndarray) -> dict[str, float]:
    if len(uncertainty) < 2:
        raise ValueError("at least two calibration users are required")
    design = np.column_stack([uncertainty, np.ones(len(uncertainty))])
    slope, intercept = np.linalg.lstsq(design, error, rcond=None)[0]
    # A calibration uncertainty must remain monotone and non-negative.
    slope = max(0.0, float(slope))
    intercept = max(0.0, float(intercept))
    return {
        "method": "nonnegative_monotone_affine_least_squares",
        "slope": slope,
        "intercept": intercept,
    }


def _load_dense(
    name: str, root: Path, run: DatasetLayout
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    layout = ExperimentLayout.from_path(root)
    dense = (
        layout.dense_baseline_embeddings
        if name == "statistical_baseline"
        else layout.dense_embeddings
    )
    metadata = json.loads(layout.prepared_metadata.read_text())
    actual = {
        filename: sha256_file(run.observed / filename)
        for filename in metadata["source_files"]
    }
    if actual != metadata["source_files"]:
        raise ValueError(f"{name}: observed source authentication failed")
    with np.load(dense, allow_pickle=False) as payload:
        required = {
            "user_id",
            "timestamp",
            "embedding",
            "preparation_hash",
            "source_file_names",
            "source_hashes",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"{name}: dense export contract is incomplete")
        if str(payload["preparation_hash"].item()) != sha256_file(
            layout.prepared_metadata
        ):
            raise ValueError(f"{name}: preparation authentication failed")
        export_sources = dict(
            zip(
                payload["source_file_names"].astype(str),
                payload["source_hashes"].astype(str),
            )
        )
        if export_sources != actual:
            raise ValueError(f"{name}: export/source authentication failed")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    return {
        "experiment_dir": str(layout.root),
        "dense_export": str(dense),
        "dense_export_sha256": sha256_file(dense),
        "prepared_metadata_sha256": sha256_file(layout.prepared_metadata),
        "source_hashes": actual,
    }, arrays


def calibrate_reliability(
    run_dir: str | Path,
    experiments: Mapping[str, str | Path],
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if (
        set(experiments) - ALLOWED
        or "statistical_baseline" not in experiments
        or "capacity_matched_single" not in experiments
    ):
        raise ValueError(
            "controls must include statistical_baseline and capacity_matched_single and only eligible diagnostics"
        )
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"configuration schema must be {SCHEMA}")
    output = Path(output_dir) / "reliability" / "calibration.json"
    existing = json.loads(output.read_text()) if output.exists() else None
    if existing is not None and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite immutable calibration report: {output}"
        )
    run = DatasetLayout.from_path(run_dir)
    run.validate(require_truth=False)
    inputs, arrays = {}, {}
    for name, root in sorted(experiments.items()):
        inputs[name], arrays[name] = _load_dense(name, Path(root), run)
    source_identities = {json.dumps(value["source_hashes"], sort_keys=True) for value in inputs.values()}
    preparation_identities = {value["prepared_metadata_sha256"] for value in inputs.values()}
    if len(source_identities) != 1 or len(preparation_identities) != 1:
        raise ValueError("controls must share source and preparation identities")
    user_sets = [set(value["user_id"].astype(str)) for value in arrays.values()]
    common = sorted(set.intersection(*user_sets))
    if any(users != set(common) for users in user_sets):
        raise ValueError(
            "post-hoc user-set changes or unequal control coverage are forbidden"
        )
    split = config["split"]
    cal_users, test_users = frozen_user_split(
        common,
        seed=int(split["seed"]),
        calibration_fraction=float(split["calibration_fraction"]),
    )
    settings = config["bootstrap"]
    reports = {}
    for control_index, name in enumerate(sorted(arrays)):
        value = arrays[name]
        users = value["user_id"].astype(str)
        timestamps = value["timestamp"].astype(str)
        matrix = np.asarray(value["embedding"], dtype=float)
        stats, exclusions = {}, []
        for user_index, user in enumerate(common):
            selected = np.flatnonzero(users == user)
            selected = selected[np.argsort(timestamps[selected], kind="stable")]
            result = bootstrap_user(
                matrix[selected],
                seed=int(settings["seed"]) + control_index * 100000 + user_index,
                replicates=int(settings["replicates"]),
                minimum_windows=int(settings["minimum_windows"]),
            )
            if result is None:
                exclusions.append(
                    {
                        "user_id": user,
                        "reason": "insufficient_windows",
                        "window_count": len(selected),
                    }
                )
            else:
                stats[user] = result
        fit_users = [u for u in cal_users if u in stats]
        evaluated = [u for u in test_users if u in stats]
        params = fit_affine(
            np.array([stats[u][0] for u in fit_users]),
            np.array([stats[u][1] for u in fit_users]),
        )
        raw = np.array([stats[u][0] for u in evaluated])
        error = np.array([stats[u][1] for u in evaluated])
        calibrated = params["slope"] * raw + params["intercept"]
        bins = int(config["reporting"]["bins"])
        minimum = int(config["reporting"]["minimum_bin_count"])
        coverages = list(map(float, config["reporting"]["coverage_levels"]))
        reports[name] = {
            "role": ROLE,
            "identity": {**inputs[name], "name": name},
            "fitted_calibration_parameters": params,
            "coverage": {
                "eligible_users": len(common),
                "calibration_users_used": len(fit_users),
                "test_users_used": len(evaluated),
                "excluded_users": exclusions,
            },
            "raw": {
                "reliability_error_bins": calibration_bins(
                    raw, error, bins=bins, minimum_count=minimum
                ),
                "coverage_risk": coverage_risk_curve(
                    raw, error, coverages, minimum_count=minimum
                ),
            },
            "calibrated": {
                "reliability_error_bins": calibration_bins(
                    calibrated, error, bins=bins, minimum_count=minimum
                ),
                "coverage_risk": coverage_risk_curve(
                    calibrated, error, coverages, minimum_count=minimum
                ),
            },
        }
    report = {
        "schema_version": SCHEMA,
        "requirement_id": "R10",
        "task_id": "T4.1a",
        "selection": {
            "roles_immutable": True,
            "role": ROLE,
            "aggregate_winner": None,
            "selected_candidate_conclusion": {
                "status": "unavailable",
                "reason": "no_selected_candidate_after_negative_T2.7_gate",
            },
        },
        "inputs": inputs,
        "user_set": _set_identity(common),
        "split": {
            "algorithm": "sha256-seeded-user-order/1.0",
            "seed": int(split["seed"]),
            "calibration_users": _set_identity(cal_users),
            "test_users": _set_identity(test_users),
            "overlap_count": 0,
        },
        "resampling": {
            "method": "seeded_window_bootstrap",
            "sampling_unit": "dense observed-history window",
            "replacement": True,
            **settings,
        },
        "controls": reports,
        "limitations": [
            "diagnostic controls only; does not complete selection-dependent T4.1",
            "factorized branch names do not establish semantics",
            "synthetic calibration may not transfer to real users",
        ],
    }
    if existing is not None:
        frozen_keys = ("user_set", "split", "inputs", "selection")
        drift = [key for key in frozen_keys if existing.get(key) != report.get(key)]
        if drift:
            raise ValueError(f"immutable calibration identity drift: {drift}")
    write_json(report, output)
    return report
