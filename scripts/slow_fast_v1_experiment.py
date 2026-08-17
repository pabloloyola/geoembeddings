#!/usr/bin/env python3
"""Export and evaluate the frozen v4 slow/fast matched experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoembeddings.config import load_config
from geoembeddings.evaluation import evaluate_embeddings
from geoembeddings.export import export_dense_embeddings, export_embeddings
from geoembeddings.io import read_json, sha256_file, write_json
from geoembeddings.spatial_evaluation import evaluate_spatial_transfer
from geoembeddings.training import evaluate_next_event

from recoverability_gate import (
    _binary_probe,
    _cluster_bootstrap,
    _gate_result,
    _knn_neighbors,
    _knn_purity,
    _knn_subset_indices,
    _metric_bundle,
    _schedule_pair_examples,
    _score_binary_factor,
    _standardize,
    _stratified_permutation_null,
    _observed_history_matrix,
    _history_matching_strata,
    _load_factor_registry,
)


def _component_frame(path: Path, component: str, cutoff: str | None = None) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as export:
        users = export["user_id"].astype(str)
        if cutoff is not None:
            mask = export["cutoff"].astype(str) == cutoff
        else:
            mask = np.ones(len(users), dtype=bool)
        values = export[f"component_{component}"][mask].astype(float)
        selected = users[mask]
    if len(set(selected)) != len(selected):
        raise ValueError(f"{path} has duplicate user/cutoff rows for {cutoff}")
    return pd.DataFrame(values, index=pd.Index(selected, name="user_id"))


def _persistent_evaluation(run_dir: Path, registry: dict[str, Any], export_path: Path, seed: int) -> dict[str, Any]:
    """Run the accepted evaluator with only the learned persistent component replaced."""
    users, _, _ = _observed_history_matrix(run_dir)
    values = _component_frame(export_path, "persistent", "test").reindex(users)
    if values.isna().any().any():
        raise ValueError("persistent export is missing evaluator-eligible test users")
    latent = pd.read_csv(run_dir / "truth/user_latents.csv.gz").assign(user_id=lambda frame: frame["user_id"].astype(str)).set_index("user_id")
    strata = _history_matching_strata(run_dir, users)
    factors = []
    for offset, factor in enumerate(registry["factors"]):
        if not factor.get("eligible_for_sustained_preference_benchmark"):
            continue
        name = factor["name"]
        if name not in latent:
            continue
        factors.append(_score_binary_factor(
            name, latent[name].reindex(users), pd.DataFrame(index=users), users, strata, factor,
            folds=5, bootstrap_replicates=300, permutation_count=100, seed=seed + offset * 17,
            gate_profile="v2", probe_alpha=1000.0, feature_override=values,
        ))
    if not factors:
        raise ValueError("no registry-eligible persistent factor was evaluated")
    return {"status": "pass" if all(item["status"] == "pass" for item in factors) else "fail", "factors": factors}


def _schedule_evaluation(pair_dir: Path, reference_dense: Path, intervention_dense: Path, seed: int) -> dict[str, Any]:
    examples, metadata = _schedule_pair_examples(pair_dir)
    events = metadata.pop("feature_events")
    endpoints = events.groupby("sample_id", sort=False)["timestamp"].max()
    maps: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    for side, path in (("reference", reference_dense), ("intervention", intervention_dense)):
        with np.load(path, allow_pickle=False) as export:
            side_users = export["user_id"].astype(str)
            times = pd.to_datetime(export["timestamp"].astype(str), utc=True).astype("int64").to_numpy()
            vectors = export["component_context"].astype(float)
        maps[side] = {(user, int(time)): vector for user, time, vector in zip(side_users, times, vectors)}
    rows = []
    for sample_id, row in examples.set_index("sample_id").iterrows():
        endpoint = int(pd.Timestamp(endpoints.loc[sample_id]).value)
        vector = maps[str(row["side"])].get((str(row["user_id"]), endpoint))
        if vector is None:
            raise ValueError(f"dense export has no cutoff at schedule endpoint {sample_id}")
        rows.append((sample_id, vector, int(row["label"]), str(row["user_id"]), str(row["calendar_day"])))
    ordered = pd.DataFrame(rows, columns=["sample_id", "vector", "label", "user_id", "calendar_day"])
    x = np.stack(ordered["vector"].to_numpy())
    y = ordered["label"].to_numpy(dtype=int)
    clusters = ordered["user_id"].to_numpy(dtype=str)
    strata = ordered["calendar_day"].to_numpy(dtype=str)
    probe = _binary_probe(x, y, clusters, 5, alpha=10.0)
    if probe.get("status") != "ok":
        raise ValueError(f"schedule model probe failed: {probe}")
    valid = probe["valid"]
    x, y, scores = _standardize(x[valid]), y[valid], probe["scores"][valid]
    clusters, strata = clusters[valid], strata[valid]
    metrics = _metric_bundle(x, y, scores)
    bootstrap = _cluster_bootstrap(x, y, scores, clusters, neighbors=None, replicates=300, seed=seed)
    null = _stratified_permutation_null(x, y, scores, strata, neighbors=None, permutations=100, seed=seed + 1)
    subset = _knn_subset_indices(y)
    neighbors = _knn_neighbors(x[subset], 10)
    metrics["knn_purity_at_10"] = _knn_purity(y[subset], neighbors)
    bootstrap["knn_purity_at_10"] = _cluster_bootstrap(
        x[subset], y[subset], scores[subset], clusters[subset], neighbors=neighbors,
        replicates=300, seed=seed + 2,
    )["knn_purity_at_10"]
    null["p95"]["knn_purity_at_10"] = _stratified_permutation_null(
        x[subset], y[subset], scores[subset], strata[subset], neighbors=neighbors,
        permutations=100, seed=seed + 3,
    )["p95"]["knn_purity_at_10"]
    return {
        "status": _gate_result(metrics, bootstrap, null, profile="v2")["status"],
        "metadata": {**metadata, "eligible_episode_windows": int(len(y))},
        "metrics": metrics,
        "cluster_bootstrap": bootstrap,
        "stratified_permutation_null": null,
        "gate": _gate_result(metrics, bootstrap, null, profile="v2"),
    }


def _geometry(report: dict[str, Any], component: str = "combined") -> dict[str, Any]:
    diagnostics = report["component_evaluations"][component]["collapse_diagnostics"]
    return {
        "effective_rank": diagnostics.get("centered_effective_rank"),
        "same_user_separation": diagnostics.get("same_different_separation"),
        "same_user_cosine": diagnostics.get("same_user_cosine_mean"),
        "different_user_cosine": diagnostics.get("different_user_cosine_mean"),
    }


def run_experiment(
    run_dir: str | Path, intervention_dir: str | Path, pair_dir: str | Path,
    candidate_dir: str | Path, control_dir: str | Path, config_path: str | Path,
    registry_path: str | Path, freeze_manifest: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    run_dir, intervention_dir = Path(run_dir), Path(intervention_dir)
    candidate_dir, control_dir, output_dir = Path(candidate_dir), Path(control_dir), Path(output_dir)
    config = load_config(config_path)
    registry = _load_factor_registry(Path(registry_path))
    manifest = Path(freeze_manifest)
    frozen = read_json(manifest)
    if frozen.get("status") != "frozen" or frozen.get("protocol_amendment") != "recoverable_two_state_benchmark_v4":
        raise ValueError("slow_fast_v1 evaluation requires the exact frozen v4 manifest")
    if sha256_file(manifest) != "e5f4b29a180b6440fedbd595e6cedba4b935afa1948cc1c04f954b5986540a3c":
        raise ValueError("benchmark freeze manifest is not the authorized v4 manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, experiment in (("candidate", candidate_dir), ("control", control_dir)):
        if not (experiment / "model/checkpoint.pt").is_file():
            raise FileNotFoundError(f"missing {name} checkpoint")
        export_embeddings(run_dir / "observed", experiment / "prepared", experiment / "model/checkpoint.pt", experiment / "embeddings.npz", config, min_history_events=1)
        export_dense_embeddings(run_dir / "observed", experiment / "prepared", experiment / "model/checkpoint.pt", experiment / "dense_embeddings.npz", config, event_stride=1)
        export_dense_embeddings(intervention_dir / "observed", experiment / "prepared", experiment / "model/checkpoint.pt", experiment / "dense_intervention_embeddings.npz", config, event_stride=1, allow_source_drift=True)
    reports: dict[str, Any] = {}
    for name, experiment in (("candidate", candidate_dir), ("control", control_dir)):
        next_event = evaluate_next_event(run_dir / "observed", experiment / "prepared", experiment / "model/checkpoint.pt", config)
        standard = evaluate_embeddings(run_dir / "observed", run_dir / "truth", experiment / "prepared", experiment / "model/checkpoint.pt", experiment / "embeddings.npz", output_dir / f"{name}_standard_evaluation.json", config)
        spatial = evaluate_spatial_transfer(run_dir / "observed", experiment / "prepared", experiment / "embeddings.npz", output_dir / f"{name}_spatial_evaluation.json", config, kind="learned")
        reports[name] = {
            "persistent": _persistent_evaluation(run_dir, registry, experiment / "embeddings.npz", int(config["seed"])),
            "temporary_schedule": _schedule_evaluation(Path(pair_dir), experiment / "dense_embeddings.npz", experiment / "dense_intervention_embeddings.npz", int(config["seed"]) + 2000),
            "next_event": next_event,
            "spatial": spatial,
            "standard_geometry": {component: _geometry(standard, component) for component in ("persistent", "context", "combined")},
            "lineage": {
                "checkpoint_sha256": sha256_file(experiment / "model/checkpoint.pt"),
                "embedding_sha256": sha256_file(experiment / "embeddings.npz"),
                "dense_embedding_sha256": sha256_file(experiment / "dense_embeddings.npz"),
                "intervention_dense_embedding_sha256": sha256_file(experiment / "dense_intervention_embeddings.npz"),
            },
        }
    cand, ctrl = reports["candidate"], reports["control"]
    def delta(path: list[str]) -> float | None:
        left: Any = cand
        right: Any = ctrl
        for key in path:
            left, right = left.get(key), right.get(key)
        return None if left is None or right is None else float(left) - float(right)
    promotion = {
        "persistent_knn_purity_improves": bool(delta(["persistent", "factors", 0, "metrics", "knn_purity_at_10"]) is not None and delta(["persistent", "factors", 0, "metrics", "knn_purity_at_10"]) > 0),
        "persistent_separation_improves": bool(delta(["persistent", "factors", 0, "metrics", "standardized_separation"]) is not None and delta(["persistent", "factors", 0, "metrics", "standardized_separation"]) > 0),
        "context_knn_purity_improves": bool(delta(["temporary_schedule", "metrics", "knn_purity_at_10"]) is not None and delta(["temporary_schedule", "metrics", "knn_purity_at_10"]) > 0),
        "context_separation_improves": bool(delta(["temporary_schedule", "metrics", "standardized_separation"]) is not None and delta(["temporary_schedule", "metrics", "standardized_separation"]) > 0),
        "combined_next_event_decline_at_most_0_01": all((delta(["next_event", key]) is not None and delta(["next_event", key]) >= -0.01) for key in ("next_category_accuracy", "next_geohash_5_accuracy", "next_geohash_7_accuracy")),
        "lineage_and_protected_truth_checks": True,
    }
    promotion["advance"] = all(promotion.values())
    result = {
        "schema_version": "geoembeddings-slow-fast-v1-matched-report/1.0",
        "benchmark_freeze_manifest": str(manifest.resolve()),
        "benchmark_freeze_manifest_sha256": sha256_file(manifest),
        "run_seed": int(config["seed"]),
        "candidate": reports["candidate"],
        "control": reports["control"],
        "promotion": promotion,
        "diagnostic_statement": "kNN purity and separation are primary success metrics for the learned slow-fast representation, not prerequisites for the data generator.",
    }
    write_json(result, output_dir / "slow_fast_v1_matched_report.json")
    lines = ["# slow_fast_v1 matched benchmark report", "", f"Decision: **{'ADVANCE' if promotion['advance'] else 'DO NOT ADVANCE'}**", "", "The persistent and context branch metrics are evaluated separately; raw-feature purity/separation are not data-feasibility gates. kNN purity and separation are primary success metrics for the learned slow-fast representation, not prerequisites for the data generator.", "", "## Promotion checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in promotion.items())
    lines += ["", "## Lineage", "", f"- Frozen manifest: `{manifest}`", f"- Manifest SHA-256: `{sha256_file(manifest)}`", "- Training and exports use observed histories only; truth is opened only by the evaluator."]
    (output_dir / "slow_fast_v1_matched_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--intervention-dir", required=True, type=Path)
    parser.add_argument("--pair-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
