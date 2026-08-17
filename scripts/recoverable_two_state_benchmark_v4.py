#!/usr/bin/env python3
"""Run v4 stable-category-affinity checks and gated recoverability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import recoverability_gate as gate
from geoembeddings.two_state_benchmark import (
    BENCHMARK_V4_SCHEMA, audit_pair, load_benchmark_spec, load_factor_registry,
    simulate_benchmark_pair,
)


SPEC = Path("configs/simulation/recoverable_two_state_benchmark_v4.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _affinity_baseline(run_dir: Path, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """Score only observed category tokens, oriented by evaluator-only pair truth."""
    users, _, metadata = gate._observed_history_matrix(run_dir)
    strata = gate._history_matching_strata(run_dir, users)
    events = gate._read(run_dir, "observed/observed_events.csv.gz")
    events["user_id"] = events["user_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    latent = gate._read(run_dir, "truth/user_latents.csv.gz")
    latent["user_id"] = latent["user_id"].astype(str)
    truth = latent.set_index("user_id")
    labels = pd.to_numeric(truth["stable_affinity_label"], errors="coerce").reindex(users).dropna().astype(int)
    pair_by_user = truth["stable_affinity_pair_id"].reindex(labels.index)
    category_by_user = truth["stable_affinity_category"].reindex(labels.index)
    pair_map = {
        str(pair["pair_id"]): list(pair["categories"])
        for pair in spec["protocol"]["affinity_pairs"]
    }
    counts = pd.DataFrame(0.0, index=labels.index, columns=["first", "second", "total"])
    for user_id, rows in events.loc[events["user_id"].isin(labels.index)].groupby("user_id"):
        categories = pair_map[str(pair_by_user.loc[user_id])]
        counts.loc[user_id, "first"] = float((rows["object_category"] == categories[0]).sum())
        counts.loc[user_id, "second"] = float((rows["object_category"] == categories[1]).sum())
        counts.loc[user_id, "total"] = float(rows["object_category"].isin(categories).sum())
    # Keep the feature model-visible: use the declared pair's canonical order,
    # never the evaluator-only preferred category, to orient the count signal.
    difference = pd.Series(0.0, index=labels.index)
    for user_id in labels.index:
        difference.loc[user_id] = counts.loc[user_id, "second"] - counts.loc[user_id, "first"]
    scores = 1.0 / (1.0 + np.exp(-np.clip(difference.to_numpy(dtype=float), -30.0, 30.0)))
    matched_mask, matching = gate._matched_user_mask(labels, strata)
    selected = matched_mask.to_numpy(dtype=bool)
    y = labels.to_numpy(dtype=int)[selected]
    selected_scores = scores[selected]
    x = selected_scores[:, None]
    clusters = labels.index.to_numpy(dtype=str)[selected]
    selected_strata = strata.reindex(labels.index).to_numpy()[selected]
    metrics = gate._metric_bundle(x, y, selected_scores)
    bootstrap = gate._cluster_bootstrap(
        x, y, selected_scores, clusters, neighbors=None,
        replicates=300, seed=seed + 4100,
    )
    null = gate._stratified_permutation_null(
        x, y, selected_scores, selected_strata, neighbors=None,
        permutations=100, seed=seed + 4200,
    )
    result = gate._gate_result(metrics, bootstrap, null, profile="v2")
    return {
        "status": result["status"],
        "feature_source": "observed/observed_events.csv.gz object_category tokens before the evaluator cutoff only",
        "feature": "count(second declared pair category) - count(first declared pair category), using observed object_category tokens only",
        "metadata": metadata,
        "matching": matching,
        "evaluated_users": int(len(clusters)),
        "metrics": metrics,
        "cluster_bootstrap": bootstrap,
        "stratified_permutation_null": null,
        "gate": result,
    }


def _causal_path_checks(run_dir: Path) -> dict[str, Any]:
    reference = Path(run_dir)
    config = yaml.safe_load((reference / "config.resolved.yaml").read_text(encoding="utf-8"))
    delay = float(config["events"]["local_commerce"]["event_delay_hours"])
    cutoff = gate._history_cutoff(reference)
    provenance = gate._read(reference, "truth/stable_category_affinity_opportunities_truth.csv.gz")
    choices = gate._read(reference, "truth/choices_truth.csv.gz").set_index("decision_id")
    events = gate._read(reference, "observed/observed_events.csv.gz")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    event_keys = {
        (str(row["user_id"]), int(pd.Timestamp(row["timestamp"]).value), str(row["object_id"])): row
        for _, row in events.loc[events["service_id"] == "local_commerce"].iterrows()
    }
    emitted = 0
    mismatches = []
    recorded_mismatches = []
    visibility_mismatches = []
    visible = 0
    for _, item in provenance.iterrows():
        choice = choices.loc[item["decision_id"]]
        choice_time = pd.Timestamp(choice["timestamp"])
        if choice_time.tzinfo is None:
            choice_time = choice_time.tz_localize("Asia/Tokyo")
        choice_visible = choice_time.tz_convert("UTC") < cutoff
        event_time = choice_time + pd.to_timedelta(delay, unit="h")
        event = event_keys.get((str(choice["user_id"]), int(event_time.tz_convert("UTC").value), str(choice["chosen_poi_id"])))
        if event is not None:
            emitted += 1
            visible += int(event_time.tz_convert("UTC") < cutoff)
            if event["object_category"] != choice["chosen_category"]:
                mismatches.append(item["decision_id"])
        recorded_category_value = item.get("emitted_observed_event_category", "")
        recorded_token_value = item.get("emitted_observed_service_token", "")
        recorded_category = "" if pd.isna(recorded_category_value) else str(recorded_category_value)
        recorded_token = "" if pd.isna(recorded_token_value) else str(recorded_token_value)
        actual_category = "" if event is None else str(event["object_category"])
        actual_token = "" if event is None else str(event["action_type"])
        if (recorded_category, recorded_token) != (actual_category, actual_token):
            recorded_mismatches.append(item["decision_id"])
        recorded_visible = bool(int(item.get("event_visible_before_cutoff", 0)))
        actual_visible = bool(event is not None and event_time.tz_convert("UTC") < cutoff)
        if recorded_visible != actual_visible or (not choice_visible and recorded_visible):
            visibility_mismatches.append(item["decision_id"])
    observed_fields = set()
    for path in sorted((reference / "observed").glob("*.csv.gz")):
        table = gate._read(reference, f"observed/{path.name}")
        observed_fields.update(table.columns)
    leaked = sorted(field for field in observed_fields if any(fragment in field.lower() for fragment in ("latent", "utility", "true_", "chosen", "episode", "change_")))
    return {
        "selected_category_matches_emitted_event": not mismatches,
        "emitted_events_checked": emitted,
        "category_mismatches": mismatches,
        "recorded_emission_matches_observed_event": not recorded_mismatches,
        "recorded_emission_mismatches": recorded_mismatches,
        "all_emitted_injected_events_before_cutoff": visible == emitted,
        "recorded_cutoff_visibility_matches": not visibility_mismatches,
        "cutoff_visibility_mismatches": visibility_mismatches,
        "visible_opportunities": visible,
        "injected_opportunities": len(provenance),
        "observed_truth_or_candidate_leakage": leaked,
        "passed": not mismatches and not recorded_mismatches and not visibility_mismatches and not leaked,
    }


def _evaluate(root: Path, spec_path: Path, seed: int, phase: str) -> dict[str, Any]:
    resolved = load_benchmark_spec(spec_path)
    if resolved["spec"].get("schema_version") != BENCHMARK_V4_SCHEMA:
        raise ValueError("the v4 runner requires the v4 benchmark specification")
    registry_path = Path(resolved["spec"]["factor_registry"])
    if not registry_path.is_absolute():
        registry_path = spec_path.resolve().parents[1] / "recoverability" / registry_path.name
    registry = load_factor_registry(registry_path)
    simulate_benchmark_pair(spec_path, root, seed=seed)
    coverage = audit_pair(root / "pair", resolved)
    _write_json(root / "v4_affinity_coverage.json", coverage)
    baseline = _affinity_baseline(root / "reference", resolved["spec"], seed)
    causal = _causal_path_checks(root / "reference")
    alpha = float(resolved["spec"]["protocol"]["oracle_probe_ridge_alpha"])
    temporary = gate.evaluate_temporary_schedule(
        root / "pair", folds=5, bootstrap_replicates=300,
        permutation_count=100, seed=seed + 2000, gate_profile="v2", probe_alpha=alpha,
    )
    report: dict[str, Any] = {
        "schema_version": "geoembeddings-recoverable-two-state-benchmark-evaluation/4.0",
        "protocol_amendment": "recoverable_two_state_benchmark_v4", "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "source": {"spec": str(spec_path.resolve()), "spec_sha256": _sha256(spec_path.resolve()),
                   "reference_run": str((root / "reference").resolve()), "pair_root": str((root / "pair").resolve())},
        "information_boundary": {"features": "observed/ only", "factor_labels": "evaluator-only truth/",
                                  "model_training_performed": False, "heldout_seed_opened": phase == "heldout"},
        "coverage": coverage, "affinity_baseline": baseline, "causal_path": causal,
        "temporary_schedule_state": temporary,
        "raw_feature_diagnostics": {"role": "reported diagnostic; not a data-generator feasibility gate"},
        "learned_representation_metric_statement": "kNN purity and separation are primary success metrics for the learned slow-fast representation, not prerequisites for the data generator.",
    }
    if baseline["status"] != "pass" or not causal["passed"]:
        report["persistent_preference"] = {"status": "blocked_by_affinity_baseline", "evaluated": False}
        report["track_a"] = {"status": "fail", "reason": "stable category-count baseline failed before full recoverability evaluation",
                              "both_factors_required": True, "no_aggregate_average": True}
    else:
        persistent = gate.evaluate_sustained_preference(
            root / "reference", registry, folds=5, bootstrap_replicates=300,
            permutation_count=100, seed=seed, gate_profile="v2", probe_alpha=alpha,
        )
        report["persistent_preference"] = persistent
        report["track_a"] = {
            "status": "pass" if coverage["status"] == "passed" and baseline["status"] == "pass"
            and persistent["status"] == "pass" and temporary["status"] == "pass" else "fail",
            "both_factors_required": True, "no_aggregate_average": True,
        }
    _write_json(root / f"{phase}_evaluation_report.json", report)
    return report


def _freeze(output_root: Path, spec_path: Path, reports: list[dict[str, Any]]) -> None:
    resolved = load_benchmark_spec(spec_path)
    frozen = output_root / "frozen_generator_config.yaml"
    frozen.write_text(yaml.safe_dump(resolved["config"], sort_keys=False), encoding="utf-8")
    _write_json(output_root / "calibration_report.json", {
        "schema_version": "geoembeddings-recoverable-two-state-benchmark-calibration/4.0",
        "protocol_amendment": "recoverable_two_state_benchmark_v4", "status": "passed",
        "development_seeds": [report["seed"] for report in reports], "development_reports": reports,
        "frozen_generator_config": str(frozen.resolve()), "heldout_seed_evaluated": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "heldout"))
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    output = args.output_root.resolve()
    resolved = load_benchmark_spec(spec_path)
    if args.phase == "development":
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"development output must be a new immutable root: {output}")
        output.mkdir(parents=True, exist_ok=True)
        reports = []
        for seed in resolved["seeds"]:
            report = _evaluate(output / f"seed{seed}", spec_path, seed, "development")
            reports.append(report)
            if report["track_a"]["status"] != "pass":
                break
        if all(report["track_a"]["status"] == "pass" for report in reports):
            _freeze(output, spec_path, reports)
            print(json.dumps({"status": "frozen", "heldout_authorized": True}, indent=2))
        else:
            _write_json(output / "calibration_report.json", {
                "schema_version": "geoembeddings-recoverable-two-state-benchmark-calibration/4.0",
                "protocol_amendment": "recoverable_two_state_benchmark_v4", "status": "failed",
                "development_reports": reports, "heldout_seed_evaluated": False,
                "model_training_performed": False,
            })
            print(json.dumps({"status": "failed", "heldout_authorized": False}, indent=2))
        return
    if args.calibration_root is None:
        parser.error("--calibration-root is required for heldout")
    calibration_path = args.calibration_root.resolve() / "calibration_report.json"
    if not calibration_path.is_file() or json.loads(calibration_path.read_text()).get("status") != "passed":
        raise RuntimeError("v4 held-out evaluation requires a passing development calibration")
    if output.exists():
        raise FileExistsError(f"held-out output must be a new immutable root: {output}")
    report = _evaluate(output, spec_path, resolved["heldout_seed"], "heldout")
    if report["track_a"]["status"] == "pass":
        (output / "final_benchmark_acceptance_report.md").write_text(
            "# recoverable_two_state_benchmark_v4 acceptance\n\nStatus: **accepted for Track A data recoverability**.\n\nNo representation model was trained.\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
