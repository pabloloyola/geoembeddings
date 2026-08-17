from __future__ import annotations

import argparse
import copy
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import recoverability_gate as gate
from geoembeddings import simulator
from geoembeddings.two_state_benchmark import (
    _stable_affinity_coverage, load_benchmark_spec, load_factor_registry,
)


SPEC = Path("configs/simulation/recoverable_two_state_benchmark_v4.yaml")
REGISTRY = Path("configs/recoverability/recoverable_two_state_benchmark_v4_factor_registry.json")


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v4_declares_fixed_balanced_affinity_pairs_and_reuses_schedule() -> None:
    resolved = load_benchmark_spec(SPEC)
    assert resolved["spec"]["benchmark_id"] == "recoverable_two_state_benchmark_v4"
    assert resolved["spec"]["protocol"]["affinity_pairs"] == [
        {"pair_id": "cafe_restaurant", "categories": ["cafe", "restaurant"]},
        {"pair_id": "grocery_mall", "categories": ["grocery", "mall"]},
    ]
    assert resolved["spec"]["protocol"]["calibration"]["mutable_parameters"] == []
    assert resolved["config"]["choice"]["category_preference_scale"] == 0.0
    assert resolved["config"]["choice"]["preference_weight"] == 0.0
    assert resolved["config"]["interventions"]["temporary_schedule_shift_v1"]["schedule_shift"] == {
        "start_day_offset": 35, "duration_days": 14, "weekday_hours": 5.0, "weekend_hours": -4.0,
    }
    assert load_factor_registry(REGISTRY)["protocol_amendment"] == "recoverable_two_state_benchmark_v4"


def test_v4_small_run_records_affinity_provenance_without_observed_truth(tmp_path: Path) -> None:
    config = copy.deepcopy(load_benchmark_spec(SPEC)["config"])
    output = tmp_path / "v4-small"
    config["run"].update(users=256, days=28, seed=20260905, output=str(output))
    config_path = tmp_path / "v4-small.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    simulator.activate_config(simulator.load_config(config_path))
    args = argparse.Namespace(
        config=str(config_path), output=str(output), overwrite=False,
        seed=20260905, users=256, days=28, start_date="2026-04-01",
        scenario="clean", requested_scenario="clean", full_kanto=False,
    )
    simulator.simulate(args)
    opportunities = _rows(output / "truth/stable_category_affinity_opportunities_truth.csv.gz")
    latents = _rows(output / "truth/user_latents.csv.gz")
    events = _rows(output / "observed/observed_events.csv.gz")
    assert len(opportunities) == 256 * 12
    assert {row["affinity_pair_id"] for row in opportunities} == {"cafe_restaurant", "grocery_mall"}
    assert {"emitted_observed_event_category", "emitted_observed_service_token", "event_visible_before_cutoff"}.issubset(opportunities[0])
    assert all(row["event_visible_before_cutoff"] in {"0", "1"} for row in opportunities)
    assert {row["stable_affinity_label"] for row in latents} == {"0", "1"}
    assert all("stable_affinity" not in row for row in events[0])
    assert all("utility" not in row for row in events[0])

    coverage = _stable_affinity_coverage(
        output, load_benchmark_spec(SPEC)["spec"]["protocol"]["affinity_pairs"], minimum=12,
    )
    assert coverage["eligible_users"] == 256
    assert coverage["balanced_discriminating_decisions"] == len(opportunities)
    assert all(count == 12 for count in coverage["per_user_counts"].values())
    # Ordinary event volume is deliberately not used as opportunity coverage;
    # only evaluator-only provenance tied to a matched candidate set counts.
    assert coverage["balanced_discriminating_decisions"] != len(events)

    users, full_features, _ = gate._observed_history_matrix(output)
    strata = gate._history_matching_strata(output, users)
    latent_frame = gate._read(output, "truth/user_latents.csv.gz")
    latent_frame["user_id"] = latent_frame["user_id"].astype(str)
    latent = latent_frame.set_index("user_id")
    labels = pd.to_numeric(latent["stable_affinity_label"], errors="coerce").reindex(users).astype(int)
    pair_map = {
        str(pair["pair_id"]): list(pair["categories"])
        for pair in load_benchmark_spec(SPEC)["spec"]["protocol"]["affinity_pairs"]
    }
    sentinel_feature = gate._observed_category_count_difference(output, users, pair_map)
    matched, _ = gate._matched_user_mask(labels, strata)
    direct_scores = 1.0 / (1.0 + np.exp(-sentinel_feature.iloc[:, 0].to_numpy()))
    direct_metrics = gate._metric_bundle(
        sentinel_feature.to_numpy()[matched.to_numpy()], labels.to_numpy()[matched.to_numpy()],
        direct_scores[matched.to_numpy()],
    )
    full = gate._score_binary_factor(
        "stable_affinity_label", labels, full_features, users, strata,
        load_factor_registry(REGISTRY)["factors"][0], folds=5,
        bootstrap_replicates=1, permutation_count=1, seed=20260905,
        gate_profile="v2", probe_alpha=1000.0, feature_override=sentinel_feature,
    )
    assert full["evaluated_users"] == int(matched.sum())
    assert abs(full["metrics"]["balanced_accuracy"] - direct_metrics["balanced_accuracy"]) <= 0.10
    assert abs(full["metrics"]["auroc"] - direct_metrics["auroc"]) <= 0.10
    with pytest.raises(ValueError, match="exact evaluator user index"):
        gate._score_binary_factor(
            "stable_affinity_label", labels, full_features, users, strata,
            load_factor_registry(REGISTRY)["factors"][0], folds=5,
            bootstrap_replicates=1, permutation_count=1, seed=20260905,
            gate_profile="v2", feature_override=sentinel_feature.iloc[:-1],
        )
