from __future__ import annotations

import argparse
import copy
import csv
import gzip
from pathlib import Path

import yaml

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
    config["run"].update(users=32, days=28, seed=20260905, output=str(output))
    config_path = tmp_path / "v4-small.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    simulator.activate_config(simulator.load_config(config_path))
    args = argparse.Namespace(
        config=str(config_path), output=str(output), overwrite=False,
        seed=20260905, users=32, days=28, start_date="2026-04-01",
        scenario="clean", requested_scenario="clean", full_kanto=False,
    )
    simulator.simulate(args)
    opportunities = _rows(output / "truth/stable_category_affinity_opportunities_truth.csv.gz")
    latents = _rows(output / "truth/user_latents.csv.gz")
    events = _rows(output / "observed/observed_events.csv.gz")
    assert len(opportunities) == 32 * 12
    assert {row["affinity_pair_id"] for row in opportunities} == {"cafe_restaurant", "grocery_mall"}
    assert {"emitted_observed_event_category", "emitted_observed_service_token", "event_visible_before_cutoff"}.issubset(opportunities[0])
    assert all(row["event_visible_before_cutoff"] in {"0", "1"} for row in opportunities)
    assert {row["stable_affinity_label"] for row in latents} == {"0", "1"}
    assert all("stable_affinity" not in row for row in events[0])
    assert all("utility" not in row for row in events[0])

    coverage = _stable_affinity_coverage(
        output, load_benchmark_spec(SPEC)["spec"]["protocol"]["affinity_pairs"], minimum=12,
    )
    assert coverage["eligible_users"] == 32
    assert coverage["balanced_discriminating_decisions"] == len(opportunities)
    assert all(count == 12 for count in coverage["per_user_counts"].values())
    # Ordinary event volume is deliberately not used as opportunity coverage;
    # only evaluator-only provenance tied to a matched candidate set counts.
    assert coverage["balanced_discriminating_decisions"] != len(events)
