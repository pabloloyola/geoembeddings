from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from geoembeddings.simulate_pair import simulate_pair
from geoembeddings import simulator
from geoembeddings.simulator import change_interval
from datetime import date


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("kind", ["exposure", "opportunity", "observation"])
def test_configured_intervention_changes_only_declared_fields(tmp_path: Path, kind: str) -> None:
    result = simulate_pair(CONFIG, tmp_path / f"{kind}-reference",
        tmp_path / f"{kind}-intervention", tmp_path / f"{kind}-pair",
        intervention=kind, users=10, days=2, seed=20260811)
    assert result["status"] == "passed"
    manifest = json.loads(Path(result["pair_manifest"]).read_text())
    report = json.loads(Path(result["pair_integrity"]).read_text())
    behavioral = json.loads(Path(result["behavioral_diagnostics"]).read_text())
    definition = yaml.safe_load(CONFIG.read_text())["interventions"][kind]
    assert manifest["invariant_entity_classes"] == definition["invariant_entities"]
    assert manifest["allowed_to_change_fields"] == definition["permitted_changes"]
    assert manifest["intervention_parameters"]["affected_random_streams"] == definition["affected_random_streams"]
    assert all(item["passed"] for item in report["entity_invariants"].values())
    assert report["table_results"]["truth.user_latents"]["allowed_changes"] == {}
    assert report["table_results"]["truth.world_regions"]["allowed_changes"] == {}
    assert report["table_results"]["truth.world_pois"]["allowed_changes"] == {}
    assert report["table_results"]["truth.episodes"]["allowed_changes"] == {}
    assert sum(value["difference_count"] for value in report["allowed_change_results"].values()) > 0
    assert behavioral["status"] == "passed"
    assert all(item["passed"] for item in behavioral["diagnostics"].values())


def test_simulate_pair_is_immutable(tmp_path: Path) -> None:
    reference, changed, pair = tmp_path / "reference", tmp_path / "changed", tmp_path / "pair"
    simulate_pair(CONFIG, reference, changed, pair, intervention="observation",
                  users=10, days=2, seed=17)
    with pytest.raises(FileExistsError, match="overwrite"):
        simulate_pair(CONFIG, reference, changed, pair, intervention="observation",
                      users=10, days=2, seed=17)


def test_explicit_clean_scenario_is_authoritative(tmp_path: Path) -> None:
    result = simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                           intervention="sustained-preference", scenario="clean", users=10, days=9, seed=17)
    for name in ("reference", "intervention"):
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        assert manifest["requested_scenario"] == "clean"
        assert manifest["resolved_scenario"] == "clean"
        assert manifest["scenario"] == "clean"
        report = json.loads((tmp_path / name / "deep_validation_report.json").read_text())
        assert report["requested_scenario"] == "clean"
        assert report["resolved_scenario"] == "clean"
    assert result["status"] == "passed"


def test_explicit_mixed_scenario_remains_mixed(tmp_path: Path) -> None:
    simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                  intervention="sustained-preference", scenario="mixed", users=10, days=9, seed=17)
    manifest = json.loads((tmp_path / "reference" / "manifest.json").read_text())
    assert manifest["requested_scenario"] == "mixed"
    assert manifest["resolved_scenario"] == "mixed"


def test_unknown_or_unversioned_mismatched_scenario_fails() -> None:
    config = simulator.load_config(CONFIG)
    with pytest.raises(ValueError, match="Unknown requested scenario"):
        simulator.resolve_scenario(config, "not-a-scenario")
    config["run"]["resolved_scenario"] = "mixed"
    with pytest.raises(ValueError, match="differs from requested"):
        simulator.resolve_scenario(config, "clean")


def test_schedule_shift_preserves_preferences_and_one_off_context(tmp_path: Path) -> None:
    result = simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                           intervention="schedule-shift", users=12, days=7, seed=20260811)
    manifest = json.loads(Path(result["pair_manifest"]).read_text())
    integrity = json.loads(Path(result["pair_integrity"]).read_text())
    assert result["status"] == "passed"
    assert manifest["intervention_parameters"]["schedule_shift"] == {"weekday_hours": 2.0, "weekend_hours": -1.0}
    assert integrity["table_results"]["truth.user_latents"]["allowed_changes"] == {}
    assert integrity["table_results"]["truth.episodes"]["allowed_changes"] == {}
    assert integrity["table_results"]["truth.choices"]["allowed_changes"]["truth.choices.timestamp"] > 0


def test_temporary_schedule_shift_is_finite_and_interval_bounded(tmp_path: Path) -> None:
    result = simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                           intervention="temporary_schedule_shift_v1", scenario="clean",
                           users=20, days=8, seed=20260811)
    manifest = json.loads(Path(result["pair_manifest"]).read_text())
    intervention_manifest = json.loads((tmp_path / "intervention" / "manifest.json").read_text())
    truth = _rows(tmp_path / "intervention" / "truth" / "temporary_schedule_shift_truth.csv.gz")
    assert result["status"] == "passed"
    assert manifest["intervention_type"] == "temporary_schedule_shift_v1"
    assert manifest["intervention_parameters"]["declaration_version"] == "geoembeddings-temporary-schedule-intervention/1.0"
    assert intervention_manifest["requested_scenario"] == intervention_manifest["resolved_scenario"] == "clean"
    assert truth and all(row["change_start_time"] < row["change_end_time"] for row in truth)
    selected = {row["user_id"] for row in truth if row["selected"] == "1"}
    assert selected
    start = truth[0]["change_start_time"]
    end = truth[0]["change_end_time"]
    reference_events = _rows(tmp_path / "reference" / "observed" / "observed_events.csv.gz")
    changed_events = _rows(tmp_path / "intervention" / "observed" / "observed_events.csv.gz")

    def outside(rows: list[dict[str, str]]) -> Counter[str]:
        return Counter(json.dumps(row, sort_keys=True) for row in rows
                       if row["timestamp"] < start or row["timestamp"] >= end)

    assert outside(reference_events) == outside(changed_events)
    affected = _rows(tmp_path / "intervention" / "truth" / "temporary_schedule_shift_events.csv.gz")
    assert affected
    assert {row["user_id"] for row in affected} <= selected
    assert all(start <= row["timestamp"] < end for row in affected)


def test_sustained_and_temporary_schedule_truth_are_independent(tmp_path: Path) -> None:
    sustained = simulate_pair(CONFIG, tmp_path / "s-ref", tmp_path / "s-int", tmp_path / "s-pair",
                              intervention="sustained-preference", scenario="clean", users=12, days=9, seed=20260811)
    temporary = simulate_pair(CONFIG, tmp_path / "t-ref", tmp_path / "t-int", tmp_path / "t-pair",
                              intervention="temporary_schedule_shift_v1", scenario="clean", users=12, days=9, seed=20260811)
    assert json.loads(Path(sustained["pair_manifest"]).read_text())["intervention_type"] == "sustained-preference"
    assert json.loads(Path(temporary["pair_manifest"]).read_text())["intervention_type"] == "temporary_schedule_shift_v1"
    for prefix in ("s", "t"):
        assert (tmp_path / f"{prefix}-int" / "truth" / "user_latents.csv.gz").read_bytes() == (tmp_path / f"{prefix}-ref" / "truth" / "user_latents.csv.gz").read_bytes()
        assert (tmp_path / f"{prefix}-int" / "truth" / "episodes_truth.csv.gz").read_bytes() == (tmp_path / f"{prefix}-ref" / "truth" / "episodes_truth.csv.gz").read_bytes()
    assert (tmp_path / "s-int" / "truth" / "change_points_truth.csv.gz").is_file()
    assert (tmp_path / "t-int" / "truth" / "temporary_schedule_shift_truth.csv.gz").is_file()


def test_change_interval_duration_and_censoring() -> None:
    assert change_interval(date(2026, 1, 1), 10, {"start_day_offset": 3, "duration_days": 2}) == (date(2026, 1, 4), date(2026, 1, 6))
    assert change_interval(date(2026, 1, 1), 10, {"start_day_offset": 3, "duration_days": None}) == (date(2026, 1, 4), None)
    with pytest.raises(ValueError, match="post-change"):
        change_interval(date(2026, 1, 1), 5, {"start_day_offset": 3, "duration_days": 2})


def test_category_preference_changes_category_probability_before_candidate_selection() -> None:
    config = simulator.load_config(CONFIG)
    simulator.activate_config(config)
    low = {"pref_cafe": 0.0}
    high = {"pref_cafe": 1.0}
    low_weights = simulator.category_weights_for_user("routine", low)
    high_weights = simulator.category_weights_for_user("routine", high)
    low_share = low_weights["cafe"] / sum(low_weights.values())
    high_share = high_weights["cafe"] / sum(high_weights.values())
    assert high_share > low_share
    assert high_weights["grocery"] == low_weights["grocery"]


@pytest.mark.parametrize("kind", ["temporary-trip", "sustained-preference"])
def test_change_pairs_preserve_identities_and_protect_change_truth(tmp_path: Path, kind: str) -> None:
    result = simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                           intervention=kind, users=10, days=9, seed=20260811)
    manifest = json.loads(Path(result["pair_manifest"]).read_text())
    assert result["status"] == "passed"
    assert set(manifest["invariant_entity_classes"]) >= {"users", "episodes", "choices"}
    assert not (tmp_path / "intervention" / "observed" / "change_points_truth.csv.gz").exists()
    assert (tmp_path / "intervention" / "truth" / "change_points_truth.csv.gz").is_file()
    behavioral = json.loads(Path(result["behavioral_diagnostics"]).read_text())
    assert behavioral["schema_version"] == "geoembeddings-pair-behavioral-diagnostics/2.0"
    assert behavioral["diagnostics"]["pre_change_observed_events_identical"]["passed"]
    assert behavioral["diagnostics"]["observed_changed_users_during_change"]["intervention"] > 0
    assert behavioral["diagnostics"]["target_category_choice_rate_during_change"]["passed"]
    assert behavioral["diagnostics"]["observed_target_category_event_rate_during_change"]["passed"]
