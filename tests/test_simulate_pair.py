from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from geoembeddings.simulate_pair import simulate_pair
from geoembeddings.simulator import change_interval
from datetime import date


CONFIG = Path("configs/simulation/kanto_v1.yaml")


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


def test_change_interval_duration_and_censoring() -> None:
    assert change_interval(date(2026, 1, 1), 10, {"start_day_offset": 3, "duration_days": 2}) == (date(2026, 1, 4), date(2026, 1, 6))
    assert change_interval(date(2026, 1, 1), 10, {"start_day_offset": 3, "duration_days": None}) == (date(2026, 1, 4), None)
    with pytest.raises(ValueError, match="post-change"):
        change_interval(date(2026, 1, 1), 5, {"start_day_offset": 3, "duration_days": 2})


@pytest.mark.parametrize("kind", ["temporary-trip", "sustained-preference"])
def test_change_pairs_preserve_identities_and_protect_change_truth(tmp_path: Path, kind: str) -> None:
    result = simulate_pair(CONFIG, tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
                           intervention=kind, users=10, days=9, seed=20260811)
    manifest = json.loads(Path(result["pair_manifest"]).read_text())
    assert result["status"] == "passed"
    assert set(manifest["invariant_entity_classes"]) >= {"users", "episodes", "choices"}
    assert not (tmp_path / "intervention" / "observed" / "change_points_truth.csv.gz").exists()
    assert (tmp_path / "intervention" / "truth" / "change_points_truth.csv.gz").is_file()
