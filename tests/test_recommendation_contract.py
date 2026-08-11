from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd
import pytest

from geoembeddings import simulator
from geoembeddings.contract import DATASET_CONTRACT_VERSION, OBSERVED_FILES
from geoembeddings.recommendation import FORBIDDEN_PARTS, IMPRESSION_FIELDS, validate_recommendation_tables


def _simulate(root: Path):
    config_path = Path("configs/simulation/kanto_v1.yaml")
    config = simulator.load_config(config_path)
    config["run"].update(users=10, days=2, seed=20260811, output=str(root))
    simulator.activate_config(config)
    return simulator.simulate(argparse.Namespace(output=str(root), overwrite=False, seed=20260811, users=10, days=2,
        start_date=config["run"]["start_date"], scenario=config["run"]["scenario"], full_kanto=False, config=str(config_path)))


def test_fixed_seed_hakone_observed_contract_and_naive_ranker_gate(tmp_path) -> None:
    manifest = _simulate(tmp_path / "run")
    assert manifest["dataset_contract"]["version"] == DATASET_CONTRACT_VERSION
    diagnostics = manifest["recommendation_contract"]
    assert diagnostics["requests"] == 10
    assert diagnostics["available_candidates"] > diagnostics["requests"]
    assert all(value["requests_scored"] == 10 for value in diagnostics["naive_rankers"].values())
    observed = tmp_path / "run" / "observed"
    frames = {key: pd.read_csv(observed / filename) for key, filename in OBSERVED_FILES.items() if key not in {"users", "events"}}
    assert set(frames["poi_catalog"].query("region_id == 'hakone'")["category"]) >= {"onsen", "restaurant", "cafe", "shop", "hotel", "attraction"}
    assert tuple(frames["impressions"].columns) == IMPRESSION_FIELDS
    assert not any(any(part in column.lower() for part in FORBIDDEN_PARTS) for frame in frames.values() for column in frame.columns)
    assert (frames["impressions"]["is_shown"] <= frames["impressions"]["is_available"]).all()


def test_recommendation_validation_rejects_field_order_availability_and_leakage() -> None:
    tables = {"poi_catalog": [], "recommendation_requests": [], "impressions": [], "interactions": []}
    bad = dict(zip(reversed(IMPRESSION_FIELDS), range(len(IMPRESSION_FIELDS))))
    tables["impressions"] = [bad]
    with pytest.raises(ValueError, match="field order"):
        validate_recommendation_tables(tables)
