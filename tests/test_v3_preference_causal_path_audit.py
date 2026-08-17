from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from geoembeddings import simulator
from geoembeddings.two_state_benchmark import load_benchmark_spec

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_two_state_v3_preference_causal_path import audit_run


SPEC_V3 = Path("configs/simulation/recoverable_two_state_benchmark_v3.yaml")


def test_causal_audit_checks_utility_observation_and_cutoff_paths(tmp_path: Path) -> None:
    config = copy.deepcopy(load_benchmark_spec(SPEC_V3)["config"])
    output = tmp_path / "v3-causal-audit-run"
    config["run"].update(users=16, days=28, seed=20260904, output=str(output))
    config_path = tmp_path / "v3-causal-audit.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    simulator.activate_config(simulator.load_config(config_path))
    args = argparse.Namespace(
        config=str(config_path), output=str(output), overwrite=False,
        seed=20260904, users=16, days=28, start_date="2026-04-01",
        scenario="clean", requested_scenario="clean", full_kanto=False,
    )
    simulator.simulate(args)
    report = audit_run(output, 20260904, tmp_path / "causal-trace.jsonl.gz")
    assert report["assertions"]["label_alignment"]
    assert report["assertions"]["utility_effect"]
    assert report["assertions"]["choice_response"]
    assert report["assertions"]["observation_path"]
    assert report["assertions"]["cutoff_path"]
    assert report["cutoff"]["visible_before_cutoff"] == report["cutoff"]["injected_opportunities"]
    assert report["observation_path"]["checked_emitted_events"] > 0

