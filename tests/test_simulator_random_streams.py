from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from geoembeddings import simulator
from geoembeddings.simulation_validation import validate


CONFIG = Path("configs/simulation/kanto_v1.yaml")


def _run(root: Path, *, seed: int = 20260811, observation_seed: int | None = None) -> dict:
    config = simulator.load_config(CONFIG)
    config["run"].update(users=50, days=7, seed=seed, output=str(root))
    if observation_seed is not None:
        config["run"]["random_streams"] = {"observation": observation_seed}
    simulator.activate_config(config)
    args = argparse.Namespace(
        output=str(root), overwrite=False, seed=seed, users=50, days=7,
        start_date=config["run"]["start_date"], scenario=config["run"]["scenario"],
        full_kanto=False, config=str(CONFIG),
    )
    return simulator.simulate(args)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_named_stream_derivation_is_repeatable_and_independent():
    first = simulator.make_random_streams(17)
    second = simulator.make_random_streams(17)
    assert first.seeds == second.seeds
    assert len(set(first.seeds.values())) == len(simulator.RANDOM_STREAM_NAMES)
    assert [first.generators[name].random() for name in simulator.RANDOM_STREAM_NAMES] == [
        second.generators[name].random() for name in simulator.RANDOM_STREAM_NAMES
    ]

    changed = simulator.make_random_streams(17, {"observation": 99})
    assert changed.seeds["observation"] == 99
    assert {name: changed.seeds[name] for name in simulator.RANDOM_STREAM_NAMES if name != "observation"} == {
        name: first.seeds[name] for name in simulator.RANDOM_STREAM_NAMES if name != "observation"
    }


def test_fixed_seed_simulation_provenance_identity_and_observation_independence(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    changed = tmp_path / "changed-observation"
    manifest = _run(reference)
    _run(repeat)
    _run(changed, observation_seed=99)

    assert validate(reference)["status"] == "passed"
    assert manifest["random_streams"]["root_seed"] == 20260811
    assert set(manifest["random_streams"]["seeds"]) == set(simulator.RANDOM_STREAM_NAMES)
    resolved = yaml.safe_load((reference / "config.resolved.yaml").read_text())
    assert resolved["run"]["random_streams"] == manifest["random_streams"]["seeds"]

    stable_truth = (
        "user_latents.csv.gz", "episodes_truth.csv.gz", "candidate_sets.csv.gz",
        "choices_truth.csv.gz", "trajectories_truth.csv.gz",
    )
    for filename in stable_truth:
        assert _digest(reference / "truth" / filename) == _digest(repeat / "truth" / filename)
        assert _digest(reference / "truth" / filename) == _digest(changed / "truth" / filename)

    assert _digest(reference / "observed" / "observed_events.csv.gz") == _digest(
        repeat / "observed" / "observed_events.csv.gz"
    )
    assert _digest(reference / "truth" / "observation_process.csv.gz") != _digest(
        changed / "truth" / "observation_process.csv.gz"
    )
    observed_names = {path.name for path in (reference / "observed").iterdir()}
    assert observed_names == {"users_observed.csv.gz", "observed_events.csv.gz"}
    assert "random_streams" not in json.loads((reference / "manifest.json").read_text())["validation"]
