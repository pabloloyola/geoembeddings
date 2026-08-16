from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from geoembeddings.context_pair_preflight import (
    build_context_pairs,
    run_context_pair_preflight,
    validate_context_pair_manifest,
)
from geoembeddings.io import sha256_file


def _events() -> pd.DataFrame:
    rows = []
    for timestamp, count in (("2026-01-01T00:00:00+00:00", 2), ("2026-01-01T01:00:00+00:00", 1),
                             ("2026-01-01T02:00:00+00:00", 1), ("2026-01-01T03:00:00+00:00", 1),
                             ("2026-01-01T08:00:00+00:00", 1), ("2026-01-01T09:00:00+00:00", 1),
                             ("2026-01-02T00:00:00+00:00", 1)):
        for index in range(count):
            rows.append({"user_id": "u1", "timestamp": timestamp, "value": f"{timestamp}-{index}"})
    return pd.DataFrame(rows)


def _pairs(**overrides):
    values = {
        "train_end": "2026-01-01T23:00:00+00:00",
        "min_history_events": 2,
        "session_gap_hours": 4.0,
        "min_intervening_groups_for_positive": 1,
        "positive_pairs_per_anchor": 1,
        "negative_pairs_per_anchor": 1,
        "seed": 7,
    }
    values.update(overrides)
    return build_context_pairs(_events(), **values)


def test_same_timestamp_records_are_one_atomic_group_and_not_visible() -> None:
    pairs, diagnostics = _pairs()
    assert diagnostics["coverage"]["timestamp_group_count"] == 6
    assert diagnostics["coverage"]["positive_pair_count"] == 1
    positive = next(pair for pair in pairs if pair["relation"] == "positive")
    assert positive["anchor_history_event_count"] == 2
    assert positive["intervening_group_count"] == 1


def test_train_cutoff_excludes_later_groups_and_pairs() -> None:
    pairs, diagnostics = _pairs(train_end="2026-01-01T08:00:00+00:00")
    assert diagnostics["exclusions"]["by_reason"]["cross_cutoff"] == 2
    assert all(pair["paired_timestamp"] <= "2026-01-01T08:00:00+00:00" for pair in pairs)


def test_duplicate_observed_records_are_rejected() -> None:
    events = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate observed"):
        build_context_pairs(events, train_end="2026-01-01T23:00:00+00:00", min_history_events=2,
                            session_gap_hours=4, min_intervening_groups_for_positive=1,
                            positive_pairs_per_anchor=1, negative_pairs_per_anchor=1, seed=7)


def test_duplicate_and_conflicting_pair_identities_are_rejected() -> None:
    pairs, _ = _pairs()
    manifest = {"schema_version": "geoembeddings-context-pair-manifest/1.0",
                "source_authentication": {"truth_files_opened": False},
                "preparation_authentication": {}, "pair_configuration": {}, "pairs": pairs}
    duplicate = copy.deepcopy(manifest)
    duplicate["pairs"].append(copy.deepcopy(pairs[0]))
    with pytest.raises(ValueError, match="duplicate context pair"):
        validate_context_pair_manifest(duplicate)
    conflict = copy.deepcopy(manifest)
    other = copy.deepcopy(pairs[0])
    other["relation"] = "negative" if other["relation"] == "positive" else "positive"
    if other["relation"] == "negative":
        other["paired_session_id"] = "conflicting-session"
    else:
        other["paired_session_id"] = other["anchor_session_id"]
    # Recompute the identity so the failure is specifically a relation conflict.
    from geoembeddings.context_pair_preflight import _pair_id, _pair_key
    other["pair_id"] = _pair_id(other["relation"], _pair_key(other))
    conflict["pairs"].append(other)
    with pytest.raises(ValueError, match="conflicting context pair"):
        validate_context_pair_manifest(conflict)


def test_pair_sampling_is_deterministic_and_does_not_use_canonical_event_id() -> None:
    first, first_diagnostics = _pairs(seed=91)
    second, second_diagnostics = _pairs(seed=91)
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert all("canonical_event_id" not in pair for pair in first)


def _write_observed_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    run = root / "run"
    observed = run / "observed"
    observed.mkdir(parents=True)
    users = pd.DataFrame([{
        "user_id": "u1", "age_group": "adult", "household_type": "single",
        "home_prefecture": "Tokyo", "home_region_id": "tokyo_core", "geo_split": "development",
    }])
    events = _events().copy()
    for column, value in {
        "service_id": "location", "action_type": "view", "observation_mode": "passive",
        "object_id": "x", "object_category": "other", "region_id": "tokyo_core",
        "prefecture": "Tokyo", "latitude": 35.0, "longitude": 139.0,
        "geohash_5": "abcde", "geohash_7": "abcdefg", "location_accuracy_m": 10.0,
        "session_id": "raw-session",
    }.items():
        events[column] = value
    for frame, path in ((users, observed / "users_observed.csv.gz"), (events, observed / "observed_events.csv.gz")):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    (run / "manifest.json").write_text(json.dumps({
        "dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"},
        "seed": 9, "users": 1, "days": 2, "simulator_version": "test",
        "config_sha256": "config",
    }))
    experiment = root / "experiment"
    prepared = experiment / "prepared"
    prepared.mkdir(parents=True)
    config = {
        "seed": 1,
        "data": {"categorical_fields": [], "continuous_fields": [], "include_object_id": False,
                 "max_sequence_length": 64, "min_history_events": 2, "train_fraction": .7,
                 "validation_fraction": .15,
                 "train_end": "2026-01-01T23:00:00+00:00",
                 "validation_end": "2026-01-02T23:00:00+00:00"},
        "model": {"variant": "two_timescale_pc"}, "objectives": {"next": 1.0},
        "training": {}, "evaluation": {},
    }
    (prepared / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "preparation_schema_version": "geoembeddings-preparation/1.0",
        "run_dir": str(run.resolve()), "observed_dir": str(observed.resolve()),
        "dataset_contract": {"name": "geoembeddings-dataset", "version": "1.0"},
        "source_files": {"users_observed.csv.gz": sha256_file(observed / "users_observed.csv.gz"),
                          "observed_events.csv.gz": sha256_file(observed / "observed_events.csv.gz")},
        "rows": {"users": 1, "events": len(events)}, "users_with_events": 1,
        "train_end": "2026-01-01T23:00:00+00:00", "validation_end": "2026-01-02T23:00:00+00:00",
        "target_events_by_split": {"train": 7, "validation": 0, "test": 0},
    }
    (prepared / "prepared_metadata.json").write_text(json.dumps(metadata))
    pair_config = root / "pair.yaml"
    pair_config.write_text(yaml.safe_dump({
        "schema_version": "geoembeddings-context-session-preflight/1.0",
        "pairing": {"session_gap_hours": 4.0, "min_intervening_groups_for_positive": 1,
                    "positive_pairs_per_anchor": 1, "negative_pairs_per_anchor": 1, "seed": 7},
    }))
    embedding_config = root / "embedding.yaml"
    embedding_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return run, experiment, pair_config, embedding_config


def test_preflight_has_no_truth_directory_dependency_and_publishes_immutable_outputs(tmp_path: Path) -> None:
    run, experiment, pair_config, embedding_config = _write_observed_fixture(tmp_path)
    output = tmp_path / "preflight"
    result = run_context_pair_preflight(run, experiment, pair_config, embedding_config, output)
    assert result["status"] == "passed"
    assert (output / "context_pair_manifest.json").is_file()
    assert (output / "context_pair_preflight.json").is_file()
    assert not (run / "truth").exists()
    with pytest.raises(FileExistsError, match="immutable"):
        run_context_pair_preflight(run, experiment, pair_config, embedding_config, output)
