from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from geoembeddings.context_contrastive import (
    FrozenContextTripletDataset,
    ContextProjectionHead,
    context_infonce_loss,
    select_epoch_triplets,
)
from geoembeddings.context_pair_preflight import build_context_pairs


def _events() -> pd.DataFrame:
    rows = []
    for timestamp, count in (
        ("2026-01-01T00:00:00+00:00", 2),
        ("2026-01-01T01:00:00+00:00", 1),
        ("2026-01-01T02:00:00+00:00", 1),
        ("2026-01-01T03:00:00+00:00", 1),
        ("2026-01-01T08:00:00+00:00", 1),
        ("2026-01-01T09:00:00+00:00", 1),
        ("2026-01-02T00:00:00+00:00", 1),
    ):
        for index in range(count):
            rows.append({"user_id": "u1", "timestamp": timestamp, "value": f"{timestamp}-{index}"})
    return pd.DataFrame(rows)


def _manifest(events: pd.DataFrame, *, drop_negative: bool = False) -> dict:
    pairs, diagnostics = build_context_pairs(
        events,
        train_end="2026-01-02T01:00:00+00:00",
        min_history_events=2,
        session_gap_hours=6,
        min_intervening_groups_for_positive=1,
        positive_pairs_per_anchor=1,
        negative_pairs_per_anchor=1,
        seed=7,
        positive_local_day_timezone="Asia/Tokyo",
        positive_same_local_day=True,
    )
    if drop_negative:
        positive_anchor = next(pair for pair in pairs if pair["relation"] == "positive")["anchor_group_id"]
        pairs = [pair for pair in pairs if not (
            pair["relation"] == "negative" and pair["anchor_group_id"] == positive_anchor
        )]
    return {
        "schema_version": "geoembeddings-context-pair-manifest/1.0",
        "source_authentication": {
            "truth_files_opened": False,
            "observed_file_hashes": {"users_observed.csv.gz": "users", "observed_events.csv.gz": "events"},
        },
        "preparation_authentication": {
            "observed_source_hashes": {"users_observed.csv.gz": "users", "observed_events.csv.gz": "events"},
        },
        "pair_configuration": {
            "session_gap_hours": 6.0,
            "min_intervening_groups_for_positive": 1,
            "positive_local_day_timezone": "Asia/Tokyo",
            "positive_same_local_day": True,
        },
        "coverage": diagnostics["coverage"],
        "pairs": pairs,
    }


def _base(events: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        events=events,
        metadata={
            "source_files": {"users_observed.csv.gz": "users", "observed_events.csv.gz": "events"},
            "train_end": "2026-01-02T01:00:00+00:00",
        },
        encoded_categories=np.arange(len(events), dtype=np.int64).reshape(-1, 1),
        continuous=np.arange(len(events), dtype=np.float32).reshape(-1, 1),
    )


def test_exact_detached_context_blocks_encoder_gradients_but_not_projection_head() -> None:
    torch.manual_seed(3)
    head = ContextProjectionHead(4, 4)
    anchor = torch.randn(3, 4, requires_grad=True)
    positive = torch.randn(3, 4, requires_grad=True)
    negative = torch.randn(3, 1, 4, requires_grad=True)
    loss = context_infonce_loss(anchor, positive, negative, head, temperature=.2, detach_context=False)
    loss.backward()
    assert anchor.grad is not None and torch.count_nonzero(anchor.grad) > 0
    assert any(parameter.grad is not None for parameter in head.parameters())

    detached_head = ContextProjectionHead(4, 4)
    detached_head.load_state_dict(copy.deepcopy(head.state_dict()))
    detached_anchor = anchor.detach().clone().requires_grad_(True)
    detached_positive = positive.detach().clone().requires_grad_(True)
    detached_negative = negative.detach().clone().requires_grad_(True)
    detached_loss = context_infonce_loss(
        detached_anchor, detached_positive, detached_negative, detached_head,
        temperature=.2, detach_context=True,
    )
    detached_loss.backward()
    assert detached_anchor.grad is None
    assert detached_positive.grad is None
    assert detached_negative.grad is None
    assert all(parameter.grad is not None for parameter in detached_head.parameters())


def test_user_balanced_selection_is_deterministic_and_capped() -> None:
    from geoembeddings.context_contrastive import ContextTriplet
    triplets = [
        ContextTriplet(user, f"p-{user}-{index}", f"a-{index}", f"b-{index}", (f"n-{index}",), "t", "t", ("t",))
        for user in ("u1", "u2") for index in range(5)
    ]
    first = select_epoch_triplets(triplets, max_positive_anchors_per_user=2, seed=11, epoch=1)
    second = select_epoch_triplets(triplets, max_positive_anchors_per_user=2, seed=11, epoch=1)
    assert first == second
    assert {triplet.user_id for triplet in first} == {"u1", "u2"}
    assert all(sum(triplet.user_id == user for triplet in first) <= 2 for user in ("u1", "u2"))


def test_triplet_coverage_atomic_prefix_and_no_truth_dependency(tmp_path) -> None:
    events = _events()
    manifest = _manifest(events)
    path = tmp_path / "context_pair_manifest.json"
    path.write_text(__import__("json").dumps(manifest))
    dataset = FrozenContextTripletDataset(
        _base(events), path, negative_pairs_per_anchor=1, max_sequence_length=64,
    )
    assert dataset.joint_coverage_report["joint_anchor_count"] == dataset.joint_coverage_report["positive_anchor_count"]
    item = dataset[0]
    positive = next(pair for pair in manifest["pairs"] if pair["pair_id"] == item["positive_pair_id"])
    assert len(item["anchor_categorical"]) == positive["anchor_history_event_count"]
    assert not (tmp_path / "truth").exists()

    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(__import__("json").dumps(_manifest(events, drop_negative=True)))
    incomplete = FrozenContextTripletDataset(
        _base(events), incomplete_path, negative_pairs_per_anchor=1, max_sequence_length=64,
    )
    assert incomplete.joint_coverage_report["joint_anchor_count"] < incomplete.joint_coverage_report["positive_anchor_count"]


def test_post_cutoff_manifest_pair_is_rejected(tmp_path) -> None:
    events = _events()
    manifest = _manifest(events)
    positive = next(pair for pair in manifest["pairs"] if pair["relation"] == "positive")
    positive["paired_timestamp"] = "2026-01-02T02:00:00+00:00"
    path = tmp_path / "post_cutoff.json"
    path.write_text(json.dumps(manifest))
    try:
        FrozenContextTripletDataset(_base(events), path, negative_pairs_per_anchor=1, max_sequence_length=64)
    except ValueError as exc:
        assert "post-cutoff" in str(exc)
    else:
        raise AssertionError("post-cutoff pair was accepted")


def test_candidate_and_detached_control_configs_are_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads(json.dumps(yaml.safe_load(
        (root / "configs/embedding/context_session_contrastive_candidate.yaml").read_text()
    )))
    control = yaml.safe_load(
        (root / "configs/embedding/context_session_contrastive_detached_control.yaml").read_text()
    )
    assert candidate["context_contrastive"].pop("mode") == "candidate"
    assert control["context_contrastive"].pop("mode") == "detached_control"
    assert candidate == control
