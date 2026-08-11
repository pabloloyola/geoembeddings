from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoembeddings.status_reconciliation import derive_status, reconcile_artifact_index


def _complete_comparison() -> dict:
    geometry = {
        "baseline": {},
        "learned": {
            "test_geometry": {"effective_rank_ratio": 0.4},
            "same_minus_different_train_test_cosine": 0.2,
            "temporal_user_retrieval": {"train_query_test_gallery_top1": 0.5},
        },
    }
    return {
        "comparison_contract": {"shared_users": 80},
        "persistent_information": {"baseline": {"mean_r2": 0.1}, "learned": {"mean_r2": 0.2}},
        "preference_beyond_geography_and_activity": {
            "baseline": {"incremental_mean_r2_over_nuisance": 0.01},
            "learned": {"incremental_mean_r2_over_nuisance": 0.04},
            "learned_minus_baseline_incremental_r2": 0.03,
        },
        "stability_and_distinctiveness": geometry,
        "episode_response_comparison": {
            "boundary_change": {"learned_minus_baseline": 0.02}
        },
        "R6_R7_robustness_comparison": {
            "R6_views": [{"matched_rows": 70, "coverage": 0.875}],
            "R7_views": [{"matched_rows": 75, "coverage": 0.9375}],
        },
        "common_future_event_probes": {
            "protocol": "held-out",
            "next_service": {
                "known_label_coverage": 0.9,
                "learned_minus_baseline_accuracy": 0.1,
            },
        },
    }


def test_status_derivation_selects_factorize_without_compositing_axes() -> None:
    result = derive_status(
        {"embedding_comparison_json": _complete_comparison()},
        {"users_without_observed_events": 20},
    )
    assert result.action == "factorize"
    assert len(result.axes) == 6
    assert all(axis.coverage and axis.missingness for axis in result.axes)


def test_missing_axis_selects_finish_evaluator_gate() -> None:
    comparison = _complete_comparison()
    del comparison["episode_response_comparison"]
    result = derive_status(
        {"embedding_comparison_json": comparison},
        {"users_without_observed_events": 20},
    )
    assert result.action == "finish_evaluator_gate"
    episode = next(axis for axis in result.axes if axis.name == "episode response")
    assert episode.conclusion == "pending"
    assert "missing required axis" in episode.missingness


def test_reconciliation_rejects_comparability_mismatch_before_reading_reports(
    tmp_path: Path,
) -> None:
    identity = {
        "observed_source_hashes": {"events": "a"},
        "cutoffs": {"train_end": "a", "validation_end": "b"},
        "categorical_field_order": ["service_id"],
        "continuous_field_order": ["latitude"],
        "user_set_sha256": "users",
        "preparation_metadata_sha256": "prepared",
    }
    index = {
        "schema_version": "geoembeddings-evidence-index/1.0",
        "task_id": "T0.2",
        "evidence_status": "complete",
        "comparability_audit": {
            "result": "passed",
            "blocking_reasons": [],
            "baseline_and_learned_source_hashes_match": True,
            "cutoffs_match": True,
            "categorical_field_order_matches": True,
            "continuous_field_order_matches": True,
            "users_match": True,
            "dense_users_and_timestamps_match": True,
            "preparation_contract_matches": True,
            "robustness_specifications_and_masks_match": True,
        },
        "evidence_identity": {
            "baseline": identity,
            "learned": {**identity, "cutoffs": {"train_end": "x", "validation_end": "b"}},
        },
        "required_artifacts": {},
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="cutoffs mismatch"):
        reconcile_artifact_index(path, repository_root=tmp_path)
