from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoembeddings.comparison_visualization import render_comparison_scorecard


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _comparison() -> dict:
    geometry = {
        "same_user_cosine": {"train_to_test": {"mean": .97}},
        "different_user_cosine": {"mean": .96},
        "same_minus_different_train_test_cosine": .01,
        "temporal_user_retrieval": {"train_query_test_gallery_top1": .02},
        "test_geometry": {"effective_rank_ratio": .03},
    }
    return {
        "comparison_contract": {"shared_users": 20},
        "persistent_information": {
            "baseline": {"mean_r2": .1}, "learned": {"mean_r2": -.2}},
        "stability_and_distinctiveness": {"baseline": geometry, "learned": geometry},
        "episode_response_comparison": {
            "boundary_change_magnitude": {"baseline": .2, "learned": .1, "learned_minus_baseline": -.1},
            "post_episode_recovery": {"baseline": .3, "learned": .4, "learned_minus_baseline": .1}},
        "R6_R7_robustness_comparison": {"R6_views": [], "R7_views": [{
            "view_id": "zero", "cosine_drift_mean": {"baseline": .0, "learned": -.1, "learned_minus_baseline": -.1},
            "coverage": {"baseline": 0, "learned": 0, "learned_minus_baseline": 0}}]},
        "requirements": {
            "R1_persistent": {"status": "partial", "coverage": 0, "confidence": "low"},
            "R9_ranking": {"status": "unavailable", "coverage": 0}},
    }


def test_scorecard_preserves_unavailable_zero_coverage_negative_delta_and_metadata(tmp_path: Path) -> None:
    source = _write(tmp_path / "comparison.json", _comparison())
    factorized = _write(tmp_path / "factorized.json", {
        "schema_version": "geoembeddings-factorized-comparison/1.0",
        "matched_identity": {"user_mask_sha256": "abc"},
        "decision": "do not advance"})
    first = tmp_path / "first.svg"
    metadata = render_comparison_scorecard(source, first, factorized_path=factorized)
    second = tmp_path / "second.svg"
    metadata_two = render_comparison_scorecard(source, second, factorized_path=factorized)

    svg = first.read_text(encoding="utf-8")
    assert first.read_bytes() == second.read_bytes()
    assert metadata["output_sha256"] == metadata_two["output_sha256"]
    assert '&quot;deterministic&quot;:true' in svg
    assert "T2.7 FAILED GATE · DO NOT ADVANCE" in svg
    assert "COLLAPSE WARNING" in svg
    assert "unavailable · coverage: 0" in svg
    assert "coverage: 0 · confidence: low" in svg
    assert "Δ -0.1000" in svg
    assert "diagnostic controls" in svg
    assert metadata["no_aggregate_winner"] is True


def test_scorecard_rejects_unauthenticated_inputs(tmp_path: Path) -> None:
    bad = _write(tmp_path / "bad.json", {})
    with pytest.raises(ValueError, match="authenticated comparison_contract"):
        render_comparison_scorecard(bad, tmp_path / "bad.svg")

    comparison = _write(tmp_path / "comparison.json", _comparison())
    ranking = _write(tmp_path / "ranking.json", {"authentication": {"status": "failed"}})
    with pytest.raises(ValueError, match="not authenticated"):
        render_comparison_scorecard(comparison, tmp_path / "ranking.svg", ranking_paths=[ranking])
