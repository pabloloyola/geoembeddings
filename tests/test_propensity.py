from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoembeddings.propensity import fit_position_propensities, clipped_inverse_propensity_weights
from geoembeddings.ranking import RankingTrainingBatch, train_frozen_head
from geoembeddings.ranking_pair_evaluation import protected_ranking_metrics


def test_propensity_fit_is_training_only_and_laplace_smoothed() -> None:
    rows = [
        {"request_id": "train", "candidate_position": 1, "is_shown": 1},
        {"request_id": "train", "candidate_position": 1, "is_shown": 0},
        {"request_id": "future", "candidate_position": 1, "is_shown": 1},
    ]
    assert fit_position_propensities(rows, {"train"}, smoothing=1) == {1: .5}
    with pytest.raises(ValueError, match="no training"):
        fit_position_propensities(rows, set())


def test_clipping_weight_distribution_ess_and_threshold_sensitivity() -> None:
    weights, diagnostics = clipped_inverse_propensity_weights([.01, .5, 1.0], minimum=.1, maximum_weight=5)
    assert np.allclose(weights, [5, 2, 1])
    assert diagnostics["clipping_rate"] == pytest.approx(1 / 3)
    assert 0 < diagnostics["effective_sample_size"] <= 3
    _, less_clipped = clipped_inverse_propensity_weights([.01, .5, 1.0], minimum=.01, maximum_weight=100)
    assert less_clipped["clipping_rate"] == 0
    with pytest.raises(ValueError, match="thresholds"):
        clipped_inverse_propensity_weights([.5], minimum=0, maximum_weight=2)


def test_weighted_head_validates_row_aligned_positive_weights() -> None:
    batch = RankingTrainingBatch(("r", "r"), ("a", "b"), np.asarray([[1., 0.], [1., 1.]]), np.asarray([1., 0.]))
    assert np.isfinite(train_frozen_head(batch, iterations=2, sample_weights=np.asarray([2., 1.]))).all()
    with pytest.raises(ValueError, match="row-aligned"):
        train_frozen_head(batch, sample_weights=np.asarray([1.]))


def test_protected_metrics_require_exact_identities_and_probability_schema() -> None:
    truth = pd.DataFrame([
        {"request_id": "r", "poi_id": "a", "utility": 2., "choice_probability": .8},
        {"request_id": "r", "poi_id": "b", "utility": 1., "choice_probability": .2},
    ])
    metrics = protected_ranking_metrics({"r": ["b", "a"]}, truth)
    assert metrics["mean_utility_regret_at_1"] == 1
    assert metrics["choice_probability_brier"] >= 0
    with pytest.raises(ValueError, match="identities mismatch"):
        protected_ranking_metrics({"r": ["a"]}, truth)
