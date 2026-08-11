from __future__ import annotations

import numpy as np

from geoembeddings.training import next_event_classification_diagnostics


def diagnostics(train: list[int], truth: list[int], prediction: list[int]):
    return next_event_classification_diagnostics(
        train_targets=np.asarray(train),
        truths=np.asarray(truth),
        predictions=np.asarray(prediction),
        class_count=4,
        unknown_label_id=0,
    )


def test_imbalanced_train_majority_and_balance_metrics_are_finite() -> None:
    report = diagnostics([1, 1, 1, 2], [1, 2, 2], [1, 2, 1])

    assert report["majority_label_id"] == 1
    assert report["naive"]["known_label_accuracy"] == 1 / 3
    assert report["learned"]["balanced_accuracy"] == 0.75
    assert report["empty_evaluation_class_count"] == 1
    assert all(
        np.isfinite(value)
        for model in ("learned", "naive")
        for key, value in report[model].items()
        if key != "per_class"
    )


def test_unknown_labels_are_coverage_failures_not_majority_successes() -> None:
    report = diagnostics([1, 1, 2], [1, 0, 0], [1, 0, 2])

    assert report["known_label_count"] == 1
    assert report["unknown_label_count"] == 2
    assert report["known_label_coverage"] == 1 / 3
    assert report["learned"]["known_label_accuracy"] == 1.0
    assert report["learned"]["coverage_aware_accuracy"] == 1 / 3


def test_zero_known_coverage_and_empty_train_distribution_are_explicit() -> None:
    report = diagnostics([0, 0], [0, 0], [0, 1])

    assert report["status"] == "zero_known_label_coverage"
    assert report["majority_label_id"] is None
    assert report["known_label_coverage"] == 0.0
    for model in ("learned", "naive"):
        assert report[model]["known_label_accuracy"] == 0.0
        assert report[model]["macro_f1"] == 0.0


def test_naive_fit_does_not_use_evaluation_frequencies() -> None:
    train = [1, 1, 1, 2]
    first = diagnostics(train, [1, 2], [1, 2])
    validation_and_test_frequency_reversed = diagnostics(train, [2] * 20 + [1], [2] * 21)

    assert first["fit_split"] == "train"
    assert first["majority_label_id"] == 1
    assert validation_and_test_frequency_reversed["majority_label_id"] == 1
    assert (
        first["train_class_counts"]
        == validation_and_test_frequency_reversed["train_class_counts"]
    )
