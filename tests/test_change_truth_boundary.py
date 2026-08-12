from __future__ import annotations

import inspect

from geoembeddings import baseline, evaluation, export, prepare, training


def test_change_truth_is_only_named_at_protected_evaluator_boundary() -> None:
    protected_name = "change_points_truth.csv.gz"
    assert protected_name in inspect.getsource(evaluation.evaluate_change)
    for model_facing_module in (prepare, baseline, training, export):
        assert protected_name not in inspect.getsource(model_facing_module)
