from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoembeddings.baseline import export_statistical_baseline
from geoembeddings.config import load_config
from geoembeddings.export import export_embeddings
from geoembeddings.layout import ExperimentLayout
from geoembeddings.pair_evaluation import evaluate_pair, match_pair_keys, representation_metrics
from geoembeddings.prepare import prepare_data
from geoembeddings.simulate_pair import simulate_pair
from geoembeddings.training import train_model


def test_pair_matching_reports_coverage_and_exclusions() -> None:
    vector = np.asarray([1.0, 0.0])
    reference = {("u1", "test"): vector, ("u2", "test"): vector}
    intervention = {("u1", "test"): vector, ("u3", "test"): vector}
    keys, coverage = match_pair_keys(reference, intervention)
    assert keys == [("u1", "test")]
    assert coverage == {
        "reference_rows": 2, "intervention_rows": 2, "matched_rows": 1,
        "reference_only_rows": 1, "intervention_only_rows": 1,
        "match_fraction_reference": .5, "match_fraction_intervention": .5,
        "reference_only_samples": [["u2", "test"]],
        "intervention_only_samples": [["u3", "test"]],
    }


def test_identical_pair_metrics_have_zero_drift_and_perfect_retrieval() -> None:
    users = [f"u{i}" for i in range(12)]
    source = {(user, "test"): np.asarray([i + 1.0, (i + 1.0) ** 2, 1.0])
              for i, user in enumerate(users)}
    labels = pd.DataFrame({"user_id": users,
        "price_sensitivity": np.arange(12, dtype=float),
        "pref_cafe": np.arange(12, dtype=float)}).set_index("user_id")
    result = representation_metrics(source, source, labels, train_fraction=.7, alpha=1.0)
    assert result["embedding_drift"]["mean_cosine_distance"] == pytest.approx(0.0, abs=1e-12)
    assert result["retrieval"]["cross_run_user_top1"] == 1.0
    assert result["effective_rank"]["reference"] == result["effective_rank"]["intervention"]


def test_pair_evaluator_is_only_modeling_module_that_names_pair_truth() -> None:
    root = Path("src/geoembeddings")
    forbidden = ["model.py", "data.py", "training.py", "baseline.py", "export.py", "prepare.py"]
    for name in forbidden:
        text = (root / name).read_text(encoding="utf-8")
        assert "pair_manifest" not in text
        assert "pair_integrity" not in text
        assert "intervention_type" not in text


@pytest.mark.integration
def test_full_paired_run_evaluation(tmp_path: Path) -> None:
    pair_result = simulate_pair(Path("configs/simulation/kanto_v1.yaml"),
        tmp_path / "reference", tmp_path / "intervention", tmp_path / "pair",
        intervention="observation", users=10, days=2, seed=20260811)
    config = load_config(Path("configs/embedding/single_vector.yaml"))
    config["training"]["epochs"] = 1
    experiments = []
    for side in ("reference", "intervention"):
        experiment = ExperimentLayout.from_path(tmp_path / f"{side}-experiment")
        observed = tmp_path / side / "observed"
        prepare_data(observed, experiment.prepared, config)
        export_statistical_baseline(observed, experiment.prepared,
            experiment.baseline_embeddings, config)
        train_model(observed, experiment.prepared, experiment.model, config)
        export_embeddings(observed, experiment.prepared, experiment.checkpoint,
            experiment.embeddings, config)
        experiments.append(experiment.root)
    report = evaluate_pair(pair_result["pair_manifest"], experiments, experiments, config)
    assert report["schema_version"] == "geoembeddings-counterfactual-comparison/1.0"
    assert report["requirements"] == {"R5": "executable", "R7": "executable"}
    assert (tmp_path / "pair" / "counterfactual_comparison.json").is_file()
    assert (tmp_path / "pair" / "counterfactual_comparison.md").is_file()
