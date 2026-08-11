from __future__ import annotations

from pathlib import Path

from geoembeddings.comparison import compare_embeddings
from geoembeddings.config import load_config


def test_identical_exports_have_zero_comparison_deltas(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "smoke" / "experiment"
    run = root / "smoke" / "run"
    embedding_path = experiment / "statistical_baseline.npz"

    report = compare_embeddings(
        run / "observed",
        run / "truth",
        experiment / "prepared",
        experiment / "prepared",
        embedding_path,
        embedding_path,
        tmp_path,
        load_config(root / "configs" / "embedding" / "single_vector.yaml"),
    )

    assert report["persistent_information"]["learned_minus_baseline_mean_r2"] == 0.0
    assert (
        report["stability_and_distinctiveness"]["baseline"]
        == report["stability_and_distinctiveness"]["learned"]
    )
    assert (tmp_path / "embedding_comparison.json").is_file()
    assert (tmp_path / "embedding_comparison.md").is_file()
