from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from geoembeddings.cli import _run_command, build_parser
from geoembeddings.embedding_visualization import fit_pca, project_export, visualize_embeddings
from geoembeddings.representation_schema import COMPONENT_NAMES, EXPORT_SCHEMA_VERSION


def _export(path: Path, *, legacy: bool = False, nonfinite: bool = False,
            duplicate: bool = False, routine: bool = False) -> Path:
    users = np.asarray(["u1", "u2", "u1", "u2"])
    cutoffs = np.asarray(["train", "train", "test", "test"])
    if duplicate:
        users[1], cutoffs[1] = "u1", "train"
    values = np.asarray([[0., 0.], [2., 1.], [1., 1.], [3., 2.]])
    if nonfinite:
        values[0, 0] = np.nan
    if legacy:
        np.savez_compressed(path, user_id=users, cutoff=cutoffs, embedding=values)
        return path
    names = COMPONENT_NAMES + (("routine",) if routine else ())
    payload = dict(user_id=users, cutoff=cutoffs, embedding=values,
        schema_version=np.asarray(EXPORT_SCHEMA_VERSION), component_names=np.asarray(names),
        component_dimensions=np.asarray([2] * len(names)),
        component_persistent=values, component_context=values + 1, component_combined=values,
        model_variant=np.asarray("factorized_pc"), categorical_fields=np.asarray(["service_id"]),
        continuous_fields=np.asarray(["latitude"]), preparation_hash=np.asarray("prep"),
        source_file_names=np.asarray(["observed_events.csv.gz"]), source_hashes=np.asarray(["abc"]),
        train_end=np.asarray("2026-01-01T00:00:00Z"), validation_end=np.asarray("2026-01-02T00:00:00Z"),
        export_cutoffs=np.asarray(["train", "test"]), compatibility=np.asarray("combined alias"))
    if routine:
        payload["component_routine"] = values + 2
    np.savez_compressed(path, **payload)
    return path


def test_pca_is_deterministic_and_fits_only_reference_cutoff(tmp_path: Path) -> None:
    path = _export(tmp_path / "export.npz")
    rows1, metadata1 = project_export(path, reference_cutoff="train", seed=41)
    rows2, metadata2 = project_export(path, reference_cutoff="train", seed=41)
    assert rows1 == rows2
    assert metadata1["models"] == metadata2["models"]
    assert metadata1["fitted_row_identities"] == [["u1", "train"], ["u2", "train"]]
    assert metadata1["models"]["combined"]["mean"] == [1.0, .5]
    direct = fit_pca(np.asarray([[0., 0.], [2., 1.]]))
    assert metadata1["models"]["combined"]["components"] == direct.components.tolist()


def test_component_discovery_and_row_alignment_including_future_routine(tmp_path: Path) -> None:
    rows, metadata = project_export(_export(tmp_path / "export.npz", routine=True))
    assert metadata["component_names"] == ["persistent", "context", "combined", "routine"]
    assert len(rows) == 16
    combined = [row for row in rows if row["component"] == "combined"]
    assert [(row["user_id"], row["cutoff"]) for row in combined] == [
        ("u1", "train"), ("u2", "train"), ("u1", "test"), ("u2", "test")]


def test_malformed_nonfinite_duplicates_and_missing_reference_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        project_export(_export(tmp_path / "nan.npz", nonfinite=True))
    with pytest.raises(ValueError, match="duplicate"):
        project_export(_export(tmp_path / "duplicate.npz", duplicate=True))
    with pytest.raises(ValueError, match="has no rows"):
        project_export(_export(tmp_path / "valid.npz"), reference_cutoff="validation")


def test_legacy_single_vector_compatibility(tmp_path: Path) -> None:
    rows, metadata = project_export(_export(tmp_path / "legacy.npz", legacy=True))
    assert metadata["component_names"] == ["persistent", "context", "combined"]
    assert "legacy single vector" in metadata["export"]["compatibility"]
    assert len(rows) == 12


def test_output_alignment_and_overwrite_protection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("geoembeddings.embedding_visualization._write_plots",
                        lambda rows, output_dir, **kwargs: [])
    output = tmp_path / "visualization"
    result = visualize_embeddings(_export(tmp_path / "export.npz"), output)
    with (output / "projections.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    with np.load(output / "projections.npz") as payload:
        assert payload["user_id"].tolist() == [row["user_id"] for row in csv_rows]
        assert payload["component"].tolist() == [row["component"] for row in csv_rows]
    assert result["rows"] == len(csv_rows)
    with pytest.raises(FileExistsError):
        visualize_embeddings(tmp_path / "export.npz", output)
    visualize_embeddings(tmp_path / "export.npz", output, overwrite=True)


@pytest.mark.integration
def test_cli_uses_only_experiment_export_and_never_resolves_truth(tmp_path: Path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    _export(experiment / "embeddings.npz")
    received: dict[str, Path] = {}
    def fake(source, output, **kwargs):
        received["source"] = Path(source); received["output"] = Path(output)
        return {"status": "complete"}
    monkeypatch.setattr("geoembeddings.embedding_visualization.visualize_embeddings", fake)
    args = build_parser().parse_args(["visualize-embeddings", "--experiment-dir", str(experiment)])
    _run_command(args)
    assert received["source"] == experiment.resolve() / "embeddings.npz"
    assert received["output"] == experiment.resolve() / "visualization" / "learned"
    assert "truth" not in Path("src/geoembeddings/embedding_visualization.py").read_text().lower().replace(
        "never a dataset or truth root", "")
