from __future__ import annotations

import numpy as np
import pytest

from geoembeddings.representation_schema import (
    COMPONENT_NAMES,
    EXPORT_SCHEMA_VERSION,
    LEGACY_EXPORT_SCHEMA_VERSION,
    load_embedding_export,
)


def _metadata():
    return {
        "model_variant": np.asarray("single_vector"),
        "categorical_fields": np.asarray(["service_id"]),
        "continuous_fields": np.asarray(["latitude"]),
        "preparation_hash": np.asarray("abc"),
        "source_file_names": np.asarray(["observed_events.csv.gz"]),
        "source_hashes": np.asarray(["def"]),
        "train_end": np.asarray("2026-01-01T00:00:00Z"),
        "validation_end": np.asarray("2026-01-02T00:00:00Z"),
        "export_cutoffs": np.asarray(["train", "test"]),
        "compatibility": np.asarray("embedding aliases component_combined"),
    }


def test_legacy_export_migrates_with_explicit_single_vector_rule(tmp_path):
    path = tmp_path / "legacy.npz"
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez_compressed(path, user_id=np.array(["u1", "u2"]),
                        cutoff=np.array(["test", "test"]), embedding=vectors)

    loaded = load_embedding_export(path)
    assert loaded.schema_version == LEGACY_EXPORT_SCHEMA_VERSION
    assert np.array_equal(loaded.components["persistent"], vectors)
    assert np.array_equal(loaded.components["combined"], vectors)
    assert np.count_nonzero(loaded.components["context"]) == 0


def test_versioned_component_export_is_unambiguous_to_evaluator_reader(tmp_path):
    path = tmp_path / "components.npz"
    persistent = np.ones((2, 2), dtype=np.float32)
    context = np.full((2, 4), 2, dtype=np.float32)
    combined = np.full((2, 3), 3, dtype=np.float32)
    np.savez_compressed(
        path, user_id=np.array(["u1", "u2"]), cutoff=np.array(["train", "test"]),
        embedding=combined, schema_version=np.asarray(EXPORT_SCHEMA_VERSION),
        component_names=np.asarray(COMPONENT_NAMES),
        component_dimensions=np.asarray([2, 4, 3]),
        component_persistent=persistent, component_context=context,
        component_combined=combined,
        **_metadata(),
    )

    loaded = load_embedding_export(path)
    assert loaded.schema_version == EXPORT_SCHEMA_VERSION
    assert loaded.components["persistent"].shape == (2, 2)
    assert loaded.components["context"].shape == (2, 4)
    assert np.array_equal(loaded.embedding, combined)

    # The comparison evaluator consumes the same shared reader and selects the
    # documented combined component, while retaining legacy key semantics.
    from geoembeddings.comparison import _load_embeddings

    evaluated = _load_embeddings(path, "component")
    assert np.array_equal(evaluated[("u2", "test")], combined[1])


def test_comparison_evaluator_reads_legacy_export_without_ambiguity(tmp_path):
    from geoembeddings.comparison import _load_embeddings

    path = tmp_path / "legacy-evaluator.npz"
    vector = np.asarray([[1.0, 2.0]])
    np.savez_compressed(path, user_id=np.asarray(["u"]),
                        cutoff=np.asarray(["test"]), embedding=vector)
    evaluated = _load_embeddings(path, "legacy")
    assert np.array_equal(evaluated[("u", "test")], vector[0])


@pytest.mark.parametrize("mutation", ["names", "dimension", "alias", "nonfinite"])
def test_component_reader_rejects_schema_mismatches(tmp_path, mutation):
    path = tmp_path / "bad.npz"
    names = np.asarray(COMPONENT_NAMES if mutation != "names" else ("context", "persistent", "combined"))
    dimensions = np.asarray([2, 2, 2] if mutation != "dimension" else [2, 9, 2])
    persistent = np.ones((1, 2))
    context = np.ones((1, 2))
    if mutation == "nonfinite":
        context[0, 0] = np.nan
    combined = np.ones((1, 2))
    alias = combined.copy() if mutation != "alias" else np.zeros_like(combined)
    np.savez_compressed(
        path, user_id=np.array(["u"]), cutoff=np.array(["test"]), embedding=alias,
        schema_version=np.asarray(EXPORT_SCHEMA_VERSION), component_names=names,
        component_dimensions=dimensions, component_persistent=persistent,
        component_context=context, component_combined=combined,
        **_metadata(),
    )
    with pytest.raises(ValueError):
        load_embedding_export(path)
