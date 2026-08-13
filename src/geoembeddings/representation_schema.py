from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPORT_SCHEMA_VERSION = "geoembeddings-component-export/2.0"
CHECKPOINT_SCHEMA_VERSION = "geoembeddings-component-checkpoint/2.0"
LEGACY_EXPORT_SCHEMA_VERSION = "geoembeddings-single-vector-export/1.0"
COMPONENT_NAMES = ("persistent", "context", "combined")


@dataclass(frozen=True)
class LoadedEmbeddingExport:
    arrays: dict[str, np.ndarray]
    components: dict[str, np.ndarray]
    schema_version: str
    compatibility: str

    @property
    def embedding(self) -> np.ndarray:
        return self.components["combined"]


def checkpoint_schema(
    *, model_variant: str, component_dimensions: dict[str, int],
    categorical_fields: list[str], continuous_fields: list[str],
    preparation_hash: str, source_files: dict[str, str], train_end: str,
    validation_end: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_variant": model_variant,
        "component_names": list(COMPONENT_NAMES),
        "component_dimensions": component_dimensions,
        "categorical_fields": categorical_fields,
        "continuous_fields": continuous_fields,
        "preparation_hash": preparation_hash,
        "source_files": source_files,
        "cutoffs": {"train_end": train_end, "validation_end": validation_end},
        "compatibility": {"legacy_embedding_component": "combined"},
    }


def load_embedding_export(path: str | Path, *, dense: bool = False) -> LoadedEmbeddingExport:
    """Read legacy or component NPZs under an explicit, ambiguity-free rule."""
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {"user_id", "timestamp" if dense else "cutoff"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Embedding export is missing arrays: {sorted(missing)}")

    raw_version = arrays.get("schema_version")
    if raw_version is None:
        if "embedding" not in arrays:
            raise ValueError("Unversioned export has no legacy embedding array")
        components = {"persistent": arrays["embedding"],
                      "context": np.zeros_like(arrays["embedding"]),
                      "combined": arrays["embedding"]}
        version = LEGACY_EXPORT_SCHEMA_VERSION
        compatibility = "legacy single vector mapped to persistent/combined; context=zeros"
    else:
        version = str(np.asarray(raw_version).item())
        if version != EXPORT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported component export schema: {version!r}")
        metadata_required = {
            "model_variant", "categorical_fields", "continuous_fields",
            "preparation_hash", "source_file_names", "source_hashes",
            "train_end", "validation_end", "export_cutoffs", "compatibility",
        }
        metadata_missing = metadata_required - arrays.keys()
        if metadata_missing:
            raise ValueError(f"Component export metadata is missing: {sorted(metadata_missing)}")
        if len(arrays["source_file_names"]) != len(arrays["source_hashes"]):
            raise ValueError("Source file names and hashes are not aligned")
        names = tuple(arrays.get("component_names", np.asarray([], dtype=str)).astype(str))
        if names != COMPONENT_NAMES:
            raise ValueError(f"Component name/order mismatch: {names!r}")
        dimensions = arrays.get("component_dimensions")
        if dimensions is None or dimensions.shape != (len(names),):
            raise ValueError("Component dimensions are missing or malformed")
        components = {}
        for name, dimension in zip(names, dimensions.astype(int)):
            key = f"component_{name}"
            if key not in arrays:
                raise ValueError(f"Component export is missing {key!r}")
            value = arrays[key]
            if value.ndim != 2 or value.shape[1] != dimension:
                raise ValueError(f"Component {name!r} dimension mismatch")
            components[name] = value
        if "embedding" not in arrays or not np.array_equal(arrays["embedding"], components["combined"]):
            raise ValueError("Legacy embedding alias does not match combined component")
        compatibility = "versioned components; embedding aliases combined"

    row_count = len(arrays["user_id"])
    for name, value in components.items():
        if value.ndim != 2:
            raise ValueError(f"Component {name!r} must be 2-D")
        if len(value) != row_count:
            raise ValueError(f"Component {name!r} is not row-aligned")
        if not np.isfinite(value).all():
            raise ValueError(f"Component {name!r} contains non-finite values")
    return LoadedEmbeddingExport(arrays, components, version, compatibility)
