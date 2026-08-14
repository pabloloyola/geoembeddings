"""Observed-contract-only projections of frozen embedding exports.

This module deliberately accepts an export path, never a dataset or truth root.
Projection pictures are exploratory diagnostics rather than evaluation evidence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file
from .representation_schema import LoadedEmbeddingExport, load_embedding_export

PROJECTION_SCHEMA_VERSION = "geoembeddings-embedding-visualization/1.0"


@dataclass(frozen=True)
class FittedProjection:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale) @ self.components.T


def fit_pca(values: np.ndarray, *, normalization: str = "standard") -> FittedProjection:
    """Fit a deterministic, sign-canonicalized two-dimensional PCA."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("PCA reference values must be a non-empty finite 2-D array")
    if normalization not in {"standard", "center", "none"}:
        raise ValueError("normalization must be standard, center, or none")
    mean = values.mean(axis=0) if normalization != "none" else np.zeros(values.shape[1])
    scale = values.std(axis=0) if normalization == "standard" else np.ones(values.shape[1])
    scale[scale == 0] = 1.0
    centered = (values - mean) / scale
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    if len(components) < 2:
        components = np.vstack([components, np.zeros((2 - len(components), values.shape[1]))])
    variance = singular**2
    ratios = variance[:2] / variance.sum() if variance.sum() else np.zeros(min(2, len(variance)))
    ratios = np.pad(ratios, (0, 2 - len(ratios)))
    return FittedProjection(mean, scale, components, ratios)


def _identities(export: LoadedEmbeddingExport, *, dense: bool) -> tuple[np.ndarray, np.ndarray]:
    users = export.arrays["user_id"].astype(str)
    times = export.arrays["timestamp" if dense else "cutoff"].astype(str)
    if users.ndim != 1 or times.ndim != 1 or len(users) != len(times):
        raise ValueError("user and cutoff/timestamp identities must be aligned one-dimensional arrays")
    identities = list(zip(users.tolist(), times.tolist()))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate user/cutoff identities are not permitted")
    return users, times


def _json_array(value: np.ndarray) -> list[Any]:
    return np.asarray(value).tolist()


def project_export(export_path: str | Path, *, dense: bool = False,
                   reference_cutoff: str | None = None, normalization: str = "standard",
                   reducer: str = "pca", seed: int = 0,
                   umap_neighbors: int = 15, umap_min_dist: float = 0.1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_path = Path(export_path).resolve()
    export = load_embedding_export(export_path, dense=dense)
    users, cutoffs = _identities(export, dense=dense)
    selected = reference_cutoff or (min(cutoffs.tolist()) if dense else "train")
    reference_mask = cutoffs == selected
    if not reference_mask.any():
        raise ValueError(f"reference cutoff/timestamp {selected!r} has no rows")
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for name, values in export.components.items():
        if not np.isfinite(values).all():
            raise ValueError(f"component {name!r} contains non-finite values")
        reference = np.asarray(values[reference_mask], dtype=np.float64)
        if reducer == "pca":
            fitted = fit_pca(reference, normalization=normalization)
            points = fitted.transform(values)
            model = {"mean": _json_array(fitted.mean), "scale": _json_array(fitted.scale),
                     "components": _json_array(fitted.components),
                     "explained_variance_ratio": _json_array(fitted.explained_variance_ratio)}
        elif reducer == "umap":
            try:
                import umap
            except ImportError as error:
                raise RuntimeError("UMAP requires the 'viz' extra: uv sync --extra viz") from error
            fitted = fit_pca(reference, normalization=normalization)
            normalized_reference = (reference - fitted.mean) / fitted.scale
            reducer_model = umap.UMAP(n_components=2, n_neighbors=umap_neighbors,
                min_dist=umap_min_dist, random_state=seed, transform_seed=seed)
            reducer_model.fit(normalized_reference)
            points = reducer_model.transform((values - fitted.mean) / fitted.scale)
            model = {"mean": _json_array(fitted.mean), "scale": _json_array(fitted.scale),
                     "hyperparameters": {"n_neighbors": umap_neighbors, "min_dist": umap_min_dist,
                                         "random_state": seed, "transform_seed": seed},
                     "warning": "UMAP neighborhood geometry and apparent clusters are exploratory, not requirement evidence."}
        else:
            raise ValueError("reducer must be pca or umap")
        models[name] = model
        for user, cutoff, point in zip(users, cutoffs, points):
            rows.append({"user_id": str(user), "timestamp" if dense else "cutoff": str(cutoff),
                         "component": name, "x": float(point[0]), "y": float(point[1])})
    source_names = export.arrays.get("source_file_names", np.asarray([], dtype=str)).astype(str)
    source_hashes = export.arrays.get("source_hashes", np.asarray([], dtype=str)).astype(str)
    metadata = {"schema_version": PROJECTION_SCHEMA_VERSION, "reducer": reducer,
        "normalization": normalization, "seed": seed, "dense": dense,
        "identity_field": "timestamp" if dense else "cutoff", "reference_selection": selected,
        "fitted_row_identities": [[str(u), str(c)] for u, c in zip(users[reference_mask], cutoffs[reference_mask])],
        "component_names": list(export.components), "models": models,
        "export": {"path": export_path.name, "sha256": sha256_file(export_path),
                   "schema_version": export.schema_version, "compatibility": export.compatibility},
        "source_hashes": dict(zip(source_names.tolist(), source_hashes.tolist())),
        "interpretation": "Two-dimensional cluster appearance does not establish factor semantics, disentanglement, causal invariance, or recommendation quality."}
    return rows, metadata


def _write_plots(rows: list[dict[str, Any]], output_dir: Path, *, dense: bool, image_format: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("figures require the 'viz' extra: uv sync --extra viz") from error
    time_field = "timestamp" if dense else "cutoff"
    components = list(dict.fromkeys(row["component"] for row in rows))
    cutoff_rank = {"train": 0, "validation": 1, "test": 2}
    cutoffs = sorted(set(row[time_field] for row in rows), key=lambda value: (cutoff_rank.get(value, 3), value))
    figure, axes = plt.subplots(len(components), len(cutoffs), squeeze=False,
                               figsize=(3.5 * len(cutoffs), 3.2 * len(components)))
    for i, component in enumerate(components):
        for j, cutoff in enumerate(cutoffs):
            selected = [r for r in rows if r["component"] == component and r[time_field] == cutoff]
            axes[i, j].scatter([r["x"] for r in selected], [r["y"] for r in selected], s=9, alpha=.35)
            axes[i, j].set_title(f"{component} · {cutoff}")
    figure.tight_layout()
    small = output_dir / f"small_multiples.{image_format}"
    figure.savefig(small, dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(components), squeeze=False, figsize=(4 * len(components), 3.8))
    for axis, component in zip(axes[0], components):
        selected = [r for r in rows if r["component"] == component]
        by_user: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            by_user.setdefault(row["user_id"], []).append(row)
        for user_rows in by_user.values():
            user_rows.sort(key=lambda r: (cutoff_rank.get(r[time_field], 3), r[time_field]))
            axis.plot([r["x"] for r in user_rows], [r["y"] for r in user_rows], alpha=.15, linewidth=.6)
            axis.scatter([r["x"] for r in user_rows], [r["y"] for r in user_rows], s=5, alpha=.25)
        axis.set_title(f"{component} trajectories")
    figure.tight_layout()
    trajectory = output_dir / f"trajectories.{image_format}"
    figure.savefig(trajectory, dpi=160)
    plt.close(figure)
    return [small, trajectory]


def visualize_embeddings(export_path: str | Path, output_dir: str | Path, *, dense: bool = False,
                         reference_cutoff: str | None = None, normalization: str = "standard",
                         reducer: str = "pca", seed: int = 0, image_format: str = "png",
                         overwrite: bool = False, umap_neighbors: int = 15,
                         umap_min_dist: float = .1) -> dict[str, Any]:
    output_dir = Path(output_dir)
    targets = [output_dir / "projection_metadata.json", output_dir / "projections.csv",
               output_dir / "projections.npz", output_dir / f"small_multiples.{image_format}",
               output_dir / f"trajectories.{image_format}"]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"visualization artifacts already exist: {existing}")
    rows, metadata = project_export(export_path, dense=dense, reference_cutoff=reference_cutoff,
        normalization=normalization, reducer=reducer, seed=seed, umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = "timestamp" if dense else "cutoff"
    with targets[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", identity, "component", "x", "y"])
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(targets[2], **{key: np.asarray([row[key] for row in rows])
                                     for key in ("user_id", identity, "component", "x", "y")})
    targets[0].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_plots(rows, output_dir, dense=dense, image_format=image_format)
    return {"status": "complete", "output_dir": str(output_dir.resolve()), "rows": len(rows),
            "components": metadata["component_names"], "reference_selection": metadata["reference_selection"],
            "artifacts": [str(path.resolve()) for path in targets]}
