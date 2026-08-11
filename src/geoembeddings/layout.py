"""Canonical paths for simulator datasets and embedding experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import DATASET_CONTRACT_NAME, DATASET_CONTRACT_VERSION, OBSERVED_FILES, TRUTH_FILES


@dataclass(frozen=True)
class DatasetLayout:
    """Resolve every dataset path from a single simulator run directory."""

    root: Path

    @classmethod
    def from_path(cls, value: str | Path) -> "DatasetLayout":
        root = Path(value).expanduser().resolve()
        if root.name in {"observed", "truth"}:
            raise ValueError("--run-dir must point to the dataset root, not observed/ or truth/")
        return cls(root)

    @property
    def observed(self) -> Path:
        return self.root / "observed"

    @property
    def truth(self) -> Path:
        return self.root / "truth"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def validate(self, *, require_truth: bool = False) -> dict[str, Any]:
        required = [self.observed / name for name in OBSERVED_FILES.values()]
        if require_truth:
            required.extend(self.truth / name for name in TRUTH_FILES.values())
        required.append(self.manifest_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete GeoEmbeddings run at {self.root}: {missing}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("dataset_contract")
        if declared:
            if declared.get("name") != DATASET_CONTRACT_NAME:
                raise ValueError(f"Unsupported dataset contract: {declared}")
            version = str(declared.get("version", ""))
            if version.split(".", 1)[0] != DATASET_CONTRACT_VERSION.split(".", 1)[0]:
                raise ValueError(
                    f"Dataset contract {version} is incompatible with supported {DATASET_CONTRACT_VERSION}"
                )
        return manifest


@dataclass(frozen=True)
class ExperimentLayout:
    """Resolve every embedding artifact from a single experiment directory."""

    root: Path

    @classmethod
    def from_path(cls, value: str | Path) -> "ExperimentLayout":
        return cls(Path(value).expanduser().resolve())

    @property
    def prepared(self) -> Path:
        return self.root / "prepared"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def checkpoint(self) -> Path:
        return self.model / "best_model.pt"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings.npz"

    @property
    def dense_embeddings(self) -> Path:
        return self.root / "dense_embeddings.npz"

    @property
    def dense_baseline_embeddings(self) -> Path:
        return self.root / "dense_statistical_baseline.npz"

    @property
    def episode_response(self) -> Path:
        return self.root / "episode_response.json"

    @property
    def baseline_episode_response(self) -> Path:
        return self.root / "baseline_episode_response.json"

    @property
    def baseline_embeddings(self) -> Path:
        return self.root / "statistical_baseline.npz"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation.json"

    @property
    def baseline_evaluation(self) -> Path:
        return self.root / "baseline_evaluation.json"

    @property
    def comparison_dir(self) -> Path:
        return self.root / "comparison"
