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

    @property
    def resolved_config(self) -> Path:
        return self.root / "config.resolved.yaml"

    @property
    def deep_validation_report(self) -> Path:
        return self.root / "deep_validation_report.json"

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
    def prepared_metadata(self) -> Path:
        return self.prepared / "prepared_metadata.json"

    @property
    def resolved_config(self) -> Path:
        return self.prepared / "config.resolved.yaml"

    @property
    def vocabularies(self) -> Path:
        return self.prepared / "vocabularies.json"

    @property
    def checkpoint(self) -> Path:
        return self.model / "best_model.pt"

    @property
    def training_report(self) -> Path:
        return self.model / "training_report.json"

    @property
    def command_log(self) -> Path:
        return self.root / "t0.2_commands.log"

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

    def temporal_routine_evaluation(self, kind: str) -> Path:
        if kind not in {"baseline", "learned"}:
            raise ValueError(f"Unsupported temporal/routine artifact kind: {kind}")
        return self.root / f"{kind}_temporal_routine.json"

    @property
    def baseline_embeddings(self) -> Path:
        return self.root / "statistical_baseline.npz"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation.json"

    @property
    def baseline_evaluation(self) -> Path:
        return self.root / "baseline_evaluation.json"

    def transfer_evaluation(self, kind: str) -> Path:
        if kind not in {"baseline", "learned"}:
            raise ValueError(f"Unsupported transfer artifact kind: {kind}")
        return self.root / f"{kind}_transfer_evaluation.json"

    @property
    def comparison_dir(self) -> Path:
        return self.root / "comparison"

    @property
    def comparison_json(self) -> Path:
        return self.comparison_dir / "embedding_comparison.json"

    @property
    def comparison_markdown(self) -> Path:
        return self.comparison_dir / "embedding_comparison.md"

    @property
    def robustness_dir(self) -> Path:
        return self.root / "robustness"

    def robustness_embeddings(self, kind: str, rate: float) -> Path:
        return self.robustness_dir / kind / f"removal_{rate:.6f}.npz"

    def robustness_report(self, kind: str) -> Path:
        return self.robustness_dir / f"{kind}_robustness.json"

    def robustness_view_dir(self, kind: str) -> Path:
        if kind not in {"baseline", "learned"}:
            raise ValueError(f"Unsupported robustness artifact kind: {kind}")
        return self.robustness_dir / kind
