"""Canonical paths for simulator datasets and embedding experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import (DATASET_CONTRACT_NAME, DATASET_CONTRACT_VERSION, LEGACY_DATASET_CONTRACT_VERSIONS,
                       LEGACY_OBSERVED_FILES, OBSERVED_FILES, TRUTH_FILES, validate_identity_manifest)


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
    def user_latents_truth(self) -> Path:
        """Canonical evaluator-only protected-label source."""
        return self.truth / TRUTH_FILES["user_latents"]

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
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete GeoEmbeddings run at {self.root}: missing manifest.json")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("dataset_contract", {})
        version = str(declared.get("version", ""))
        if declared.get("name") != DATASET_CONTRACT_NAME:
            raise ValueError(f"Unsupported dataset contract: {declared}")
        if version == DATASET_CONTRACT_VERSION:
            observed_files = OBSERVED_FILES
        elif version in LEGACY_DATASET_CONTRACT_VERSIONS:
            # Explicit read-only migration behavior: v1 datasets remain usable by
            # event models, but recommendation tables are neither invented nor consumed.
            observed_files = LEGACY_OBSERVED_FILES
        else:
            raise ValueError(f"Dataset contract {version} is incompatible with supported {DATASET_CONTRACT_VERSION}")
        required = [self.observed / name for name in observed_files.values()]
        if require_truth:
            required.extend(self.truth / name for name in TRUTH_FILES.values())
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete GeoEmbeddings run at {self.root}: {missing}")

        # Identity metadata is run-level rather than part of the public table
        # contract. New simulator artifacts must nevertheless be complete.
        if "identity" in manifest:
            validate_identity_manifest(manifest["identity"], stream_names=("world", "user_latents", "episodes", "choices", "observation"))
        return manifest


@dataclass(frozen=True)
class PairLayout:
    """Resolve protected paired-run artifacts from one pair directory."""

    root: Path

    @classmethod
    def from_path(cls, value: str | Path) -> "PairLayout":
        return cls(Path(value).expanduser().resolve())

    @classmethod
    def from_manifest_path(cls, value: str | Path) -> "PairLayout":
        path = Path(value).expanduser().resolve()
        if path.name != "pair_manifest.json":
            raise ValueError("pair manifest path must name the canonical pair_manifest.json artifact")
        return cls(path.parent)

    @property
    def manifest(self) -> Path:
        return self.root / "pair_manifest.json"

    @property
    def integrity_report(self) -> Path:
        return self.root / "pair_integrity.json"

    @property
    def counterfactual_comparison_json(self) -> Path:
        return self.root / "counterfactual_comparison.json"

    @property
    def counterfactual_comparison_markdown(self) -> Path:
        return self.root / "counterfactual_comparison.md"

    @property
    def change_evaluation_json(self) -> Path:
        return self.root / "change_evaluation.json"

    @property
    def change_evaluation_markdown(self) -> Path:
        return self.root / "change_evaluation.md"

    @property
    def exposure_counterfactual_json(self) -> Path:
        return self.root / "ranking" / "exposure_counterfactual.json"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def nonstationarity_audit_json(self) -> Path:
        return self.audits_dir / "nonstationarity.json"

    @property
    def nonstationarity_audit_markdown(self) -> Path:
        return self.audits_dir / "nonstationarity.md"

    @property
    def privacy_audit_json(self) -> Path:
        """Canonical authoritative R12 privacy-audit artifact."""
        return self.audits_dir / "privacy.json"

    @property
    def privacy_audit_markdown(self) -> Path:
        """Canonical human-readable rendering of the R12 privacy audit."""
        return self.audits_dir / "privacy.md"


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
    def training_participation(self) -> Path:
        """Canonical immutable learned-training participation artifact."""
        return self.model / "training_participation.json"

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

    def reliability_evaluation(self, kind: str) -> Path:
        if kind not in {"baseline", "learned"}:
            raise ValueError(f"Unsupported reliability artifact kind: {kind}")
        return self.root / ("baseline_reliability.json" if kind == "baseline" else "reliability.json")

    @property
    def benchmarks_dir(self) -> Path:
        return self.root / "benchmarks"

    @property
    def offline_benchmark(self) -> Path:
        return self.benchmarks_dir / "offline.json"

    @property
    def online_benchmark(self) -> Path:
        """Canonical immutable online-update benchmark report."""
        return self.benchmarks_dir / "online.json"

    @property
    def online_workload(self) -> Path:
        return self.benchmarks_dir / "online_workload.json"

    @property
    def ranking_dir(self) -> Path:
        return self.root / "ranking"

    def ranking_predictions(self, model: str) -> Path:
        return self.ranking_dir / f"{model}.npz"

    def ranking_report(self, model: str) -> Path:
        return self.ranking_dir / f"{model}.json"

    @property
    def frozen_ranking_checkpoint(self) -> Path:
        return self.ranking_dir / "frozen_embedding_checkpoint.npz"

    @property
    def exposure_ranking_checkpoint(self) -> Path:
        return self.ranking_dir / "exposure_aware_checkpoint.npz"

    @property
    def ranking_transfer_slices(self) -> Path:
        return self.ranking_dir / "transfer_slices.json"

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
    def factorized_comparison_json(self) -> Path:
        return self.comparison_dir / "factorized_comparison.json"

    @property
    def factorized_comparison_markdown(self) -> Path:
        return self.comparison_dir / "factorized_comparison.md"

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


@dataclass(frozen=True)
class PrivacyEvidenceLayout:
    """Canonical files below one privacy evidence root."""

    root: Path

    @classmethod
    def from_path(cls, value: str | Path) -> "PrivacyEvidenceLayout":
        return cls(Path(value).expanduser().resolve())

    @property
    def evidence_index(self) -> Path:
        return self.root / "evidence_index.json"


@dataclass(frozen=True)
class UtilityReportLayout:
    """Resolve immutable, named utility reports from their common root."""

    root: Path

    @classmethod
    def from_path(cls, value: str | Path) -> "UtilityReportLayout":
        return cls(Path(value).expanduser().resolve())

    def report(self, name: str) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"Invalid utility-report name: {name!r}")
        return self.root / f"{name}.json"
