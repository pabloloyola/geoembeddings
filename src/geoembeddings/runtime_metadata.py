"""Versioned, truth-independent runtime provenance for JSON reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import math
import platform
from pathlib import Path
import subprocess
from typing import Any

import torch


RUNTIME_METADATA_SCHEMA_VERSION = "geoembeddings-runtime-metadata/1.0"


@dataclass(frozen=True)
class RuntimeMetadata:
    """Required portable fields plus nullable accelerator-specific details."""

    schema_version: str
    python_version: str
    package_version: str | None
    pytorch_version: str
    operating_system: str
    device_type: str | None
    source_commit: str | None
    wall_clock_duration_seconds: float
    seed: int
    accelerator: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        duration = self.wall_clock_duration_seconds
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise TypeError("wall_clock_duration_seconds must be a real number")
        if not math.isfinite(float(duration)) or duration < 0:
            raise ValueError("wall_clock_duration_seconds must be finite and non-negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-compatible data without dropping nulls."""
        return asdict(self)


def collect_runtime_metadata(
    *, duration_seconds: float, seed: int, device: str | torch.device | None = None,
    source_root: str | Path | None = None,
) -> RuntimeMetadata:
    """Collect process metadata without accessing dataset inputs (including truth/)."""
    device_type = torch.device(device).type if device is not None else None
    accelerator: dict[str, Any] | None = None
    if device_type == "cuda":
        accelerator = {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_version": torch.version.cuda,
        }
    elif device_type == "mps":
        accelerator = {"name": None, "mps_available": bool(torch.backends.mps.is_available())}
    try:
        package_version = version("geoembeddings")
    except PackageNotFoundError:
        package_version = None
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        source_commit = None
    return RuntimeMetadata(
        schema_version=RUNTIME_METADATA_SCHEMA_VERSION,
        python_version=platform.python_version(),
        package_version=package_version,
        pytorch_version=torch.__version__,
        operating_system=platform.platform(),
        device_type=device_type,
        source_commit=source_commit,
        wall_clock_duration_seconds=float(duration_seconds),
        seed=seed,
        accelerator=accelerator,
    )
