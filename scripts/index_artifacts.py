#!/usr/bin/env python3
"""Create a hash index for a matched GeoEmbeddings run and experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from geoembeddings.artifact_index import build_artifact_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Canonical dataset root")
    parser.add_argument("--experiment-dir", help="Canonical experiment root")
    parser.add_argument("--factorized-comparison", help="T2.7 matrix comparison JSON")
    parser.add_argument("--output", required=True, help="Destination JSON index")
    parser.add_argument("--task-id", default="T0.1a/T0.2")
    args = parser.parse_args()
    if args.factorized_comparison:
        comparison_path = Path(args.factorized_comparison).resolve()
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        artifacts = {}
        paths = [comparison_path, comparison_path.with_suffix(".md")]
        for record in comparison["experiments"].values():
            root = Path(record["root"])
            paths += [root / "prepared/prepared_metadata.json", root / "model/training_report.json",
                      root / "model/best_model.pt", root / "embeddings.npz", root / "dense_embeddings.npz",
                      root / "evaluation.json", root / "episode_response.json",
                      root / "learned_temporal_routine.json", root / "robustness/learned_robustness.json"]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts[str(path)] = {"bytes": path.stat().st_size,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        payload = {"schema_version": "geoembeddings-factorization-evidence-index/1.0",
                   "task_id": args.task_id, "matched_identity": comparison["matched_identity"],
                   "decision": comparison["decision"], "artifacts": artifacts}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        if not args.experiment_dir or not args.run_dir:
            parser.error("--run-dir and --experiment-dir are required unless --factorized-comparison is used")
        build_artifact_index(args.run_dir, args.experiment_dir, args.output, task_id=args.task_id)


if __name__ == "__main__":
    main()
