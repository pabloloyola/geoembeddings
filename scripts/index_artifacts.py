#!/usr/bin/env python3
"""Create a hash index for a matched GeoEmbeddings run and experiment."""

from __future__ import annotations

import argparse

from geoembeddings.artifact_index import build_artifact_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Canonical dataset root")
    parser.add_argument("--experiment-dir", required=True, help="Canonical experiment root")
    parser.add_argument("--output", required=True, help="Destination JSON index")
    parser.add_argument("--task-id", default="T0.1a/T0.2")
    args = parser.parse_args()
    build_artifact_index(args.run_dir, args.experiment_dir, args.output, task_id=args.task_id)


if __name__ == "__main__":
    main()
