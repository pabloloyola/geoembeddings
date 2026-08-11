#!/usr/bin/env python3
"""Reconcile a complete T0.2 evidence index into a per-axis decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from geoembeddings.status_reconciliation import reconcile_artifact_index, render_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = reconcile_artifact_index(args.artifact_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_decision(result, args.artifact_index), encoding="utf-8")


if __name__ == "__main__":
    main()
