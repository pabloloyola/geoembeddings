from __future__ import annotations

from geoembeddings.cli import build_parser


def test_embedding_commands_use_roots_instead_of_internal_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "--run-dir",
            "runs/pilot",
            "--experiment-dir",
            "experiments/baseline",
        ]
    )
    assert str(args.run_dir) == "runs/pilot"
    assert str(args.experiment_dir) == "experiments/baseline"
    assert not hasattr(args, "observed_dir")
    assert not hasattr(args, "truth_dir")
    assert not hasattr(args, "prepared_dir")


def test_compare_accepts_shared_experiment_directory() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--run-dir",
            "runs/pilot",
            "--experiment-dir",
            "experiments/single_vector",
        ]
    )
    assert str(args.run_dir) == "runs/pilot"
    assert str(args.experiment_dir) == "experiments/single_vector"
