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


def test_dense_export_accepts_observed_event_stride() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "export-dense",
            "--run-dir",
            "runs/pilot",
            "--experiment-dir",
            "experiments/single_vector",
            "--event-stride",
            "5",
        ]
    )
    assert args.event_stride == 5
    assert not hasattr(args, "truth_dir")


def test_robustness_command_uses_only_canonical_roots() -> None:
    args = build_parser().parse_args(["robustness", "--kind", "baseline", "--run-dir",
        "runs/pilot", "--experiment-dir", "experiments/pilot"])
    assert args.kind == "baseline"
    assert not hasattr(args, "observed_dir") and not hasattr(args, "truth_dir")


def test_reliability_and_benchmark_use_canonical_roots() -> None:
    parser = build_parser()
    reliability = parser.parse_args(["evaluate", "--reliability", "--kind", "baseline",
        "--run-dir", "runs/pilot", "--experiment-dir", "experiments/pilot"])
    benchmark = parser.parse_args(["benchmark", "--run-dir", "runs/pilot",
        "--experiment-dir", "experiments/pilot", "--warmup", "2", "--iterations", "7"])
    assert reliability.reliability and reliability.kind == "baseline"
    assert benchmark.warmup == 2 and benchmark.iterations == 7
    assert not hasattr(benchmark, "truth_dir")


def test_pair_manifest_uses_run_roots_and_canonical_output() -> None:
    args = build_parser().parse_args(["pair-manifest", "--reference-run-dir", "runs/ref",
        "--intervention-run-dir", "runs/int", "--output", "pairs/pair_manifest.json"])
    assert str(args.reference_run_dir) == "runs/ref"
    assert str(args.intervention_run_dir) == "runs/int"
    assert not hasattr(args, "observed_dir") and not hasattr(args, "truth_dir")
