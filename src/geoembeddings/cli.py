from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config
from .io import write_json
from .layout import DatasetLayout, ExperimentLayout


DEFAULT_SIMULATION_CONFIG = Path("configs/simulation/kanto_v1.yaml")
DEFAULT_EMBEDDING_CONFIG = Path("configs/embedding/single_vector.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoembed",
        description="Generate GeoEmbeddings data, train representations, and evaluate them.",
    )
    parser.add_argument("--version", action="version", version=f"geoembed {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="Generate one versioned simulator run")
    _add_simulation_arguments(simulate)

    simulate_pair = commands.add_parser("simulate-pair", help="Generate and validate a configured matched intervention")
    simulate_pair.add_argument("--config", type=Path, default=DEFAULT_SIMULATION_CONFIG)
    simulate_pair.add_argument("--intervention", required=True, choices=("exposure", "opportunity", "observation", "temporary-trip", "sustained-preference", "schedule-shift"))
    simulate_pair.add_argument("--reference-run-dir", required=True, type=Path)
    simulate_pair.add_argument("--intervention-run-dir", required=True, type=Path)
    simulate_pair.add_argument("--pair-dir", required=True, type=Path)
    simulate_pair.add_argument("--users", type=int)
    simulate_pair.add_argument("--days", type=int)
    simulate_pair.add_argument("--seed", type=int)

    validate = commands.add_parser("validate", help="Deep-validate a simulator run")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--output", type=Path)

    pair_manifest = commands.add_parser("pair-manifest", help="Declare a protected matched simulator-run pair")
    pair_manifest.add_argument("--reference-run-dir", required=True, type=Path)
    pair_manifest.add_argument("--intervention-run-dir", required=True, type=Path)
    pair_manifest.add_argument("--output", required=True, type=Path)
    pair_manifest.add_argument("--overwrite", action="store_true")

    validate_pair = commands.add_parser("validate-pair", help="Validate a declared simulator pair at field level")
    validate_pair.add_argument("--pair-manifest", required=True, type=Path)

    evaluate_pair = commands.add_parser("evaluate-pair", help="Evaluate matched counterfactual representations for R5/R7")
    evaluate_pair.add_argument("--pair-manifest", required=True, type=Path)
    evaluate_pair.add_argument("--baseline-experiment-dir", required=True, type=Path, nargs=2,
        metavar=("REFERENCE", "INTERVENTION"))
    evaluate_pair.add_argument("--learned-experiment-dir", required=True, type=Path, nargs=2,
        metavar=("REFERENCE", "INTERVENTION"))
    evaluate_pair.add_argument("--config", type=Path, default=DEFAULT_EMBEDDING_CONFIG)
    evaluate_pair.add_argument("--overwrite", action="store_true")

    evaluate_change = commands.add_parser("evaluate-change", help="Evaluate R1/R11 adaptation on a protected change pair")
    evaluate_change.add_argument("--pair-manifest", required=True, type=Path)
    evaluate_change.add_argument("--baseline-experiment-dir", required=True, type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))
    evaluate_change.add_argument("--learned-experiment-dir", required=True, type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))
    evaluate_change.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser("prepare", help="Fit leakage-safe preprocessing")
    _add_embedding_arguments(prepare)

    train = commands.add_parser("train", help="Train the single-vector sequence encoder")
    _add_embedding_arguments(train)

    baseline = commands.add_parser("baseline", help="Export the non-learned comparator")
    _add_embedding_arguments(baseline)

    export = commands.add_parser("export", help="Export learned user embeddings")
    _add_embedding_arguments(export)

    export_dense = commands.add_parser(
        "export-dense", help="Export learned embeddings at observed event timestamps"
    )
    _add_embedding_arguments(export_dense)
    export_dense.add_argument(
        "--event-stride",
        type=int,
        default=1,
        help="Export every Nth observed event per user, always including first and last",
    )
    export_dense.add_argument("--kind", choices=("learned", "baseline"), default="learned")

    evaluate = commands.add_parser("evaluate", help="Evaluate learned or baseline embeddings")
    _add_embedding_arguments(evaluate)
    evaluate.add_argument("--kind", choices=("learned", "baseline"), default="learned")
    evaluate.add_argument("--episodes", action="store_true", help="Evaluate dense embeddings at protected episode boundaries")
    evaluate.add_argument("--transfer", action="store_true", help="Evaluate versioned R2/R8 spatial transfer slices")
    evaluate.add_argument("--temporal-routine", action="store_true", help="Evaluate protected R3/R4 temporal and routine diagnostics")
    evaluate.add_argument("--reliability", action="store_true", help="Evaluate seeded R10 representation reliability")
    evaluate.add_argument("--overwrite", action="store_true", help="Replace the selected supplemental report")

    benchmark = commands.add_parser("benchmark", help="Benchmark frozen offline exports and evaluation on CPU")
    _add_embedding_arguments(benchmark)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.add_argument("--overwrite", action="store_true")

    rank = commands.add_parser("rank", help="Run an observable dataset-2.0 recommendation baseline")
    rank.add_argument("--run-dir", required=True, type=Path)
    rank.add_argument("--experiment-dir", required=True, type=Path)
    rank.add_argument("--model", required=True, choices=("popularity", "nearest", "category_preference", "frozen_embedding"))
    rank.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    rank.add_argument("--overwrite", action="store_true")

    robustness = commands.add_parser("robustness", help="Re-encode deterministic observed-data robustness views for R6/R7")
    _add_embedding_arguments(robustness)
    robustness.add_argument("--kind", choices=("learned", "baseline"), default="learned")
    robustness.add_argument("--views", default=None,
        help="Comma-separated views: gps,timestamp,leave-one-service-out,recent-truncation")

    compare = commands.add_parser(
        "compare", help="Compare baseline and learned embeddings with common frozen probes"
    )
    compare.add_argument("--run-dir", required=True, type=Path)
    compare.add_argument(
        "--experiment-dir",
        type=Path,
        help="Shorthand when baseline and learned artifacts share one experiment directory",
    )
    compare.add_argument("--baseline-experiment-dir", type=Path)
    compare.add_argument("--learned-experiment-dir", type=Path)
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--config", type=Path, default=DEFAULT_EMBEDDING_CONFIG)
    compare.add_argument(
        "--factorized-experiment", action="append", default=[], metavar="NAME=PATH",
        help="Build the T2.7 matrix comparison from repeated immutable NAME=PATH roots",
    )

    pipeline = commands.add_parser(
        "pipeline", help="Run simulation, validation, preparation, embedding, and evaluation"
    )
    _add_simulation_arguments(pipeline)
    pipeline.add_argument("--embedding-config", type=Path, default=DEFAULT_EMBEDDING_CONFIG)
    pipeline.add_argument("--experiment-dir", required=True, type=Path)
    pipeline.add_argument("--mode", choices=("baseline", "learned"), default="baseline")
    return parser


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--users", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario")
    parser.add_argument("--full-kanto", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_EMBEDDING_CONFIG)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)


def _simulate(args: argparse.Namespace) -> dict[str, Any]:
    from . import simulator

    config_path = Path(args.config).expanduser().resolve()
    config = simulator.load_config(config_path)
    run = config["run"]
    values = {
        "users": args.users,
        "days": args.days,
        "start_date": args.start_date,
        "seed": args.seed,
        "scenario": args.scenario,
        "full_kanto": args.full_kanto,
    }
    for name, value in values.items():
        if value is None:
            value = run[name]
        run[name] = value
        setattr(args, name, value)
    args.output = str(DatasetLayout.from_path(args.run_dir).root)
    run["output"] = args.output
    args.config = str(config_path)
    simulator.activate_config(config)
    if args.scenario not in simulator.SCENARIO_SETTINGS:
        raise ValueError(
            f"Unknown scenario {args.scenario!r}; choose one of {sorted(simulator.SCENARIO_SETTINGS)}"
        )
    if args.users < 10 or args.days < 2:
        raise ValueError("Use at least 10 users and 2 days so validation is meaningful")
    return simulator.simulate(args)


def _validate(run: DatasetLayout, output: Path | None = None) -> dict[str, Any]:
    from .simulation_validation import validate

    run.validate(require_truth=True)
    report = validate(run.root)
    destination = output or (run.root / "deep_validation_report.json")
    write_json(report, destination)
    if report["status"] != "passed":
        raise RuntimeError(f"Deep simulator validation failed; inspect {destination}")
    return report


def _prepare(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path) -> dict[str, Any]:
    from .prepare import prepare_data

    run.validate(require_truth=False)
    return prepare_data(run.observed, experiment.prepared, load_config(config_path))


def _baseline(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path) -> dict[str, Any]:
    from .baseline import export_statistical_baseline

    run.validate(require_truth=False)
    return export_statistical_baseline(
        run.observed, experiment.prepared, experiment.baseline_embeddings, load_config(config_path)
    )


def _train(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path) -> dict[str, Any]:
    from .training import train_model

    run.validate(require_truth=False)
    return train_model(run.observed, experiment.prepared, experiment.model, load_config(config_path))


def _export(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path) -> dict[str, Any]:
    from .export import export_embeddings

    run.validate(require_truth=False)
    if not experiment.checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {experiment.checkpoint}")
    return export_embeddings(
        run.observed,
        experiment.prepared,
        experiment.checkpoint,
        experiment.embeddings,
        load_config(config_path),
    )


def _export_dense(
    run: DatasetLayout,
    experiment: ExperimentLayout,
    config_path: Path,
    event_stride: int,
    kind: str = "learned",
) -> dict[str, Any]:
    run.validate(require_truth=False)
    if event_stride < 1:
        raise ValueError("--event-stride must be at least 1")
    if kind == "baseline":
        from .baseline import export_dense_statistical_baseline
        return export_dense_statistical_baseline(run.observed, experiment.prepared,
            experiment.dense_baseline_embeddings, load_config(config_path), event_stride=event_stride)
    from .export import export_dense_embeddings
    if not experiment.checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {experiment.checkpoint}")
    return export_dense_embeddings(
        run.observed,
        experiment.prepared,
        experiment.checkpoint,
        experiment.dense_embeddings,
        load_config(config_path),
        event_stride=event_stride,
    )


def _evaluate(
    run: DatasetLayout,
    experiment: ExperimentLayout,
    config_path: Path,
    kind: str,
    episodes: bool = False,
    transfer: bool = False,
    temporal_routine: bool = False,
    reliability: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    from .evaluation import evaluate_embeddings, evaluate_episode_response

    learned = kind == "learned"
    if sum((episodes, transfer, temporal_routine, reliability)) > 1:
        raise ValueError("Supplemental evaluation modes select separate reports")
    if reliability:
        run.validate(require_truth=False)
        from .reliability import evaluate_reliability
        embeddings = experiment.embeddings if learned else experiment.baseline_embeddings
        if not embeddings.is_file():
            raise FileNotFoundError(f"Missing {kind} embeddings: {embeddings}")
        return evaluate_reliability(run.observed, experiment.prepared, embeddings,
            experiment.reliability_evaluation(kind), load_config(config_path), kind=kind,
            overwrite=overwrite)
    if temporal_routine:
        run.validate(require_truth=True)
        from .temporal_routine_evaluation import evaluate_temporal_routine
        dense = experiment.dense_embeddings if learned else experiment.dense_baseline_embeddings
        if not dense.is_file():
            raise FileNotFoundError(f"Missing dense {kind} embeddings: {dense}")
        return evaluate_temporal_routine(
            run.truth, experiment.prepared, dense,
            experiment.temporal_routine_evaluation(kind), load_config(config_path), kind=kind,
        )
    if transfer:
        run.validate(require_truth=False)
        from .spatial_evaluation import evaluate_spatial_transfer
        embeddings = experiment.embeddings if learned else experiment.baseline_embeddings
        if not embeddings.is_file():
            raise FileNotFoundError(f"Missing {kind} embeddings: {embeddings}")
        return evaluate_spatial_transfer(run.observed, experiment.prepared, embeddings,
            experiment.transfer_evaluation(kind), load_config(config_path), kind=kind)
    run.validate(require_truth=True)
    if episodes:
        dense = experiment.dense_embeddings if learned else experiment.dense_baseline_embeddings
        output = experiment.episode_response if learned else experiment.baseline_episode_response
        if not dense.is_file():
            raise FileNotFoundError(f"Missing dense {kind} embeddings: {dense}")
        return evaluate_episode_response(run.truth, experiment.prepared, dense, output,
                                         load_config(config_path), kind=kind)
    checkpoint = experiment.checkpoint if learned else None
    embeddings = experiment.embeddings if learned else experiment.baseline_embeddings
    output = experiment.evaluation if learned else experiment.baseline_evaluation
    if learned and not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    if not embeddings.is_file():
        raise FileNotFoundError(f"Missing {kind} embeddings: {embeddings}")
    return evaluate_embeddings(
        run.observed,
        run.truth,
        experiment.prepared,
        checkpoint,
        embeddings,
        output,
        load_config(config_path),
    )


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.factorized_experiment:
        from .factorization_comparison import compare_factorization_matrix
        if args.experiment_dir is not None or args.baseline_experiment_dir is not None or args.learned_experiment_dir is not None:
            raise ValueError("--factorized-experiment cannot be combined with pairwise experiment arguments")
        roots: dict[str, Path] = {}
        for item in args.factorized_experiment:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in roots:
                raise ValueError("Each --factorized-experiment must be a unique NAME=PATH")
            roots[name] = Path(path).expanduser().resolve()
        run = DatasetLayout.from_path(args.run_dir)
        run.validate(require_truth=True)
        output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
                  ExperimentLayout.from_path(roots["factorized_pc"]).comparison_dir)
        return compare_factorization_matrix(run, roots, output)
    from .comparison import compare_embeddings

    run = DatasetLayout.from_path(args.run_dir)
    run.validate(require_truth=True)
    if args.experiment_dir is not None:
        if args.baseline_experiment_dir is not None or args.learned_experiment_dir is not None:
            raise ValueError(
                "Use either --experiment-dir or the two explicit experiment-directory arguments"
            )
        baseline_experiment = ExperimentLayout.from_path(args.experiment_dir)
        learned_experiment = baseline_experiment
    else:
        if args.baseline_experiment_dir is None or args.learned_experiment_dir is None:
            raise ValueError(
                "Provide --experiment-dir, or both --baseline-experiment-dir and "
                "--learned-experiment-dir"
            )
        baseline_experiment = ExperimentLayout.from_path(args.baseline_experiment_dir)
        learned_experiment = ExperimentLayout.from_path(args.learned_experiment_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else learned_experiment.comparison_dir
    )
    return compare_embeddings(
        run.observed, run.truth, baseline_experiment.prepared, learned_experiment.prepared,
        baseline_experiment.baseline_embeddings, learned_experiment.embeddings, output_dir,
        load_config(Path(args.config).expanduser().resolve()),
    )


def _robustness(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path,
                kind: str, views: str | None = None) -> dict[str, Any]:
    from .evaluation import evaluate_event_removal
    from .robustness import export_robustness_views

    run.validate(require_truth=False)
    config = load_config(config_path)
    checkpoint = experiment.checkpoint if kind == "learned" else Path("unused")
    original = experiment.embeddings if kind == "learned" else experiment.baseline_embeddings
    if not original.is_file():
        raise FileNotFoundError(f"Missing unmodified {kind} embeddings: {original}")
    if kind == "learned" and not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    requested = None if views is None else [value.strip() for value in views.split(",") if value.strip()]
    if views is not None and not requested:
        raise ValueError("--views must contain at least one view")
    manifest = export_robustness_views(run.observed, experiment.prepared, checkpoint,
                                       experiment.robustness_dir, config, kind=kind, views=requested)
    # This validation is deliberately deferred until view construction and encoding finish.
    run.validate(require_truth=True)
    return evaluate_event_removal(run.truth, original, manifest,
                                   experiment.robustness_report(kind), config)


def _pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run = DatasetLayout.from_path(args.run_dir)
    experiment = ExperimentLayout.from_path(args.experiment_dir)
    simulation = _simulate(args)
    validation = _validate(run)
    embedding_config = Path(args.embedding_config).expanduser().resolve()
    prepared = _prepare(run, experiment, embedding_config)
    if args.mode == "baseline":
        representation = _baseline(run, experiment, embedding_config)
    else:
        training = _train(run, experiment, embedding_config)
        exported = _export(run, experiment, embedding_config)
        representation = {"training": training, "export": exported}
    evaluation = _evaluate(run, experiment, embedding_config, args.mode)
    return {
        "mode": args.mode,
        "run_dir": str(run.root),
        "experiment_dir": str(experiment.root),
        "simulation": {"users": simulation["users"], "days": simulation["days"]},
        "validation": validation["status"],
        "preparation": prepared["rows"],
        "representation": representation,
        "evaluation": evaluation,
    }


def _benchmark(run: DatasetLayout, experiment: ExperimentLayout, config_path: Path,
               warmup: int, iterations: int, overwrite: bool) -> dict[str, Any]:
    from .benchmark import run_offline_benchmark
    run.validate(require_truth=False)
    return run_offline_benchmark(run.observed, experiment.prepared,
        {"baseline": experiment.baseline_embeddings, "learned": experiment.embeddings},
        experiment.offline_benchmark, load_config(config_path), warmup=warmup,
        iterations=iterations, overwrite=overwrite)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulate":
        result = _simulate(args)
    elif args.command == "simulate-pair":
        from .simulate_pair import simulate_pair
        result = simulate_pair(args.config, args.reference_run_dir, args.intervention_run_dir,
                               args.pair_dir, intervention=args.intervention, users=args.users,
                               days=args.days, seed=args.seed)
    elif args.command == "validate":
        result = _validate(DatasetLayout.from_path(args.run_dir), args.output)
    elif args.command == "pair-manifest":
        from .pair_manifest import create_pair_manifest
        result = create_pair_manifest(args.reference_run_dir, args.intervention_run_dir,
                                      args.output, overwrite=args.overwrite)
    elif args.command == "validate-pair":
        from .pair_integrity import validate_pair
        result = validate_pair(args.pair_manifest)
    elif args.command == "evaluate-pair":
        from .pair_evaluation import evaluate_pair
        result = evaluate_pair(args.pair_manifest, args.baseline_experiment_dir,
            args.learned_experiment_dir, load_config(Path(args.config).expanduser().resolve()),
            overwrite=args.overwrite)
    elif args.command == "evaluate-change":
        from .evaluation import evaluate_change
        result = evaluate_change(args.pair_manifest, args.baseline_experiment_dir,
                                 args.learned_experiment_dir, overwrite=args.overwrite)
    elif args.command in {"prepare", "train", "baseline", "export", "export-dense", "evaluate", "robustness", "benchmark"}:
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        config_path = Path(args.config).expanduser().resolve()
        if args.command == "prepare":
            result = _prepare(run, experiment, config_path)
        elif args.command == "train":
            result = _train(run, experiment, config_path)
        elif args.command == "baseline":
            result = _baseline(run, experiment, config_path)
        elif args.command == "export":
            result = _export(run, experiment, config_path)
        elif args.command == "export-dense":
            result = _export_dense(run, experiment, config_path, args.event_stride, args.kind)
        elif args.command == "evaluate":
            result = _evaluate(run, experiment, config_path, args.kind, args.episodes, args.transfer,
                               args.temporal_routine, args.reliability, args.overwrite)
        elif args.command == "benchmark":
            result = _benchmark(run, experiment, config_path, args.warmup, args.iterations, args.overwrite)
        else:
            result = _robustness(run, experiment, config_path, args.kind, args.views)
    elif args.command == "compare":
        result = _compare(args)
    elif args.command == "rank":
        from .ranking import run_ranking
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        manifest = run.validate(require_truth=False)
        result = run_ranking(run.observed, manifest, experiment.ranking_predictions(args.model),
                             experiment.ranking_report(args.model), model=args.model,
                             ks=args.k, overwrite=args.overwrite,
                             embedding_path=experiment.embeddings,
                             checkpoint_path=experiment.frozen_ranking_checkpoint,
                             baseline_report_paths={name: experiment.ranking_report(name) for name in
                                 ("popularity", "nearest", "category_preference")})
    elif args.command == "pipeline":
        result = _pipeline(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
