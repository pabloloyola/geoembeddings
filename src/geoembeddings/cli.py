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

    validate = commands.add_parser("validate", help="Deep-validate a simulator run")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--output", type=Path)

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

    evaluate = commands.add_parser("evaluate", help="Evaluate learned or baseline embeddings")
    _add_embedding_arguments(evaluate)
    evaluate.add_argument("--kind", choices=("learned", "baseline"), default="learned")

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
) -> dict[str, Any]:
    from .export import export_dense_embeddings

    run.validate(require_truth=False)
    if event_stride < 1:
        raise ValueError("--event-stride must be at least 1")
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
) -> dict[str, Any]:
    from .evaluation import evaluate_embeddings

    run.validate(require_truth=True)
    learned = kind == "learned"
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
    prepared_dir = learned_experiment.prepared
    return compare_embeddings(
        run.observed,
        run.truth,
        baseline_experiment.prepared,
        prepared_dir,
        baseline_experiment.baseline_embeddings,
        learned_experiment.embeddings,
        output_dir,
        load_config(Path(args.config).expanduser().resolve()),
    )


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


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulate":
        result = _simulate(args)
    elif args.command == "validate":
        result = _validate(DatasetLayout.from_path(args.run_dir), args.output)
    elif args.command in {"prepare", "train", "baseline", "export", "export-dense", "evaluate"}:
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
            result = _export_dense(run, experiment, config_path, args.event_stride)
        else:
            result = _evaluate(run, experiment, config_path, args.kind)
    elif args.command == "compare":
        result = _compare(args)
    elif args.command == "pipeline":
        result = _pipeline(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
