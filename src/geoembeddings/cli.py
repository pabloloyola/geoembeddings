from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config, load_mapping_config
from .io import write_json
from .layout import DatasetLayout, ExperimentLayout, PairLayout


DEFAULT_SIMULATION_CONFIG = Path("configs/simulation/kanto_v1.yaml")
DEFAULT_EMBEDDING_CONFIG = Path("configs/embedding/single_vector.yaml")
DEFAULT_PRIVACY_CONFIG = Path("configs/privacy/diagnostic_v1.yaml")
DEFAULT_CONTEXT_PAIR_CONFIG = Path("configs/preflight/context_session_v1.yaml")
DEFAULT_CONTEXT_EMBEDDING_CONFIG = Path("configs/embedding/two_timescale_pc.yaml")


class ScientificMetricUnavailable(RuntimeError):
    """Signal an expected lack of scientific support, not a command failure."""

    def __init__(self, section: str, reason: str) -> None:
        super().__init__(reason)
        self.section = section
        self.reason = reason


def _error_message(error: Exception) -> tuple[int, str]:
    """Map expected failures to stable, path-safe CLI messages and exit codes."""
    if isinstance(error, FileExistsError):
        return 3, "output already exists and is immutable; choose a new output or use a supported --overwrite option"
    if isinstance(error, FileNotFoundError):
        return 4, "required source artifact is missing; run the prerequisite command and verify the supplied root"
    if isinstance(error, (ValueError, PermissionError)):
        return 2, "input schema or identity authentication failed; verify the artifact versions, hashes, and matching roots"
    return 1, "unexpected internal error; rerun with validated inputs and report this failure if it persists"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoembed",
        description="Generate GeoEmbeddings data, train representations, and evaluate them.",
    )
    parser.add_argument("--version", action="version", version=f"geoembed {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_evidence = commands.add_parser(
        "inspect-evidence", help="Read-only verification of documentation evidence indexes"
    )
    inspect_evidence.add_argument("--index-dir", type=Path, default=Path("docs/artifacts"))

    simulate = commands.add_parser("simulate", help="Generate one versioned simulator run")
    _add_simulation_arguments(simulate)

    simulate_pair = commands.add_parser("simulate-pair", help="Generate and validate a configured matched intervention")
    simulate_pair.add_argument("--config", type=Path, default=DEFAULT_SIMULATION_CONFIG)
    simulate_pair.add_argument("--intervention", required=True, choices=("exposure", "opportunity", "observation", "temporary-trip", "sustained-preference", "schedule-shift", "temporary_schedule_shift_v1"))
    simulate_pair.add_argument("--scenario", help="Explicit scenario requested for both matched runs")
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
    evaluate_pair.add_argument("--ranking-predictions", type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))
    evaluate_pair.add_argument("--ranking-reports", type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))

    evaluate_change = commands.add_parser("evaluate-change", help="Evaluate R1/R11 adaptation on a protected change pair")
    evaluate_change.add_argument("--pair-manifest", required=True, type=Path)
    evaluate_change.add_argument("--baseline-experiment-dir", required=True, type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))
    evaluate_change.add_argument("--learned-experiment-dir", required=True, type=Path, nargs=2, metavar=("REFERENCE", "INTERVENTION"))
    evaluate_change.add_argument("--overwrite", action="store_true")

    audit_change = commands.add_parser("audit-nonstationarity", help="Audit matched R11 adaptation and forgetting")
    audit_change.add_argument("--no-change-report", required=True, type=Path)
    audit_change.add_argument("--temporary-report", required=True, type=Path)
    audit_change.add_argument("--sustained-report", required=True, type=Path)
    audit_change.add_argument("--output-dir", required=True, type=Path)
    audit_change.add_argument("--adaptation-threshold", type=float, default=0.1)
    audit_change.add_argument("--recovery-threshold", type=float, default=0.05)
    audit_change.add_argument("--overwrite", action="store_true")

    audit_privacy = commands.add_parser("audit-privacy", help="Run the authenticated R12 diagnostic-control audit")
    audit_privacy.add_argument("--run-dir", required=True, type=Path)
    audit_privacy.add_argument("--experiment-dir", required=True, action="append", metavar="NAME=ROOT")
    audit_privacy.add_argument("--evidence-dir", required=True, type=Path)
    audit_privacy.add_argument("--utility-report-dir", required=True, type=Path)
    audit_privacy.add_argument("--config", type=Path, default=DEFAULT_PRIVACY_CONFIG)
    audit_privacy.add_argument("--output-dir", required=True, type=Path)
    audit_privacy.add_argument("--overwrite", action="store_true")

    calibrate = commands.add_parser("calibrate-reliability", help="Calibrate R10 diagnostic-control uncertainty")
    calibrate.add_argument("--run-dir", required=True, type=Path)
    calibrate.add_argument("--experiment-dir", required=True, action="append", metavar="NAME=ROOT")
    calibrate.add_argument("--config", type=Path, default=Path("configs/reliability/diagnostic_v1.yaml"))
    calibrate.add_argument("--output-dir", required=True, type=Path)
    calibrate.add_argument("--overwrite", action="store_true")

    context_preflight = commands.add_parser(
        "context-pair-preflight", help="Build an observed-only context-session pair manifest"
    )
    context_preflight.add_argument("--run-dir", required=True, type=Path)
    context_preflight.add_argument("--experiment-dir", required=True, type=Path)
    context_preflight.add_argument("--config", type=Path, default=DEFAULT_CONTEXT_PAIR_CONFIG)
    context_preflight.add_argument("--embedding-config", type=Path, default=DEFAULT_CONTEXT_EMBEDDING_CONFIG)
    context_preflight.add_argument("--output-dir", required=True, type=Path)

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

    visualize = commands.add_parser(
        "visualize-embeddings", help="Project and plot an observed-only frozen embedding export"
    )
    visualize.add_argument("--experiment-dir", required=True, type=Path)
    visualize.add_argument("--kind", choices=("learned", "baseline"), default="learned")
    visualize.add_argument("--dense", action="store_true", help="Use the timestamped dense export")
    visualize.add_argument("--reference-cutoff", help="Exact cutoff or timestamp used to fit the reducer")
    visualize.add_argument("--normalization", choices=("standard", "center", "none"), default="standard")
    visualize.add_argument("--reducer", choices=("pca", "umap"), default="pca")
    visualize.add_argument("--seed", type=int, default=0)
    visualize.add_argument("--format", choices=("png", "svg"), default="png")
    visualize.add_argument("--umap-neighbors", type=int, default=15)
    visualize.add_argument("--umap-min-dist", type=float, default=.1)
    visualize.add_argument("--overwrite", action="store_true")

    journey = commands.add_parser("user-journey", help="Build a deterministic evaluator-only R1/R4/R8/R9 report")
    journey.add_argument("--run-dir", required=True, type=Path)
    journey.add_argument("--experiment-dir", required=True, type=Path)
    journey.add_argument("--user-id", required=True)
    journey.add_argument("--start", required=True, help="Inclusive ISO-8601 interval start")
    journey.add_argument("--end", required=True, help="Inclusive ISO-8601 interval end")
    journey.add_argument("--evaluator-truth", action="store_true", help="Explicitly permit protected episode truth")
    journey.add_argument("--ranking-model", action="append", dest="ranking_models")
    journey.add_argument("--max-events", type=int, default=500)
    journey.add_argument("--max-requests", type=int, default=50)
    journey.add_argument("--max-candidates", type=int, default=20)
    journey.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate learned or baseline embeddings")
    _add_embedding_arguments(evaluate)
    evaluate.add_argument("--kind", choices=("learned", "baseline"), default="learned")
    evaluate.add_argument("--episodes", action="store_true", help="Evaluate dense embeddings at protected episode boundaries")
    evaluate.add_argument("--transfer", action="store_true", help="Evaluate versioned R2/R8 spatial transfer slices")
    evaluate.add_argument("--temporal-routine", action="store_true", help="Evaluate protected R3/R4 temporal and routine diagnostics")
    evaluate.add_argument("--reliability", action="store_true", help="Evaluate seeded R10 representation reliability")
    evaluate.add_argument("--overwrite", action="store_true", help="Replace the selected supplemental report")

    benchmark = commands.add_parser("benchmark", help="Benchmark frozen exports and atomic online updates")
    _add_embedding_arguments(benchmark)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--overwrite", action="store_true")

    rank = commands.add_parser("rank", help="Run an observable dataset-2.0 recommendation baseline")
    rank.add_argument("--run-dir", required=True, type=Path)
    rank.add_argument("--experiment-dir", required=True, type=Path)
    rank.add_argument("--model", required=True, choices=("popularity", "nearest", "category_preference", "frozen_embedding", "exposure_aware"))
    rank.add_argument("--ranking-config", type=Path, default=Path("configs/ranking/exposure_v1.yaml"))
    rank.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    rank.add_argument("--overwrite", action="store_true")

    evaluate_ranking = commands.add_parser(
        "evaluate-ranking", help="Evaluate frozen seen/unseen ranking transfer slices"
    )
    evaluate_ranking.add_argument("--run-dir", required=True, type=Path)
    evaluate_ranking.add_argument("--experiment-dir", required=True, type=Path)
    evaluate_ranking.add_argument("--models", nargs="+", default=[
        "popularity", "nearest", "category_preference", "frozen_embedding"])
    evaluate_ranking.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    evaluate_ranking.add_argument("--overwrite", action="store_true")

    visualize_ranking = commands.add_parser(
        "visualize-ranking", help="Render an authenticated observed-only R9 ranking explanation"
    )
    visualize_ranking.add_argument("--run-dir", required=True, type=Path)
    visualize_ranking.add_argument("--experiment-dir", required=True, type=Path)
    visualize_ranking.add_argument("--overwrite", action="store_true")

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


def _named_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, root = value.partition("=")
        if not separator or not name or not root or name in result:
            raise ValueError("--experiment-dir must use unique NAME=ROOT values")
        result[name] = Path(root)
    return result


def _simulate(args: argparse.Namespace) -> dict[str, Any]:
    from . import simulator

    config_path = Path(args.config).expanduser().resolve()
    config = simulator.load_config(config_path)
    run = config["run"]
    requested_scenario = args.scenario if args.scenario is not None else run["scenario"]
    values = {
        "users": args.users,
        "days": args.days,
        "start_date": args.start_date,
        "seed": args.seed,
        "scenario": requested_scenario,
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
    requested_scenario, resolved_scenario = simulator.resolve_scenario(config, requested_scenario)
    run["requested_scenario"] = requested_scenario
    run["resolved_scenario"] = resolved_scenario
    args.requested_scenario = requested_scenario
    args.scenario = resolved_scenario
    run["scenario"] = resolved_scenario
    simulator.activate_config(config)
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
    from .benchmark import run_offline_benchmark, run_online_benchmark
    run.validate(require_truth=False)
    offline = run_offline_benchmark(run.observed, experiment.prepared,
        {"baseline": experiment.baseline_embeddings, "learned": experiment.embeddings},
        experiment.offline_benchmark, load_config(config_path), warmup=warmup,
        iterations=iterations, overwrite=overwrite)
    online = run_online_benchmark(run.observed, experiment.prepared, experiment.checkpoint,
        experiment.online_workload, experiment.online_benchmark, load_config(config_path),
        warmup=warmup, iterations=iterations, overwrite=overwrite)
    return {"offline": offline, "online": online}


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "simulate":
        result = _simulate(args)
    elif args.command == "inspect-evidence":
        from .artifact_index import inspect_evidence_indexes
        result = inspect_evidence_indexes(args.index_dir)
    elif args.command == "simulate-pair":
        from .simulate_pair import simulate_pair
        result = simulate_pair(args.config, args.reference_run_dir, args.intervention_run_dir,
                               args.pair_dir, intervention=args.intervention, users=args.users,
                               days=args.days, seed=args.seed, scenario=args.scenario)
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
        if (args.ranking_predictions is None) != (args.ranking_reports is None):
            raise ValueError("--ranking-predictions and --ranking-reports must be supplied together")
        if args.ranking_predictions is not None:
            from .ranking_pair_evaluation import evaluate_ranking_pair
            pair_layout = PairLayout.from_manifest_path(args.pair_manifest)
            result["ranking"] = evaluate_ranking_pair(args.pair_manifest, args.ranking_predictions,
                args.ranking_reports, pair_layout.exposure_counterfactual_json, overwrite=args.overwrite)
    elif args.command == "evaluate-change":
        from .evaluation import evaluate_change
        result = evaluate_change(args.pair_manifest, args.baseline_experiment_dir,
                                 args.learned_experiment_dir, overwrite=args.overwrite)
    elif args.command == "audit-nonstationarity":
        from .nonstationarity import audit_nonstationarity
        result = audit_nonstationarity(args.no_change_report, args.temporary_report,
            args.sustained_report, args.output_dir,
            adaptation_threshold=args.adaptation_threshold,
            recovery_threshold=args.recovery_threshold, overwrite=args.overwrite)
    elif args.command == "audit-privacy":
        from .privacy_audit import audit_privacy
        result = audit_privacy(run_dir=args.run_dir, experiments=_named_roots(args.experiment_dir),
            evidence_dir=args.evidence_dir, utility_report_dir=args.utility_report_dir,
            config_path=args.config, output_dir=args.output_dir, overwrite=args.overwrite)
    elif args.command == "calibrate-reliability":
        from .calibration import calibrate_reliability
        result = calibrate_reliability(args.run_dir, _named_roots(args.experiment_dir),
            args.config, args.output_dir, overwrite=args.overwrite)
    elif args.command == "context-pair-preflight":
        from .context_pair_preflight import run_context_pair_preflight
        result = run_context_pair_preflight(
            args.run_dir, args.experiment_dir, args.config, args.embedding_config, args.output_dir
        )
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
    elif args.command == "visualize-embeddings":
        from .embedding_visualization import visualize_embeddings
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        source = ((experiment.dense_embeddings if args.kind == "learned" else experiment.dense_baseline_embeddings)
                  if args.dense else
                  (experiment.embeddings if args.kind == "learned" else experiment.baseline_embeddings))
        if not source.is_file():
            raise FileNotFoundError(f"Missing {args.kind} embedding export: {source}")
        result = visualize_embeddings(source,
            experiment.visualization_artifact_dir(args.kind, dense=args.dense), dense=args.dense,
            reference_cutoff=args.reference_cutoff, normalization=args.normalization,
            reducer=args.reducer, seed=args.seed, image_format=args.format, overwrite=args.overwrite,
            umap_neighbors=args.umap_neighbors, umap_min_dist=args.umap_min_dist)
    elif args.command == "rank":
        from .ranking import run_ranking
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        manifest = run.validate(require_truth=False)
        result = run_ranking(run.observed, manifest, experiment.ranking_predictions(args.model),
                             experiment.ranking_report(args.model), model=args.model,
                             ks=args.k, overwrite=args.overwrite,
                             embedding_path=experiment.dense_embeddings,
                             checkpoint_path=(experiment.exposure_ranking_checkpoint if args.model == "exposure_aware"
                                              else experiment.frozen_ranking_checkpoint),
                             exposure_config=(load_mapping_config(args.ranking_config) if args.model == "exposure_aware" else None),
                             baseline_report_paths={name: experiment.ranking_report(name) for name in
                                 ("popularity", "nearest", "category_preference")})
    elif args.command == "user-journey":
        from .user_journey_visualization import build_user_journey_report
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        models = tuple(args.ranking_models or ("popularity", "nearest"))
        unknown = sorted(set(models) - {"popularity", "nearest", "category_preference", "frozen_embedding", "exposure_aware"})
        if unknown:
            raise ValueError(f"unsupported ranking models: {unknown}")
        result = build_user_journey_report(run, experiment, user_id=args.user_id,
            start=args.start, end=args.end, truth_access=args.evaluator_truth,
            ranking_models=models, max_events=args.max_events, max_requests=args.max_requests,
            max_candidates=args.max_candidates, overwrite=args.overwrite)
    elif args.command == "evaluate-ranking":
        from .ranking_evaluation import DEFAULT_MODELS, evaluate_ranking_transfer
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        run.validate(require_truth=False)
        unknown = sorted(set(args.models) - set(DEFAULT_MODELS))
        if unknown:
            raise ValueError(f"unsupported ranking models: {unknown}")
        result = evaluate_ranking_transfer(run.observed, experiment.ranking_dir,
            experiment.ranking_transfer_slices, models=args.models, ks=args.k,
            overwrite=args.overwrite)
    elif args.command == "visualize-ranking":
        from .ranking_visualization import render_ranking_explanation
        run = DatasetLayout.from_path(args.run_dir)
        experiment = ExperimentLayout.from_path(args.experiment_dir)
        run.validate(require_truth=False)
        result = render_ranking_explanation(
            run.observed, experiment.ranking_dir, experiment.ranking_visualization_dir,
            overwrite=args.overwrite,
        )
    elif args.command == "pipeline":
        result = _pipeline(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the CLI with stable failure categories and JSON report semantics."""
    args = build_parser().parse_args(argv)
    try:
        result = _run_command(args)
    except ScientificMetricUnavailable as error:
        # Lack of coverage/support is a valid scientific result when report generation
        # itself succeeded.  Keep it machine-readable and do not turn it into failure.
        result = {
            "status": "unavailable",
            "section": error.section,
            "reason": error.reason,
        }
    except Exception as error:
        exit_code, message = _error_message(error)
        print(f"geoembed: error: {message}", file=sys.stderr)
        raise SystemExit(exit_code) from None
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
