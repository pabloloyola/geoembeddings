# GeoEmbeddings unified codebase v0.5.0

The current code writes `geoembeddings-dataset/2.0` and explicitly retains
event-only 1.0 compatibility. It includes paired simulator interventions and
evaluation, protected change evaluation, component exports, a factorized
persistent/context model family, an observable recommendation contract,
reliability evaluation, and an offline benchmark. These executable surfaces do
not by themselves establish factorization, causal invariance, or external validity.

Start with [`START_HERE.md`](START_HERE.md). A Codex agent must also read the
repository-root [`AGENTS.md`](AGENTS.md) before changing code.

One repository now owns the complete path from semi-synthetic Kanto data
generation to user-embedding training and protected evaluation. The simulator
and model remain logically separated by a versioned dataset contract, while a
single CLI resolves every filename and directory.

## Quick start

From the extracted repository root:

```bash
uv sync --extra dev
uv run geoembed --version
```

Run the complete, fast non-learned path:

```bash
uv run geoembed pipeline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_baseline \
  --mode baseline
```

This generates and validates the data, fits train-only preprocessing, exports
the statistical embedding, and evaluates it against protected simulator truth.

Run the learned GRU path:

```bash
uv run geoembed pipeline \
  --run-dir runs/kanto_pilot_learned \
  --experiment-dir experiments/kanto_single_vector \
  --mode learned
```

The default simulator configuration creates 500 users over 14 days. For a
smaller plumbing test, add `--users 50 --days 7`.

The learned encoder uses an MPS-safe padded GRU and mask-based final-state
selection on Apple Silicon. Training validates all categorical IDs, targets,
sequence lengths, and continuous values on CPU before accelerator execution.
Sequence lengths remain CPU control metadata, and every batch is validated before
the GRU runs. Categorical tensor columns use the explicit order stored in
`prepared_metadata.json`; JSON key ordering is never treated as a model schema.
To force CPU execution for diagnosis, set `training.device: cpu` in
`configs/embedding/single_vector.yaml`.

## One path convention

Every command uses:

| Argument | Meaning |
|---|---|
| `--run-dir` | Dataset root created by the simulator |
| `--experiment-dir` | All artifacts for one embedding experiment |

Do not pass `observed/`, `truth/`, checkpoint names, or embedding filenames.
The shared layout code resolves them.

```text
runs/kanto_pilot/
├── manifest.json
├── config.resolved.yaml
├── validation_report.json
├── deep_validation_report.json
├── observed/
│   ├── users_observed.csv.gz
│   ├── observed_events.csv.gz
│   ├── poi_catalog.csv.gz
│   ├── recommendation_requests.csv.gz
│   ├── impressions.csv.gz
│   └── interactions.csv.gz
└── truth/
    ├── user_latents.csv.gz
    ├── episodes_truth.csv.gz
    ├── candidate_sets.csv.gz
    ├── choices_truth.csv.gz
    ├── trajectories_truth.csv.gz
    └── observation_process.csv.gz

experiments/kanto_single_vector/
├── prepared/
├── model/
├── embeddings.npz
└── evaluation.json
```

Training-related commands resolve only `observed/`. Protected evaluator
commands alone resolve and open `truth/`. Legacy event-only dataset/1.0 runs
remain readable for their supported modeling path; recommendation consumers
require the dataset/2.0 tables.

The CLI additionally provides `simulate-pair`, `pair-manifest`,
`validate-pair`, `evaluate-pair`, `evaluate-change`, `export-dense`,
supplemental `evaluate` modes (`--episodes`, `--transfer`,
`--temporal-routine`, `--reliability`), `robustness`, and `benchmark`. See
`docs/COMMAND_REFERENCE.md` for their immutable inputs and artifacts.

## Stage-by-stage commands

```bash
# Generate and validate
uv run geoembed simulate \
  --config configs/simulation/kanto_v1.yaml \
  --run-dir runs/kanto_pilot

uv run geoembed validate --run-dir runs/kanto_pilot

# Prepare model inputs
uv run geoembed prepare \
  --config configs/embedding/single_vector.yaml \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

# Non-learned comparator
uv run geoembed baseline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

uv run geoembed evaluate \
  --kind baseline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

# Learned encoder
uv run geoembed train \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

uv run geoembed export \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

uv run geoembed evaluate \
  --kind learned \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

# Fair frozen-embedding comparison (both exports in one experiment directory)
uv run geoembed compare \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

The baseline export is inexpensive and can share the learned experiment's
prepared data:

```bash
uv run geoembed baseline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector

uv run geoembed compare \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

`compare` writes `comparison/embedding_comparison.json` and a readable
`comparison/embedding_comparison.md`. It uses identical users, temporal cutoffs,
held-out-user splits, and frozen ridge probes for both representations. The
report covers latent-trait and category-preference recovery, signal beyond
home/work geography and activity volume, stability versus collapse, temporal
user retrieval, effective rank, event-count dependence, and common future-event
probes. Criteria that cannot be tested from three global cutoff embeddings are
explicitly marked as unavailable.

If the two exports live in different experiment directories, use:

```bash
uv run geoembed compare \
  --run-dir runs/kanto_pilot \
  --baseline-experiment-dir experiments/kanto_baseline \
  --learned-experiment-dir experiments/kanto_single_vector
```

The command rejects comparisons when the two prepared artifacts do not identify
the same source files and temporal split.

## Configuration

- `configs/simulation/kanto_v1.yaml`: world, population, episodes, choices,
  observation process, events, and controlled scenarios.
- `configs/embedding/single_vector.yaml`: temporal split, input fields, model,
  objective weights, training, and probing.

Command-line simulation overrides include `--users`, `--days`, `--start-date`,
`--seed`, `--scenario`, and `--full-kanto`. Resolved simulation and embedding
configurations are saved with their artifacts.

## Tests and visualization

```bash
uv run pytest

uv sync --extra viz
GEOEMBED_RUN_DIR=runs/kanto_pilot \
  uv run python scripts/kanto_visualization_validation.py
```

The visualization reads both observed and truth tables and is evaluator-only.
The notebook under `notebooks/` uses the same `GEOEMBED_RUN_DIR` variable.

## Repository map

```text
src/geoembeddings/        simulator, contract, CLI, training, export, evaluation
configs/simulation/       data-generating-process configuration
configs/embedding/        representation objective and training configuration
tests/                    contract, split, and model tests
scripts/                  validation visualization and explorer helpers
notebooks/                interactive simulator inspection
docs/                     data flow, objectives, evaluation, migration
references/               project paper list
```

See `docs/SIMULATION_FLOW.md` for the exact handoff and
`docs/OBJECTIVES_AND_EVALUATION.md` for the representation requirements.

## Agent and research documentation

- `START_HERE.md`: shortest route from extraction to a verified local run.
- `AGENTS.md`: binding instructions for a Codex coding agent.
- `docs/AGENT_HANDOFF.md`: architecture, evidence, risks, and ownership map.
- `docs/COMMAND_REFERENCE.md`: every command, input, operation, and output.
- `docs/REQUIREMENTS_MATRIX.md`: measurable R1--R13 research requirements.
- `docs/EXPERIMENT_PROTOCOL.md`: fair comparisons and artifact conventions.
- `docs/ROADMAP.md`: ordered simulator, evaluator, and model milestones.
- `docs/LITERATURE_GUIDE.md`: how the supplied mobility papers inform the work.
- `docs/VERIFICATION.md`: exact release tests and end-to-end smoke evidence.
- `TASKS.md`: executable backlog with acceptance gates.
