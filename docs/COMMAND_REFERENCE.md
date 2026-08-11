# Command reference

All commands are run from the extracted repository root:

```bash
uv sync --locked --extra dev
uv run geoembed --version
```

Default configuration paths are relative to the current working directory, so
run from the repository root or pass explicit `--config` paths.

## Path model

Two roots identify all artifacts:

| Argument | Meaning | Created/consumed by |
|---|---|---|
| `--run-dir` | One simulator dataset instance | simulation, validation, all later stages |
| `--experiment-dir` | One embedding experiment on a run | prepare, model, export, evaluation, comparison |

`--run-dir` must be the parent of `observed/` and `truth/`. Do not pass either
subdirectory. The CLI resolves internal paths centrally.

Canonical dataset tree:

```text
RUN_DIR/
├── config.resolved.yaml
├── manifest.json
├── validation_report.json
├── deep_validation_report.json
├── observed/
│   ├── users_observed.csv.gz
│   └── observed_events.csv.gz
└── truth/
    ├── user_latents.csv.gz
    ├── episodes_truth.csv.gz
    ├── candidate_sets.csv.gz
    ├── choices_truth.csv.gz
    ├── trajectories_truth.csv.gz
    └── observation_process.csv.gz
```

Canonical experiment tree after all stages:

```text
EXPERIMENT_DIR/
├── prepared/
│   ├── config.resolved.yaml
│   ├── prepared_metadata.json
│   └── vocabularies.json
├── model/
│   ├── best_model.pt
│   └── training_report.json
├── embeddings.npz
├── dense_embeddings.npz
├── statistical_baseline.npz
├── evaluation.json
├── baseline_evaluation.json
└── comparison/
    ├── embedding_comparison.json
    └── embedding_comparison.md
```

## `simulate`

### Purpose

Generate one complete versioned Kanto dataset, including public observed tables,
protected truth tables, the resolved configuration, manifest, and fast internal
validation.

### Command

```bash
uv run geoembed simulate \
  --config configs/simulation/kanto_v1.yaml \
  --run-dir runs/kanto_pilot
```

### Inputs

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--config PATH` | No | `configs/simulation/kanto_v1.yaml` | Simulation YAML |
| `--run-dir PATH` | Yes | — | New dataset root |
| `--users INT` | No | YAML `run.users` | Cohort size; minimum 10 |
| `--days INT` | No | YAML `run.days` | Simulated days; minimum 2 |
| `--start-date DATE` | No | YAML | ISO start date |
| `--seed INT` | No | YAML | Generator seed |
| `--scenario NAME` | No | YAML | Controlled scenario |
| `--full-kanto` / `--no-full-kanto` | No | YAML | Include full geographic population support |
| `--overwrite` | No | false | Replace the exact existing run directory |

Valid scenario names are `clean`, `mixed`, `opportunity_confounded`,
`exposure_confounded`, and `observation_biased`.

### Operations

1. Load and validate the YAML.
2. Apply command-line overrides and save the resolved values.
3. Create spatial regions, synthetic POIs, users, and persistent latents.
4. Generate daily episodes, candidate sets, utility-based choices, and true
   trajectories.
5. Apply service adoption, recording/dropout, GPS noise, and event processes.
6. Split public observed data from protected truth.
7. Run fast integrity checks and write all tables.

### Outputs

- `observed/users_observed.csv.gz`: public user attributes and service adoption.
- `observed/observed_events.csv.gz`: chronological observed cross-service events.
- six protected `truth/*.csv.gz` tables.
- `config.resolved.yaml`: exact data-generating settings.
- `manifest.json`: version, contract, seed, row counts, splits, and validation.
- `validation_report.json`: fast in-process integrity diagnostics.

### Reads protected truth?

It creates truth as part of the data-generating process. This is not a training
stage.

### Important behavior

The output directory must not exist unless `--overwrite` is supplied. Use a new
run directory for each scientific condition; do not overwrite evidence needed
for comparisons.

## `validate`

### Purpose

Run deeper structural and behavioral diagnostics on an existing simulator run.

### Command

```bash
uv run geoembed validate --run-dir runs/kanto_pilot
```

Optional custom report path:

```bash
uv run geoembed validate \
  --run-dir runs/kanto_pilot \
  --output reports/kanto_pilot_validation.json
```

### Inputs

- complete `RUN_DIR/observed/` and `RUN_DIR/truth/` tables;
- `RUN_DIR/manifest.json`;
- optional `--output PATH`.

### Operations

Checks keys and references, truth/observed separation, passive trajectory
matching, GPS plausibility, candidate/choice consistency, distance/exposure and
utility relationships, service overlap, event density, and related behavioral
diagnostics. The exact current checks live in `simulation_validation.py`.

### Output

By default: `RUN_DIR/deep_validation_report.json`.

The command exits with an error if the report status is not `passed`.

### Reads protected truth?

Yes. It is a simulator evaluator, not model training.

## `prepare`

### Purpose

Fit the leakage-safe preprocessing and temporal split contract for one
experiment. It does **not** materialize transformed tensors.

### Command

```bash
uv run geoembed prepare \
  --config configs/embedding/single_vector.yaml \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

### Inputs

- `RUN_DIR/observed/users_observed.csv.gz`
- `RUN_DIR/observed/observed_events.csv.gz`
- embedding YAML

### Operations

1. Validate the observed schema and reject truth-like columns.
2. Sort events by user and timestamp.
3. Compute global chronological train and validation cutoffs.
4. Select explicit categorical and continuous fields.
5. Fit categorical vocabularies on training events only.
6. Fit continuous normalization statistics on training events only.
7. Hash the two observed source files.
8. Store all field orders, counts, cutoffs, and resolved settings.

### Outputs

Under `EXPERIMENT_DIR/prepared/`:

- `vocabularies.json`: `<PAD>=0`, `<UNK>=1`, train-known tokens after that;
- `prepared_metadata.json`: source hashes, cutoffs, field order, vocabulary
  sizes, statistics, row counts, and target counts;
- `config.resolved.yaml`: exact embedding configuration.

### Reads protected truth?

No. It receives only the resolved `observed/` path.

### Important behavior

`train`, `baseline`, and `export` reread the observed CSVs and encode them under
this contract. If the source files, input fields, split fractions, vocabulary,
or normalization logic changes, rerun `prepare` and downstream stages.

## `baseline`

### Purpose

Export a non-learned representation using the exact same preprocessing and
cutoffs as learned models.

### Command

```bash
uv run geoembed baseline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

Pass `--config` if not using the default embedding YAML.

### Inputs

- observed event CSV;
- `EXPERIMENT_DIR/prepared/vocabularies.json`;
- `EXPERIMENT_DIR/prepared/prepared_metadata.json`;
- embedding configuration for maximum history length.

### Operations

At train, validation, and final test cutoffs for each user:

1. take at most the latest `max_sequence_length` events;
2. form normalized histograms for every categorical field;
3. append means and standard deviations of normalized continuous features;
4. concatenate the components without fitting model parameters.

### Output

`EXPERIMENT_DIR/statistical_baseline.npz`, containing:

- `user_id`: string array;
- `cutoff`: `train`, `validation`, or `test`;
- `embedding`: dense 2-D float array.

### Reads protected truth?

No.

## `train`

### Purpose

Train the current single-vector sequence encoder and select the checkpoint with
minimum validation loss.

### Command

```bash
uv run geoembed train \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

### Inputs

- observed events;
- prepared field/vocabulary/statistics/split contract;
- embedding YAML with model, objectives, and training settings.

### Window semantics

For each target event in the requested split, the input contains up to the
previous `max_sequence_length` observed events for the same user. The target
event is never part of its own history. Histories shorter than
`min_history_events` are not used as prediction windows.

### Model operations

1. Embed each categorical field separately.
2. Sum categorical embeddings and combine with projected continuous features.
3. Run an MPS-safe padded GRU.
4. Select the final valid state with a floating mask.
5. Project to the 128-dimensional user-history embedding.
6. Predict configured next-event fields.
7. Align early/late history representations with cosine consistency.
8. Apply event dropout during training.

Before accelerator execution, every batch is checked for field count, ID/target
ranges, valid lengths, and finite continuous values.

### Outputs

Under `EXPERIMENT_DIR/model/`:

- `best_model.pt`: state, resolved config, vocabularies, explicit field order,
  continuous fields, selected epoch, and validation metrics;
- `training_report.json`: device, fields, window counts, best validation loss,
  checkpoint path, and metrics for every epoch.

### Reads protected truth?

No.

### Device behavior

`training.device: auto` prefers CUDA, then Apple MPS, then CPU. Set it to `cpu`
for diagnosis. Do not move sequence lengths to MPS; they are CPU metadata.

## `export`

### Purpose

Run the best checkpoint over complete user histories at three cutoffs and save
frozen learned embeddings.

### Command

```bash
uv run geoembed export \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

### Inputs

- observed events;
- prepared contract;
- `EXPERIMENT_DIR/model/best_model.pt`;
- embedding config for batch/device settings.

### Operations

For each user and each cutoff with at least one observed event, select up to the
latest `max_sequence_length` events, encode with augmentation disabled, and emit
one vector.

### Output

`EXPERIMENT_DIR/embeddings.npz`, containing `user_id`, `cutoff`, and
`embedding` arrays with the same structural convention as the baseline export.

### Reads protected truth?

No.

## `export-dense`

### Purpose

Export learned histories after observed events so a protected evaluator can
later align them to episode boundaries and change points. This command reads
only `observed/`; it never reads or writes episode identifiers or other truth
labels.

### Command

```bash
uv run geoembed export-dense \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector \
  --event-stride 1
```

`--event-stride N` retains the first event, every Nth event thereafter, and the
last event for each user. It defaults to `1`. Each embedding uses at most the
configured `max_sequence_length` most recent events, while
`history_event_count` records the total observed history available at that
timestamp.

### Output

`EXPERIMENT_DIR/dense_embeddings.npz` contains row-aligned arrays:

- `user_id`: public user identifier;
- `timestamp`: ISO timestamp of the latest included observed event;
- `cutoff_kind`: currently the constant `observed_event`;
- `embedding`: learned frozen vector;
- `history_event_count`: total observed events available for that user.

It also stores `categorical_fields` and `continuous_fields` metadata arrays in
the checkpoint's explicit model-input order. These metadata arrays describe the
schema and are not row-aligned.

The export intentionally contains no episode IDs or protected labels. A later
evaluator may join these timestamps to `truth/` without changing the model
input boundary.

## `evaluate`

### Purpose

Evaluate one learned or baseline export using protected simulator truth.

### Learned command

```bash
uv run geoembed evaluate \
  --kind learned \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

### Baseline command

```bash
uv run geoembed evaluate \
  --kind baseline \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

`--kind` defaults to `learned`.

### Inputs

- the corresponding `.npz` export;
- protected `truth/user_latents.csv.gz`;
- prepared contract and embedding config;
- for learned evaluation, the checkpoint and observed events for test
  next-event metrics.

### Operations

- learned only: test next-event loss, top-1, and top-5 metrics;
- both: held-out-user ridge probes for persistent latent traits using test
  cutoff embeddings;
- both: cosine stability across train, validation, and test cutoffs;
- emit explicit requirement coverage/missing status.

### Outputs

- learned: `EXPERIMENT_DIR/evaluation.json`;
- baseline: `EXPERIMENT_DIR/baseline_evaluation.json`.

### Reads protected truth?

Yes. This is the protected evaluator.

## `compare`

### Purpose

Compare statistical and learned frozen representations with identical users,
cutoffs, held-out-user split, and ridge penalty.

### Same experiment directory

```bash
uv run geoembed compare \
  --run-dir runs/kanto_pilot \
  --experiment-dir experiments/kanto_single_vector
```

### Separate experiment directories

```bash
uv run geoembed compare \
  --run-dir runs/kanto_pilot \
  --baseline-experiment-dir experiments/kanto_baseline \
  --learned-experiment-dir experiments/kanto_single_vector
```

Optional `--output-dir PATH` changes only the comparison-report destination.

### Inputs

- both embedding exports;
- both preparation metadata files;
- observed events;
- protected user latents;
- embedding evaluation settings.

The command rejects mismatched source hashes, cutoffs, categorical fields, or
continuous fields.

### Operations

On common users with all three cutoffs:

- held-out persistent-trait and category-preference ridge probes;
- incremental preference probes beyond home/work geography and event count;
- same-user and different-user temporal geometry;
- temporal identity retrieval and effective rank;
- activity-volume dependence;
- common frozen future-event classifiers;
- requirement coverage status.

### Outputs

By default under the learned experiment:

- `comparison/embedding_comparison.json`;
- `comparison/embedding_comparison.md`.

### Reads protected truth?

Yes. It is a protected comparative evaluator.

### Interpretation

There is intentionally no aggregate winner. Read stability together with
distinctiveness, effective rank, and retained information.

## `pipeline`

### Purpose

Convenience orchestration for a complete **from-scratch** baseline or learned
run.

### Learned pipeline

```bash
uv run geoembed pipeline \
  --run-dir runs/kanto_pilot_learned \
  --experiment-dir experiments/kanto_single_vector \
  --mode learned
```

### Baseline pipeline

```bash
uv run geoembed pipeline \
  --run-dir runs/kanto_pilot_baseline \
  --experiment-dir experiments/kanto_baseline \
  --mode baseline
```

Simulation arguments accepted by `simulate` also apply. The embedding config is
passed as `--embedding-config`, not `--config`.

### Stage order

For `--mode baseline`:

```text
simulate -> validate -> prepare -> baseline -> evaluate baseline
```

For `--mode learned`:

```text
simulate -> validate -> prepare -> train -> export -> evaluate learned
```

### Important behavior

- It always begins with simulation; it is not a resume command.
- Existing run directories fail unless `--overwrite` is supplied.
- Learned mode does not create the statistical baseline.
- It does not run `compare`.
- To resume after training, call `export`, `evaluate`, and `compare` separately.

## Inspecting artifacts

Readable JSON/YAML:

```bash
sed -n '1,220p' RUN_DIR/manifest.json
sed -n '1,220p' EXPERIMENT_DIR/prepared/prepared_metadata.json
sed -n '1,260p' EXPERIMENT_DIR/model/training_report.json
sed -n '1,260p' EXPERIMENT_DIR/comparison/embedding_comparison.md
```

NPZ shapes without modifying data:

```bash
uv run python - <<'PY'
import numpy as np
p = np.load("experiments/kanto_single_vector/embeddings.npz", allow_pickle=False)
print(p.files)
for key in p.files:
    print(key, p[key].shape, p[key].dtype)
PY
```

Checkpoint metadata:

```bash
uv run python - <<'PY'
import torch
p = torch.load(
    "experiments/kanto_single_vector/model/best_model.pt",
    map_location="cpu",
    weights_only=False,
)
for key in p:
    if key != "model_state":
        print(key, p[key])
PY
```

## Recommended transparent workflow

Use individual commands during development. Reserve `pipeline` for clean smoke
or reference runs. Individual commands make stage boundaries, inputs, outputs,
and rerun decisions visible and reduce accidental regeneration.

## Dense episode response (`evaluate --episodes`)

```bash
uv run geoembed export-dense --kind baseline --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed export-dense --kind learned --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Artifacts are `dense_statistical_baseline.npz`, `dense_embeddings.npz`, `baseline_episode_response.json`, and `episode_response.json`. `compare` rejects differing source hashes, users, timestamps/cutoffs, or bin edges and adds learned-minus-baseline episode deltas. Sparse exports remain valid; coverage reports missing users and bins.
