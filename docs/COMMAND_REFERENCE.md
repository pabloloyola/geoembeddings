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
```

Dataset contract `geoembeddings-dataset/2.0` adds the four recommendation
tables. Readers retain read-only compatibility with `1.0` for event-only model
stages; they do not invent missing recommendation data. Recommendation
consumers must require `2.0`, and CSV column order is normative. Hakone request
generation uses documented synthetic YAML assumptions (24 km/h travel-speed
conversion, 12% temporary unavailability, ten displayed candidates), not facts
about real businesses or transport. Utility, probabilities, latent intent and
episodes, inaccessible alternatives, and counterfactual outcomes are excluded.

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

Protected matched-run declarations use a third canonical root:

```text
PAIR_DIR/
├── reference_input.yaml               # immutable resolved pair input
├── intervention_input.yaml            # immutable resolved pair input
├── pair_manifest.json
├── pair_integrity.json                 # T1.11d
├── behavioral_diagnostics.json         # T1.11e
├── counterfactual_comparison.json      # T1.11f
└── counterfactual_comparison.md        # T1.11f
```

`evaluate-pair` consumes the canonical protected pair manifest plus two ordered
experiment directories (reference then intervention) for each representation:

```bash
uv run geoembed evaluate-pair --pair-manifest PAIR_DIR/pair_manifest.json \
  --baseline-experiment-dir REF_BASELINE INT_BASELINE \
  --learned-experiment-dir REF_LEARNED INT_LEARNED
```

It requires the passing hash-matched `pair_integrity.json` and writes the two
versioned `counterfactual_comparison` reports. Existing reports require explicit
`--overwrite`. The command is evaluator-only and is the only stage that joins
protected invariant/intervention labels to frozen representation keys.

## `pair-manifest`

### Purpose

Declare two identity-compatible simulator runs for later R5/R7 pair validation
and evaluation. This evaluator/simulator-side artifact is never an input to
`prepare`, `baseline`, `train`, or `export`.

```bash
uv run geoembed pair-manifest \
  --reference-run-dir runs/reference \
  --intervention-run-dir runs/observation-intervention \
  --output pairs/observation/pair_manifest.json

uv run geoembed validate-pair \
  --pair-manifest pairs/observation/pair_manifest.json
```

`validate-pair` resolves both declared run roots through `DatasetLayout` and
writes the canonical sibling `pair_integrity.json`. It first authenticates the
pair declaration, run manifests, resolved configurations, source tables,
entity hashes, schemas, matching keys, and stream lineage. It then compares
public and protected tables at key/field granularity, recording row/entity
coverage, missing and duplicate keys, allowed changes, and bounded exact
mismatch samples. Stale or structurally invalid inputs abort without creating a
passing report; disallowed value differences create a failing report and make
the command fail. Future paired representation evaluation must authenticate a
current passing report before computing metrics.

This is an integrity check for a controlled synthetic intervention. It does not
show that simulator assumptions are calibrated to Tokyo/Kanto, and it does not
establish external causal validity. Because no standalone POI truth table is
currently emitted, the complete world is deterministically reconstructed from
its authenticated configuration and named world-stream seed for field-level
region and POI checks.

Both arguments must be dataset roots; internal observed/truth filenames cannot
be supplied. The command reads both complete run manifests and canonical tables,
infers `identity`, `observation`, `exposure`, or `opportunity` from scenario and
stream lineage, hashes every source, and writes schema
`geoembeddings-pair-manifest/1.0`. The contract records run roots, simulator and
dataset identities, manifest/config/source/entity hashes, intervention
parameters, invariant entity classes, allowed field changes, semantic matching
keys, stream lineage, and creation provenance.

The command rejects missing or malformed hashes, unsupported identity or pair
schemas, incompatible dataset contracts, ambiguous/reused matching keys,
overlapping invariant/change declarations, and identity mismatches in declared
invariants. Existing output is immutable by default. `--overwrite` is accepted
only for the exact canonical filename and only when the existing regular file
is itself a valid pair manifest; it does not overwrite arbitrary files.

## `simulate-pair`

Generate a complete configured exposure, opportunity, or observation pair and
execute its structural, field-integrity, and behavioral gates:

```bash
uv run geoembed simulate-pair \
  --intervention opportunity \
  --reference-run-dir runs/opportunity-reference \
  --intervention-run-dir runs/opportunity-intervention \
  --pair-dir pairs/opportunity \
  --users 50 --days 7 --seed 20260811
```

The command reads overrides, affected streams, invariants, permitted changes,
and expected diagnostics from the versioned simulation YAML. It creates both
dataset roots and the protected pair root, saves exact input YAML snapshots,
runs deep validation while requiring all structural integrity checks, creates
and validates the pair manifest, and writes `behavioral_diagnostics.json`. It
refuses to start when any destination exists and deliberately has no overwrite
option. Small smoke cohorts may miss deep coverage targets; those results stay
visible in each run's `deep_validation_report.json` and are not confused with
the separately enumerated structural gate.

The defaults are synthetic experimental assumptions, not measurements or
claims about Tokyo, Kanto, opportunity, exposure, or observation processes.

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
8. Derive semantic-key identities and serialize the versioned run-level
   identity/stream manifest. CSV row order is not an identity input.

### Outputs

- `observed/users_observed.csv.gz`: public user attributes and service adoption.
- `observed/observed_events.csv.gz`: chronological observed cross-service events.
- six protected `truth/*.csv.gz` tables.
- `config.resolved.yaml`: exact data-generating settings.
- `manifest.json`: version, contract, seed, row counts, splits, validation, and
  `identity` (`geoembeddings-simulation-identity/1.0`) with resolved streams,
  identity algorithm, entity counts, and deterministic identity-set hashes.
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
It also rejects a missing/unsupported identity schema, missing or non-integer
stream seeds, missing entity declarations, malformed hashes/semantic IDs,
duplicate primary identities, table/hash count disagreement, and disagreement
between top-level and identity-section stream provenance.

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

`training_report.json` also contains the shared `runtime_metadata` object
described under **Runtime metadata schema** below. The timed interval covers
dataset construction, model training, checkpoint selection, and report
construction up to metadata collection.

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

Learned exports use `geoembeddings-component-export/2.0`. `embedding` remains
an alias for `component_combined`; independently addressable
`component_persistent`, `component_context`, and `component_combined` arrays
are accompanied by names/dimensions, model variant, ordered input fields,
preparation/source hashes, boundaries, exported cutoffs, and compatibility
metadata. Evaluator readers migrate an unversioned legacy vector by mapping it
to `persistent` and `combined` with zero `context`; this preserves old artifacts
without claiming learned factorization.

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

For learned reports, `next_event` is a backward-compatible extension under
schema `geoembeddings-next-event-evaluation/2.0`: the existing loss, accuracy,
and top-5 keys remain, while `predictive_diagnostics` reports each target's
train-only class counts and majority label, known/unknown-label coverage,
learned and naive known-label accuracy, coverage-aware accuracy, macro-F1,
balanced accuracy, per-class support, and deltas. Unknown test labels are never
counted as correct merely because they map to the train-fitted unknown token;
zero known-label coverage is explicit and all aggregate metrics remain finite.
These are predictive diagnostics only. Beating the majority control is not
evidence of embedding quality, spatial transfer, or disentanglement.

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

Both formats expose the same `runtime_metadata` provenance (the Markdown report
renders it as a runtime section). The comparison duration covers input
validation, all frozen probes and optional supplemental-report merging.

## Runtime metadata schema

JSON reports produced by `train`, `evaluate` (base, episode, transfer, and
temporal-routine modes), `robustness`, and `compare` contain a top-level
`runtime_metadata` object using schema
`geoembeddings-runtime-metadata/1.0`. Collection uses process, package, source
checkout, and accelerator APIs only; it never opens `observed/` or `truth/`.

Required keys are always serialized, including explicit JSON `null` when a
portable value cannot be discovered:

| Field | Type | Semantics |
|---|---|---|
| `schema_version` | string | Runtime metadata contract identifier. |
| `python_version` | string | Running interpreter version. |
| `package_version` | string or null | Installed `geoembeddings` distribution version; null for an uninstalled source tree. |
| `pytorch_version` | string | Imported PyTorch version. |
| `operating_system` | string | Python platform description of the host OS. |
| `device_type` | string or null | PyTorch device class (`cpu`, `cuda`, or `mps`); null when the report does not execute model tensors. |
| `source_commit` | string or null | Git `HEAD` commit of the source checkout; null outside a discoverable Git work tree. |
| `wall_clock_duration_seconds` | finite non-negative number | Command-path interval measured with a monotonic clock, ending immediately before serialization. |
| `seed` | integer | Resolved command/config seed; never coerced to a floating-point value. |

`accelerator` is the only optional accelerator-specific value. It is an object
for CUDA/MPS execution (and may itself contain null values when an API cannot
name hardware) and is explicitly `null` for CPU or non-tensor evaluator paths.
Runtime duration is reproducibility and rough-cost provenance, not a normalized
cross-device benchmark.

### Reads protected truth?

Yes. It is a protected comparative evaluator.

### Interpretation

There is intentionally no aggregate winner. Read stability together with
distinctiveness, effective rank, and retained information.

## Spatial and transfer evaluation (`evaluate --transfer`)

Run both frozen representations against the identical versioned slice contract:

```bash
uv run geoembed evaluate --transfer --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --transfer --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

The first two commands write `baseline_transfer_evaluation.json` and
`learned_transfer_evaluation.json`. `compare` rejects differences in observed
source hashes, users, cutoffs, fitted training contract, slice-definition hash,
or slice coverage, then adds independent R2/R8 deltas. It never creates an
aggregate spatial score. Thresholds, known regions, and known geohashes are fit
from training events only; held-out and unseen labels affect evaluation slices
and coverage only.

## `robustness`

Construct versioned, deterministic observed-data views and evaluate R6/R7:

```bash
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

The versioned specifications live under `evaluation.robustness` in the embedding
YAML. Each stable `view_id` hashes its view kind, parameters, and specification
version. GPS offsets and timestamp jitter, service selection, and truncation are
keyed by the observed-event source hash, seed, user, original timestamp, and
canonical row discriminator; they do not use global RNG state and do not depend
on CSV row order. `recent-truncation` removes the configured number of most
recent observed events per user; leave-one-service-out removes exactly the
configured public `service_id`.

View exports are `robustness/{kind}/{view_id}.npz`; reports are
`robustness/{kind}_robustness.json`. Exports and reports contain the source and
specification hashes, algorithm/version, seed, view IDs/specifications, mask
hashes, field order, requested and realized corruption, encoded keys, coverage,
cosine drift, frozen-probe degradation, and explicit insufficient-history
exclusions. Empty views stay unencodable and never fall back to clean history.
View construction and encoding read only `observed/`; the CLI validates and
opens `truth/` only after encoding, for protected frozen probes. `compare`
rejects mismatched specifications, hashes, masks, view IDs, keys, coverage,
users, or cutoffs before reporting distinct learned-minus-baseline R6 and R7
axes. These are controlled sensitivity tests, not causal or real-noise claims.

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

### `scripts/index_artifacts.py`

Create the T0.1a evidence inventory only after the baseline and learned
cutoff/dense exports, base/episode/transfer/temporal-routine/reliability
evaluations, robustness views, offline benchmark, and comparison reports have
been generated from the same preparation:

```bash
uv run python scripts/index_artifacts.py \
  --run-dir runs/reference500 \
  --experiment-dir experiments/reference500 \
  --output docs/artifacts/t0.2-reference500.json \
  --task-id T0.2
```

The command never edits generated run or experiment artifacts. It resolves
canonical filenames through `DatasetLayout` and `ExperimentLayout`, hashes each
file as raw bytes with SHA-256, and writes only `--output`. Local identifiers
are normalized relative to the repository when possible; external URI schemes
and hosts are normalized without retrieving their content.

The output schema is `geoembeddings-evidence-index/1.0`:

| Field | Meaning |
|---|---|
| `schema_version`, `task_id`, `index_location` | Index identity and normalized destination |
| `provenance` | Git commit, simulator-manifest hash, resolved simulation/training/evaluation seeds, cohort size, train/validation cutoffs, observed-source hashes, preparation contract, ordered categorical/continuous fields, and preparation-metadata hash |
| `evidence_identity.{baseline,learned}` | Exact user set/hash, cutoff set, source hashes, and preparation identity for each representation |
| `required_artifacts.shared` | Manifest, resolved simulator/embedding configs, deep validation, preparation metadata, and vocabularies |
| `required_artifacts.baseline` | Cutoff/dense exports and base, episode, robustness, transfer, temporal/routine, and reliability reports |
| `required_artifacts.learned` | Checkpoint/training report, cutoff/dense exports, and base, episode, robustness, transfer, temporal/routine, and reliability reports |
| `required_artifacts.robustness_views` | Every baseline and learned versioned robustness NPZ |
| `required_artifacts.comparison` | JSON and Markdown matched-comparison reports |
| `required_artifacts.benchmarks` | Matched observed-only offline benchmark report |
| `comparability_audit` | Separate source, cutoff, field-order, user-set, and preparation-contract checks with blocking diagnostics |

Indexing aborts before writing output if required artifacts are absent, an
embedding is non-finite, observed files no longer match preparation hashes, or
baseline and learned users/cutoffs/report provenance differ. This is an
identity and completeness audit, not an aggregate scientific winner.

### `scripts/reconcile_status.py`

After a T0.2 index passes its comparability audit, authenticate and reconcile
the indexed reports without calculating an aggregate winner:

```bash
uv run python scripts/reconcile_status.py \
  --artifact-index docs/artifacts/t0.2-reference500.json \
  --output docs/decisions/t0.2a-reference-decision.md
```

The command verifies the T0.2 index schema and passing audit, baseline/learned
source hashes, preparation identity, cutoffs, ordered categorical and
continuous fields, user identity, robustness specifications and masks, and the
SHA-256 digest of every consumed report. Missing or remote-only report files
abort reconciliation rather than allowing the index to stand in for the
scientific evidence. The output keeps persistent probes, incremental
information, collapse/geometry, episode response, robustness, and next-event
performance/coverage separate; every axis includes coverage and missingness
qualifications and exactly one permitted next action.

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

## Temporal and routine diagnostics (`evaluate --temporal-routine`)

```bash
uv run geoembed evaluate --temporal-routine --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --temporal-routine --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

The commands consume the corresponding dense export and write
`baseline_temporal_routine.json` or `learned_temporal_routine.json`. Only the
evaluator joins public timestamps to protected episode and intent labels. All
temporal definitions live under `evaluation.temporal_routine` in embedding YAML.
Reports keep cyclic probes, duration tasks, periodic user/state retrieval,
repeated-routine-versus-one-off classification, coverage, separation, and
effective rank separate. `compare` requires matched sources, dense keys,
definitions, split, and row coverage and computes no aggregate winner. Use
`simulate-pair --intervention schedule-shift` followed by `evaluate-pair` for
the controlled response axis; observational calendar probes are not a substitute.

## Reliability evaluation (`evaluate --reliability`)

```bash
uv run geoembed evaluate --reliability --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --reliability --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

The commands consume `statistical_baseline.npz` or `embeddings.npz` plus the
observed sources and preparation metadata; they do not open `truth/`. Outputs
are respectively `baseline_reliability.json` and `reliability.json`. Existing
reports are rejected unless `--overwrite` is explicit. Schema
`geoembeddings-reliability-report/1.0` records runtime metadata, seed,
representation kind, source hashes, preparation and cutoff identities,
resampling settings, per-user sample counts, insufficient users/bins, seeded
cutoff-bootstrap embedding variance, reliability-error bins, and coverage-risk
points. This is a repeatability diagnostic over the three frozen cutoffs, not
calibrated real-world uncertainty; event/window bootstrap is future work.

## Offline benchmark (`benchmark`)

```bash
uv run geoembed benchmark --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR \
  --warmup 1 --iterations 5
```

The observed-only command reads existing baseline and learned exports and writes
`benchmarks/offline.json` under schema
`geoembeddings-offline-benchmark/1.0`. Each representation reports workload,
artifact bytes/hash, frozen-export read/validation, in-memory export serialization, and reliability-evaluation
latency (mean/p50/p95/min/max), throughput, Python peak allocation, process peak
RSS, warmup/iteration counts, CPU/software metadata, and explicit missing
artifact status. It rejects source/preparation mismatch and existing output
unless `--overwrite` is passed. It never accepts a truth path and does not
measure training, online update latency, or hardware-normalized performance.

## `evaluate-change` (T1.15)

Generate either versioned intervention with at least one pre-change day and, for
a temporary trip, at least one post-change day:

```bash
uv run geoembed simulate-pair --intervention temporary-trip \
  --reference-run-dir runs/change-control --intervention-run-dir runs/change-trip \
  --pair-dir pairs/change-trip --users 50 --days 14 --seed 20260811
uv run geoembed evaluate-change --pair-manifest pairs/change-trip/pair_manifest.json \
  --baseline-experiment-dir experiments/control-baseline experiments/trip-baseline \
  --learned-experiment-dir experiments/control-learned experiments/trip-learned
```

Both experiment roots must already contain observed-only dense exports. The
protected evaluator authenticates the passing, current pair-integrity report,
then reads `truth/change_points_truth.csv.gz`; preparation, training, baseline,
and export never receive that path. The immutable outputs are
`change_evaluation.json` and `.md` beside the pair manifest. They report
relative-day matched-control drift curves, adaptation, recovery, forgetting,
permanent drift, coverage, exclusions, and right-censoring separately for the
statistical and learned representations. `--overwrite` is required to replace
both existing regular outputs.
