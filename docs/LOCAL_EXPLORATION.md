# Local exploration runbook

This runbook is the shortest reproducible path from a clean checkout to
inspectable baseline and learned artifacts. It is intended for local plumbing
and scientific orientation, not for producing publication evidence. For every
CLI option and artifact contract, use the [command reference](COMMAND_REFERENCE.md).

## Prerequisites and scope

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for the locked environment and commands.
- Enough free disk for immutable datasets and experiments. Each pipeline keeps
  compressed observed and protected tables plus model artifacts; learned runs
  also keep a checkpoint and can use substantially more space than the small
  baseline below. Check free space before starting several runs.
- The 50-user, 7-day baseline is a smoke run, but simulation, validation, and
  evaluation can still take minutes depending on the machine. Learned training
  is the slow step and may take much longer on CPU. The default 500-user,
  14-day configuration is materially larger.
- Visualization is optional. Install its locked dependencies only when needed:

  ```bash
  uv sync --locked --extra dev --extra viz
  ```

From the repository root, create the normal locked development environment:

```bash
uv sync --locked --extra dev
```

The commands below deliberately use distinct, descriptive roots. Treat every
`--run-dir` as one immutable dataset and every `--experiment-dir` as one
immutable modeling attempt. Do not pass `RUN_DIR/observed/` or
`RUN_DIR/truth/` as `--run-dir`; always pass their parent. Do not reuse an
output root merely for convenience. Choose a fresh path unless you explicitly
intend and have validated a command's `--overwrite` behavior.

## Ordered exploration

### 1. Confirm the CLI

```bash
uv run geoembed --version
```

### 2. Run the test suite

```bash
uv run pytest
```

### 3. Build a small baseline from scratch

Use fresh paths (change the suffix if either already exists):

```bash
uv run geoembed pipeline \
  --run-dir runs/local_baseline_50u_7d \
  --experiment-dir experiments/local_baseline_50u_7d \
  --mode baseline \
  --users 50 \
  --days 7
```

`pipeline` always starts with simulation. It is not resumable, and it is not a
way to attach a new experiment to an existing dataset. To reuse an existing
dataset, call the stage commands (`validate`, `prepare`, `baseline`, `train`,
`export`, and `evaluate`) with that dataset root. A baseline pipeline executes
`simulate -> validate -> prepare -> baseline -> evaluate baseline`.

Optionally project and plot the baseline cutoff export after installing the
`viz` extra:

```bash
uv run --extra viz geoembed visualize-embeddings --experiment-dir experiments/local_baseline_50u_7d --kind baseline --reference-cutoff train --normalization standard --seed 1729 --format png
```

### 4. Inspect the baseline artifacts

Read the provenance and reports without modifying them:

```bash
sed -n '1,220p' runs/local_baseline_50u_7d/manifest.json
sed -n '1,220p' runs/local_baseline_50u_7d/validation_report.json
sed -n '1,260p' runs/local_baseline_50u_7d/deep_validation_report.json
sed -n '1,240p' experiments/local_baseline_50u_7d/prepared/prepared_metadata.json
sed -n '1,260p' experiments/local_baseline_50u_7d/baseline_evaluation.json
```

Inspect the baseline export's keys, shapes, and dtypes:

```bash
uv run python - <<'PY'
import numpy as np

path = "experiments/local_baseline_50u_7d/statistical_baseline.npz"
with np.load(path, allow_pickle=False) as artifact:
    for key in artifact.files:
        value = artifact[key]
        print(key, value.shape, value.dtype)
PY
```

Pay particular attention to the dataset contract and source hashes in the
manifest, validation status and coverage, the explicit categorical and
continuous field order in `prepared_metadata.json`, the three cutoff labels in
the export, and coverage/missing-requirement fields in the evaluation. JSON key
order is not a tensor schema.

### 5. Run all observable ranking controls

Use the **same dataset and experiment roots** for all three controls so their
request and candidate identities can be checked directly:

```bash
uv run geoembed rank \
  --run-dir runs/local_baseline_50u_7d \
  --experiment-dir experiments/local_baseline_50u_7d \
  --model popularity

uv run geoembed rank \
  --run-dir runs/local_baseline_50u_7d \
  --experiment-dir experiments/local_baseline_50u_7d \
  --model nearest

uv run geoembed rank \
  --run-dir runs/local_baseline_50u_7d \
  --experiment-dir experiments/local_baseline_50u_7d \
  --model category_preference
```

These are observable controls, not evidence of learned personalization or
counterfactual recommendation quality.

### 6. Optionally run the slower learned pipeline

Because `pipeline` begins with simulation, give this run its own fresh dataset
and experiment roots. The identical `--users`, `--days`, and default seed make
this a convenient smoke setup; artifact identity checks, not similar arguments,
determine whether later comparisons are fair.

```bash
uv run geoembed pipeline \
  --run-dir runs/local_learned_50u_7d \
  --experiment-dir experiments/local_learned_50u_7d \
  --mode learned \
  --users 50 \
  --days 7
```

Learned mode executes
`simulate -> validate -> prepare -> train -> export -> evaluate learned`. It
does not produce the statistical baseline and does not run `compare`.

### 7. Add the matched baseline and compare

Create and evaluate the baseline in the learned experiment, against the same
prepared dataset contract, then compare the frozen exports:

```bash
uv run geoembed baseline \
  --run-dir runs/local_learned_50u_7d \
  --experiment-dir experiments/local_learned_50u_7d

uv run geoembed evaluate \
  --kind baseline \
  --run-dir runs/local_learned_50u_7d \
  --experiment-dir experiments/local_learned_50u_7d

uv run geoembed compare \
  --run-dir runs/local_learned_50u_7d \
  --experiment-dir experiments/local_learned_50u_7d
```

Read each reported requirement axis separately. There is intentionally no
aggregate winner: stability can be caused by collapse and must be interpreted
with separation, retrieval, effective rank, task information, and coverage.

With the `viz` extra installed, create the corresponding learned and matched
baseline cutoff visualizations:

```bash
uv run --extra viz geoembed visualize-embeddings --experiment-dir experiments/local_learned_50u_7d --kind learned --reference-cutoff train --normalization standard --seed 1729 --format png
uv run --extra viz geoembed visualize-embeddings --experiment-dir experiments/local_learned_50u_7d --kind baseline --reference-cutoff train --normalization standard --seed 1729 --format png
```

Each projection is fitted separately. Baseline and learned projection axes are
therefore not aligned, so do not visually compare their positions as though the
plots shared coordinates.

### 8. Optional diagnostic surfaces

These commands assume that step 7 completed. Dense exports are prerequisites
for episode and temporal/routine evaluation.

```bash
# Dense exports
uv run geoembed export-dense --kind baseline --event-stride 1 --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed export-dense --kind learned --event-stride 1 --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Episode response
uv run geoembed evaluate --episodes --kind baseline --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed evaluate --episodes --kind learned --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Spatial transfer
uv run geoembed evaluate --transfer --kind baseline --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed evaluate --transfer --kind learned --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Temporal/routine diagnostics (consume dense exports)
uv run geoembed evaluate --temporal-routine --kind baseline --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed evaluate --temporal-routine --kind learned --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Robustness views
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind baseline --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind learned --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Reliability (observed-only)
uv run geoembed evaluate --reliability --kind baseline --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
uv run geoembed evaluate --reliability --kind learned --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d

# Offline frozen-artifact benchmark (observed-only)
uv run geoembed benchmark --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d --warmup 1 --iterations 5

# Merge available matched supplemental reports into comparison
uv run geoembed compare --run-dir runs/local_learned_50u_7d --experiment-dir experiments/local_learned_50u_7d
```

Optional visualization after installing the `viz` extra:

To visualize dense trajectories, first create the selected representation with
`export-dense` (as in the dense-export commands above), and then add `--dense`
to the corresponding `visualize-embeddings` command. Dense outputs are written
under `visualization/{kind}_dense/`, separately from cutoff visualizations.

```bash
GEOEMBED_RUN_DIR=runs/local_baseline_50u_7d \
  uv run python scripts/kanto_visualization_validation.py
```

The visualization reads protected truth and is evaluator-only.

Create a bounded, interactive Folium trajectory view with:

```bash
uv run --extra viz python scripts/kanto_trajectory_explorer.py \
  --run-dir runs/local_baseline_50u_7d \
  --date 2026-04-02 \
  --max-users 25 --seed 1729 --output visualizations/local_map.html
```

This defaults to observed-only access. Evaluator-only latent trajectories need
`--include-truth`; protected home/work anchors additionally need
`--include-anchors`. The output and companion metadata stay outside immutable
run directories, report deterministic truncation, and preserve source and filter
provenance. Opening the HTML normally requires network access to load
OpenStreetMap tiles.


## Artifact checklist

Artifacts are immutable stage outputs; absence often means the preceding stage
did not complete.

| After stage | Expected artifacts |
|---|---|
| Baseline simulation | `RUN_DIR/config.resolved.yaml`, `manifest.json`, `validation_report.json`, `observed/*.csv.gz`, and `truth/*.csv.gz` |
| Deep validation | `RUN_DIR/deep_validation_report.json` |
| Preparation | `EXPERIMENT_DIR/prepared/config.resolved.yaml`, `prepared_metadata.json`, and `vocabularies.json` |
| Baseline export/evaluation | `statistical_baseline.npz` and `baseline_evaluation.json` |
| Cutoff embedding visualization | `visualization/{learned,baseline}/projection_metadata.json`, `visualization/{learned,baseline}/projections.csv`, `visualization/{learned,baseline}/projections.npz`, `visualization/{learned,baseline}/small_multiples.png`, and `visualization/{learned,baseline}/trajectories.png` |
| Each ranker | `ranking/{model}.npz` and `ranking/{model}.json` |
| Learned training/export/evaluation | `model/best_model.pt`, `model/training_report.json`, `embeddings.npz`, and `evaluation.json` |
| Comparison | `comparison/embedding_comparison.json` and `comparison/embedding_comparison.md` |
| Dense exports | `dense_statistical_baseline.npz` and `dense_embeddings.npz` |
| Episode diagnostics | `baseline_episode_response.json` and `episode_response.json` |
| Transfer diagnostics | `baseline_transfer_evaluation.json` and `learned_transfer_evaluation.json` |
| Temporal/routine diagnostics | `baseline_temporal_routine.json` and `learned_temporal_routine.json` |
| Robustness | `robustness/{kind}/{view_id}.npz` and `robustness/{kind}_robustness.json` |
| Reliability | `baseline_reliability.json` and `reliability.json` |
| Benchmark | `benchmarks/offline.json` |

## Troubleshooting

### Small-cohort deep-validation coverage

A 50-user run is deliberately small. Rare services, regions, episode types, or
recommendation interactions may have weak or zero coverage, and a deep
behavioral check can fail or a metric can be unavailable. Read
`deep_validation_report.json` and the report's denominators rather than
silencing the check. Reproduce the command and seed, then increase users or days
in a **new** run root when the diagnostic requires more support.

### Force CPU execution

`training.device: auto` prefers CUDA, then Apple MPS, then CPU. For diagnosis,
copy the embedding YAML to a new versioned config, set `training.device: cpu`,
and pass it as `--embedding-config` to `pipeline` (or `--config` to the relevant
stage command). Do not edit the resolved config inside an existing experiment.

### Apple MPS constraints

The GRU intentionally uses padded execution and a floating-mask final-state
selector. Sequence lengths remain CPU control metadata. Do not replace this
with `pack_padded_sequence` or integer `gather` on MPS; those paths caused Metal
failures. If an accelerator issue occurs, preserve the failing report and use a
fresh CPU-configured experiment to distinguish device behavior from bad data.

### Missing histories at early cutoffs

Users can have no observed history at an early cutoff. Exports may legitimately
omit those user/cutoff rows, while training windows must still satisfy
`min_history_events`. Inspect cutoff coverage and exclusion counts; do not add
future events, loosen temporal ordering, or fabricate empty histories to make a
smoke report look complete.

### Ranking rejects dataset 1.0

`geoembed rank` requires the observable recommendation tables introduced by
`geoembeddings-dataset/2.0`. It intentionally rejects event-only dataset 1.0
runs rather than inventing catalogs, requests, impressions, or interactions.
Create a fresh dataset with the current simulator; do not patch generated
manifests or tables.

## Interpretation boundary

Smoke metrics demonstrate only that local contracts and command paths execute.
They are not scientific evidence of persistent preference recovery,
factorization, causal invariance, geographic transfer, robustness to real
missingness, or recommendation utility. Do not compare outputs with different
source hashes, cutoffs, user sets, field orders, or candidate sets, and do not
promote one scalar as an aggregate winner.
