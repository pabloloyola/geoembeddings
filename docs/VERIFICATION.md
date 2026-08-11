# Release verification

Verification date: 2026-08-11 UTC.

## Environment

- Python 3.12.13
- `uv` 0.11.33
- PyTorch 2.13.0
- Device used for neural smoke run: CPU

## Package and tests

```bash
uv sync --locked --extra dev
uv run pytest
uv run geoembed --version
uv lock --check
```

Result:

```text
13 passed
geoembed 0.5.0
lock check passed
```

## End-to-end learned smoke

Command:

```bash
uv run geoembed pipeline \
  --run-dir runs/agent_handoff_smoke50 \
  --experiment-dir experiments/agent_handoff_smoke50 \
  --mode learned \
  --users 50 \
  --days 7 \
  --seed 20260811
```

Result:

- deep simulator validation: passed;
- observed users/events: 50 / 1,023;
- training/validation windows: 613 / 162;
- eight epochs completed;
- best validation loss: 5.6836;
- learned export: 45 users, 135 user/cutoff rows, 128 dimensions;
- test evaluation completed.

## Baseline and comparison smoke

The same run and preparation were used for `baseline`, baseline `evaluate`, and
`compare`.

Result:

- baseline export: 45 users, 135 rows, 685 dimensions;
- shared three-cutoff comparison users: 45;
- JSON and Markdown comparison reports produced successfully.

The sample is intentionally too small for scientific conclusions. For example,
the held-out probe set contains only nine users and fine-geohash future probes
have zero known-label coverage. This run verifies execution and contracts, not
model quality.

## Small-sample validation caution

A 20-user, 4-day run failed deep validation because it did not cover every
episode type and had insufficient cross-region overlap. This is expected from
stochastic small cohorts. Use at least the documented 50-user, 7-day smoke size
for the full `pipeline`; use unit tests for smaller plumbing cases.


## T1.2 episode evaluator

```bash
uv run pytest tests/test_episode_evaluation.py tests/test_dense_export.py tests/test_cli_paths.py
uv run geoembed export-dense --kind baseline --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed export-dense --kind learned --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect coverage and matched deltas. Tests cover exact boundaries, malformed/overlapping intervals, sparse exports, missing users, duplicate/non-monotonic timestamps, non-finite values, dimensions, and the direct `truth/` boundary. The observed-only dense exporter test runs without any `truth/` directory.

## T1.3 event-removal robustness

```bash
uv run geoembed robustness --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed robustness --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Defaults are versioned in `configs/embedding/single_vector.yaml` (seed
`20260811`; rates `0`, `0.1`, `0.25`, `0.5`). Inspect the two
`robustness/*_event_removal.json` reports for realized thinning, unencodable
keys, matched coverage, cosine drift, and persistent-probe degradation. Full
removal is intentionally reported as unencodable rather than imputed. Event
removal provides partial R7 evidence; it is one of the implemented deterministic
sensitivity views and does not establish real-noise robustness.

Same-run smoke evidence used `smoke/run`, `smoke/experiment`, seed `20260811`,
and 1,176 observed events. At requested rates 0/0.1/0.25/0.5, realized removals
were 0/134/320/582 and matched-row coverage was 0.9787/0.9574/0.9574/0.9504
for both representations. Learned-minus-baseline mean cosine drift was
0.0000/0.0106/0.0312/0.0517; the corresponding frozen-probe-degradation deltas
were approximately 0.0000/-0.0019/0.2637/0.2520. These smoke estimates are not
scientific evidence: only nine held-out probe users were available. Existing
episode comparison still ran; learned-minus-baseline within-episode cosine,
boundary change, and post-episode recovery were -0.0627, +0.0505, and -0.0907.
Persistent/preference probe mean-R2 deltas were -0.4870/-0.1655, so this trained
smoke model does not outperform the baseline on representation-quality axes.

## T1.4 deterministic robustness views

```bash
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect the matched view specifications and masks, realized perturbations,
coverage, cosine drift, and frozen-probe degradation. These executable GPS,
timestamp, service-removal, and truncation operators are deterministic
sensitivity tests; they are not calibration to real noise or evidence of causal
invariance.

## T1.5 spatial-transfer evaluation

```bash
uv run geoembed evaluate --transfer --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --transfer --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect train-scaled distance retrieval, boundary-pair cosine, held-out-region
coverage, and seen/unseen geohash slices separately. Empty or unknown-label
slices remain explicit coverage results. These tests do not measure unseen-POI
transfer and do not establish external geographic validity.
