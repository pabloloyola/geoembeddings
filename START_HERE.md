# Start here

This repository is a research harness for learning cross-service user
representations from semi-synthetic Kanto mobility and commerce histories. It
contains both the data-generating process and the embedding pipeline, separated
by a versioned observed/truth contract.

## For a Codex agent

Read these files in order before editing code:

1. `AGENTS.md`
2. `docs/AGENT_HANDOFF.md`
3. `docs/OBJECTIVES_AND_EVALUATION.md`
4. `docs/REQUIREMENTS_MATRIX.md`
5. `docs/EXPERIMENT_PROTOCOL.md`
6. `docs/ROADMAP.md`
7. `docs/VERIFICATION.md`
8. `TASKS.md`

Then inspect the implementation and tests associated with the first selected
task. Do not start by replacing the model. The highest-priority work is to make
the requirements measurable before optimizing them.

## Reproduce the tested path

```bash
uv sync --locked --extra dev
uv run pytest

uv run geoembed pipeline \
  --run-dir runs/reproduction_learned \
  --experiment-dir experiments/reproduction_single_vector \
  --mode learned \
  --users 50 \
  --days 7

uv run geoembed baseline \
  --run-dir runs/reproduction_learned \
  --experiment-dir experiments/reproduction_single_vector

uv run geoembed compare \
  --run-dir runs/reproduction_learned \
  --experiment-dir experiments/reproduction_single_vector
```

For the 500-user reference scale, omit `--users 50 --days 7`.

## Understand the two roots

- `--run-dir` is one immutable simulated dataset instance.
- `--experiment-dir` is one modeling attempt on a dataset.

Never point `--run-dir` at `observed/` or `truth/`. Training-related commands
may read only `observed/`; protected simulator truth is evaluator-only.

## Current contract and recommended task

New simulations write `geoembeddings-dataset/2.0`, including the public POI,
recommendation-request, impression, and interaction tables. Event-only 1.0 runs
remain explicitly readable by the legacy modeling path; readers never fabricate
missing 2.0 recommendation tables.

Evaluator foundations, paired interventions/change evaluation, component
exports, and the persistent/context implementation now exist. The immediate
scientific task is the T2.7 matched-factorization gate in `TASKS.md`: generate
immutable statistical, capacity-matched single-vector, factorized, and ablation
artifacts on identical data/cutoffs and evaluate the required axes separately.
Do not open a routine branch until that gate passes. If that matched run is not
next, T3.4 observable naive rankers are the independent implementation option.
