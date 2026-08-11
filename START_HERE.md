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

## First recommended development task

Implement the evaluator foundations in roadmap Phase 1 before introducing the
factorized model:

- episode-boundary and dense temporal embedding exports;
- controlled GPS, event-removal, and missing-service robustness tests;
- explicit held-out-region evaluation;
- matched-seed counterfactual exposure/opportunity comparisons.

These tests are necessary to determine whether a later factorized model truly
separates persistent preference, routine, and current context.
