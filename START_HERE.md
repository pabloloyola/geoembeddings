# Start here

This repository is a research harness for learning cross-service user
representations from semi-synthetic Kanto mobility and commerce histories. It
contains both the data-generating process and the embedding pipeline, separated
by a versioned observed/truth contract.

**Running the project locally?** Follow the
**[local exploration runbook](docs/LOCAL_EXPLORATION.md)** for prerequisites, a
fresh 50-user smoke workflow, artifact inspection, ranking controls, optional
learned comparison, and troubleshooting.

Read **[external-validity and evidence boundaries](docs/EXTERNAL_VALIDITY.md)**
before interpreting any simulator or indexed result as evidence about real
people, places, services, or deployments.

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
task. Do not start by replacing the model. The evaluator foundations are now
implemented; preserve the completed negative T2.7 matched-factorization gate
before optimizing or adding a routine branch.

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
exports, the persistent/context implementation, and T3.4--T3.7 recommendation
work now exist. T3.5 provides the observed-only frozen-embedding candidate
ranker. T3.6 adds observed-only exposure-aware training; only its protected
paired evaluator may report utility regret and probability recovery, and only
after pair-integrity and ranking-identity authentication. T3.7 provides frozen
seen/unseen region and POI plus early/late slices from observed data alone, so
it cannot report utility regret. These synthetic results are neither real-world
causal evidence nor external-validity evidence.

T2.7 completed on matched immutable artifacts with a **do not advance**
decision because persistent and combined gates failed; do not open a routine
branch. T4.2 adaptation/forgetting auditing and T4.5 external-validity limits
are complete. Because no representation has the `selected_candidate` role,
T4.2 currently provides diagnostic-control evidence only, not a
selection-dependent R11 conclusion. The remaining open top-level tasks are T4.1
uncertainty calibration and T4.3 privacy audits.
