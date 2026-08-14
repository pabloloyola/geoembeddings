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

The root **[agent instructions](AGENTS.md)** are the binding policy for Codex
work in this repository. This document is the first document in the required
reading sequence; do not restart the sequence after arriving here.

Read these files in order before editing code:

1. `START_HERE.md` (this document)
2. `docs/AGENT_HANDOFF.md`
3. `docs/SIMULATION_FLOW.md`
4. `docs/OBJECTIVES_AND_EVALUATION.md`
5. `docs/REQUIREMENTS_MATRIX.md`
6. `docs/EXPERIMENT_PROTOCOL.md`
7. `docs/ROADMAP.md`
8. `docs/VERIFICATION.md`
9. `TASKS.md`

For a concise operational checklist and current-state transfer after completing
the required reading, see the **[next-agent handoff](docs/NEXT_AGENT_HANDOFF.md)**.

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

For a root-relative command-to-artifact index and reverse file lookup, use the
**[Command and artifact quick reference](docs/COMMAND_REFERENCE.md#command-and-artifact-quick-reference)**.

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
selection-dependent R11 conclusion. The remaining open top-level tasks are the
selection-dependent T4.1 uncertainty-calibration and T4.3 privacy conclusions;
the bounded T4.1a and T4.3a diagnostic-control surfaces are complete.
