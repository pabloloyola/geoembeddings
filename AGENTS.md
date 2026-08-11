# Instructions for Codex agents

## Mission

Improve the simulator, representation models, and evaluation harness so that a
user embedding can support cross-service personalization and recommendation in
new contexts. The central scientific question is:

> Can a representation recover persistent user preferences while adapting to
> recurring routines and temporary episode state, without mistaking geographic
> opportunity, platform exposure, observation missingness, or one trip for
> long-term preference?

The intended representation is factorized as

\[
z_{u,t}=(p_u,r_{u,t},c_{u,t}),
\]

where `p` is persistent preference, `r` is recurring routine, and `c` is current
episode or intent. The implemented GRU emits a single vector and is a diagnostic
baseline, not the final architecture.

## Required reading before edits

Read, in this order:

1. `START_HERE.md`
2. `docs/AGENT_HANDOFF.md`
3. `docs/SIMULATION_FLOW.md`
4. `docs/OBJECTIVES_AND_EVALUATION.md`
5. `docs/REQUIREMENTS_MATRIX.md`
6. `docs/EXPERIMENT_PROTOCOL.md`
7. `docs/ROADMAP.md`
8. `docs/VERIFICATION.md`
9. `TASKS.md`

For command behavior, read `docs/COMMAND_REFERENCE.md`. For research context,
read `docs/LITERATURE_GUIDE.md` and `references/Relevant-papers.txt`.

## Binding invariants

1. **No truth leakage.** `prepare`, `baseline`, `train`, and `export` must read
   only `observed/`. Only evaluator code may read `truth/`.
2. **One canonical path contract.** Public commands accept dataset roots through
   `--run-dir` and modeling roots through `--experiment-dir`. Centralize internal
   filenames in `layout.py` and `contract.py`.
3. **Explicit schemas.** Tensor column order comes from
   `prepared_metadata.json`, never JSON dictionary ordering. Store field order
   in checkpoints and exports.
4. **Leakage-safe time.** Fit vocabularies and normalization only on training
   events. A target event sees only earlier events for that user.
5. **Protected counterfactuals.** Latent preferences, true utilities, true
   episodes, chosen flags, and noiseless coordinates remain evaluator-only.
6. **No aggregate winner.** Report requirement axes separately. High temporal
   stability can indicate collapse; interpret it with separation, retrieval,
   effective rank, and task information.
7. **Configuration before hard-coding.** Put data-generating assumptions,
   ablations, loss weights, and model variants in versioned YAML where practical.
8. **Reproducibility.** Use fixed seeds, resolved configuration snapshots,
   source hashes, and immutable run directories for comparisons.
9. **Apple MPS compatibility.** Do not reintroduce `pack_padded_sequence` or
   integer `gather` for final valid GRU states without an MPS regression test.
   Sequence lengths remain CPU control metadata.
10. **Preserve user work.** Do not delete or overwrite run/experiment artifacts
    unless the requested command explicitly uses a validated `--overwrite`
    target. Never edit generated artifacts as source code.

## Development order

Follow this order unless evidence justifies a documented change:

1. Reproduce the current tests and a small baseline/learned comparison.
2. Add missing evaluators and exports needed to measure R1--R8 and R10--R13.
3. Extend matched-seed simulator interventions needed by those evaluators.
4. Add the factorized persistent/context model behind the existing CLI.
5. Add a routine component only once tests can distinguish routine from both
   persistent traits and temporary episodes.
6. Extend the observable recommendation contract.
7. Add candidate-aware ranking and Tokyo-to-Hakone evaluation.

Evaluation capability should normally precede the model intended to improve it.

## Change protocol

Before coding:

- State the requirement IDs affected.
- Record the baseline artifact or explain why none exists.
- Identify whether the change touches the simulator, observed contract, model,
  evaluator, or several layers.
- If the observed contract changes, bump its version and add migration/contract
  tests before consuming it in the model.

While coding:

- Make the smallest coherent change.
- Add unit tests for local logic and an integration test for cross-stage
  contracts.
- Keep simulator truth out of model constructors, datasets, and training APIs.
- Prefer typed/dataclass boundaries over positional tuples for new multi-output
  embeddings.
- Validate IDs, shapes, field order, finiteness, timestamps, and source hashes
  before accelerator execution.

Before handing off:

```bash
uv sync --locked --extra dev
uv run pytest
uv run geoembed --version
```

For simulator changes, also run a fixed-seed small simulation and `validate`.
For model/evaluator changes, run baseline and learned exports on the same run and
execute `compare`. Report exact commands, artifact paths, seeds, and deltas.

## Evidence rules

- Treat simulator constants as hypotheses, not facts about Tokyo or Kanto.
- Do not claim disentanglement from next-event accuracy or cosine stability.
- Do not claim causal invariance from observational probes.
- A simulator improvement must show both integrity and behavioral diagnostics.
- A model improvement must beat relevant baselines on held-out users and the
  targeted requirement without unacceptable regression on other axes.
- Results from different source hashes, cutoffs, users, or candidate sets are not
  a fair comparison.
- Mark unmeasurable requirements as such; do not substitute a convenient proxy
  without naming its limitations.

## Definition of done for a task

A task is complete only when:

- the affected requirement has an executable metric or explicit contract;
- tests cover the failure mode and information boundary;
- documentation and configuration are updated;
- a reproducible command produces the new artifact;
- the result is compared against the appropriate baseline;
- limitations and remaining blockers are recorded in `TASKS.md` or the relevant
  design document.

## Current known hazards

- Earlier releases allowed JSON key sorting to scramble categorical tensor
  columns. Preserve the explicit categorical-field order contract.
- Variable-length packed GRUs and gather-based final-state selection caused
  Apple Metal failures. Preserve the padded GRU and floating mask selector.
- Users may have no history at an early cutoff; exports may omit that
  user/cutoff, but training windows must satisfy `min_history_events`.
- The single-vector consistency objective may reward a collapsed representation.
- Geohash-7 has a large, sparse vocabulary and can reward memorization.
- `pipeline` starts from simulation; it is not a resume command and does not run
  baseline-versus-learned comparison.

## Scope discipline

Do not jump directly to a large transformer, graph model, LLM component, or
real-data integration. Introduce a method only when it addresses a named
requirement, is testable under the current contract, and has a simpler baseline.
The literature is a source of components and experimental ideas, not a mandate
to reproduce every architecture.
