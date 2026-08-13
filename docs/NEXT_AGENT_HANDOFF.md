# Next-agent handoff

This document is the short operational handoff for the next Codex agent. It is
an index and checklist, not a replacement for the repository's authoritative
contracts. If this summary conflicts with `AGENTS.md`, `TASKS.md`, or a design
document, follow the more specific authoritative document and update this file.

## Mission and scientific boundary

The project asks whether a cross-service user representation can recover
persistent preference while adapting to recurring routine and temporary episode
state, without confusing those signals with opportunity, exposure, missingness,
or a single trip. The target decomposition is

\[
z_{u,t}=(p_u,r_{u,t},c_{u,t}),
\]

where `p` is persistent preference, `r` is recurring routine, and `c` is current
episode or intent. The single-vector GRU is a diagnostic baseline, not the
intended final architecture.

Do not claim disentanglement from next-event accuracy or cosine stability. Do
not claim causal invariance from observational probes, and do not present
synthetic Kanto/Hakone results as evidence about real users or deployments.

## Release readiness

- **Assessment source:** commit
  `d2c3f6652f84d6320ba8b4fb2564ad2f190885d1`; do not treat historical results
  as verification of this revision.
- **Implementation:** T4.1a diagnostic-control uncertainty calibration and
  T4.3a diagnostic-control privacy auditing are implemented, while full
  selection-dependent T4.1 and T4.3 remain unavailable.
- **Full suite:** the latest authoritative executed-suite record is `218
  passed in 22.00s` after repairs at
  `f8a1f73d8833174e170266ab7f5f9f5d8c723248`. It was not run after a successful
  locked dev sync and is neither locked nor current-assessment evidence.
- **Stable gate and CI:** every stable-release checklist item remains
  unchecked. CI is configured for the bounded CPU smoke and Python 3.11--3.14
  CPU matrix with an immutable verification-record job, but no successful
  current-assessment workflow record exists; the release-candidate matrix has
  not passed.
- **Bounded CLI evidence:**
  `docs/artifacts/release-cli-integration-smoke-20260813.json` records the
  latest indexed successful 50-user/seven-day CPU contract smoke at source
  `4dcba63462570fda439e24e4e4812a85c6854753`. It used `UV_NO_SYNC=1` after a
  package-index failure and is historical bounded evidence, not a clean locked
  release baseline for this revision.
- **Blockers and scientific decision:** a clean locked CPU sync/full suite,
  the remaining package/CLI and bounded-workflow gates, a passing CI matrix and
  immutable workflow record, and all remaining checklist evidence block a
  stable release. No `selected_candidate` exists. Preserve T2.7's **do not
  advance** decision and do not open a routine branch.
- **Recommended next action:** obtain a clean release verification baseline for
  the assessment revision or a newly identified immutable successor.

## Mandatory reading before edits

Read these files in order:

1. `START_HERE.md`
2. `docs/AGENT_HANDOFF.md`
3. `docs/SIMULATION_FLOW.md`
4. `docs/OBJECTIVES_AND_EVALUATION.md`
5. `docs/REQUIREMENTS_MATRIX.md`
6. `docs/EXPERIMENT_PROTOCOL.md`
7. `docs/ROADMAP.md`
8. `docs/VERIFICATION.md`
9. `TASKS.md`

Also read `docs/COMMAND_REFERENCE.md` for command behavior,
`docs/EXTERNAL_VALIDITY.md` before interpreting results, and the task-specific
design document and tests before changing code. Research context lives in
`docs/LITERATURE_GUIDE.md` and `references/Relevant-papers.txt`.

## State at this handoff

- Dataset contract `geoembeddings-dataset/2.0`, evaluator foundations, paired
  interventions, persistent/context components, and T3.4--T3.7 recommendation
  work are implemented.
- T2.7 ended with a **do not advance** decision: persistent and combined
  task-information/collapse gates failed. Preserve that result and do not open
  a routine branch.
- T4.2 adaptation/forgetting, T4.3a diagnostic-control privacy auditing, T4.4
  online benchmarking, and T4.5 external-validity boundaries are complete.
- No representation has the immutable `selected_candidate` role. Current
  representation comparisons are `diagnostic_control` evidence only.
- The open top-level tasks are T4.1 (R10, uncertainty calibration) and T4.3
  (R12, selection-dependent privacy conclusion). T4.1 and T4.3 cannot be
  completed without a selected candidate, although T4.1a remains open and the
  explicitly scoped T4.3a diagnostic-control command is complete. Reconfirm all
  statuses in `TASKS.md` before starting because this paragraph will age.

Run T4.3a with `uv run geoembed audit-privacy --run-dir RUN_DIR
--experiment-dir NAME=EXPERIMENT_DIR --evidence-dir EVIDENCE_DIR
--utility-report-dir UTILITY_REPORT_DIR --config
configs/privacy/diagnostic_v1.yaml --output-dir AUDIT_OUTPUT_DIR`. It writes
immutable `audits/privacy.{json,md}` under schema
`geoembeddings-privacy-audit/1.0`, authenticates inputs before protected-label
access, and reports attacks, utility axes, uncertainty, coverage, exclusions,
and unavailable reasons separately. Its scope is diagnostic-control evidence:
it neither selects a representation nor certifies privacy, and AUC near 0.5 is
not proof of safety. Preserve the T2.7 **do not advance** decision.

## How to select and record work

Select one coherent open task from `TASKS.md`. Before coding, record in the work
note or PR description:

1. the task and requirement IDs;
2. the baseline artifact, or why none exists;
3. the affected layers (simulator, observed contract, model, evaluator, or
   documentation); and
4. whether the observed contract changes.

For this documentation-only handoff, the affected requirements are R1--R13 as
process and claim boundaries; no scientific baseline artifact is applicable.

Evaluation should normally precede the model intended to improve it. Avoid a
large transformer, graph model, LLM component, or real-data integration unless
it addresses a named measurable requirement and has a simpler baseline.

## Non-negotiable implementation contracts

### Information boundary

- `prepare`, `baseline`, `train`, and `export` read only `observed/`.
- Only evaluators may read `truth/`.
- Latent preferences, true utilities, true episodes, protected chosen flags,
  noiseless coordinates, and counterfactual probabilities must never enter
  model constructors, datasets, features, checkpoints, or training APIs.
- Add leakage tests whenever work approaches this boundary.

### Paths, schemas, and artifacts

- Public commands take dataset roots through `--run-dir` and modeling roots
  through `--experiment-dir`; never pass `observed/` or `truth/` as a run root.
- Centralize filenames in `layout.py` and `contract.py`.
- Tensor field order comes from `prepared_metadata.json`, never JSON key order.
  Preserve field order in checkpoints and exports.
- Validate IDs, shapes, field order, finiteness, timestamps, source hashes, and
  preparation identity before accelerator execution.
- Preserve user artifacts. Do not overwrite immutable run or experiment output
  except through an explicit, validated `--overwrite` operation.

### Time safety and reproducibility

- Fit vocabularies and normalization only on training events.
- A target event sees only earlier events for that user.
- Keep fixed seeds, resolved configuration snapshots, source hashes, and
  immutable roots for comparisons.
- Match source hashes, users, cutoffs, seeds, candidate sets, and intervention
  identity before reporting deltas.

### Apple MPS

Do not reintroduce `pack_padded_sequence` or integer `gather` for final valid
GRU states without an MPS regression test. Sequence lengths remain CPU control
metadata; preserve padded GRUs and the floating-mask selector.

## Evidence and evaluation rules

- Report requirement axes separately; there is no aggregate winner.
- Interpret high temporal stability with separation, retrieval, effective rank,
  and task information because stability alone can indicate collapse.
- A simulator change needs integrity and behavioral diagnostics.
- A model change must be tested on held-out users against relevant baselines and
  must not hide regression on other axes.
- Mark an unmeasurable requirement as unmeasurable. Do not silently replace it
  with an easier proxy.
- Treat simulator constants as hypotheses rather than facts about Tokyo or
  Kanto.

## Recommended execution loop

1. Confirm `git status`, the current `TASKS.md` status, and applicable
   `AGENTS.md` files.
2. Read the selected task, design contract, implementation, tests, and existing
   verification evidence.
3. Reproduce the smallest relevant baseline and record its immutable paths and
   hashes.
4. Make the smallest coherent change. Prefer typed/dataclass boundaries for new
   multi-output embeddings and versioned YAML over hard-coded assumptions.
5. Add unit tests for local logic and an integration test for cross-stage or
   information-boundary behavior.
6. Run targeted checks, then the full checks below.
7. For simulator work, run a fixed-seed small simulation and `validate`. For
   model/evaluator work, produce baseline and learned exports from the same run
   and execute `compare`.
8. Update configuration, command docs, verification evidence, and `TASKS.md` as
   required. Record exact commands, seeds, artifact paths, hashes, per-axis
   deltas, limitations, and blockers.
9. Commit the coherent change and leave the tree in an understandable state.

## Required checks before handoff

```bash
uv sync --locked --extra dev
uv run pytest
uv run geoembed --version
```

Use `docs/LOCAL_EXPLORATION.md` for a fresh small workflow and
`docs/COMMAND_REFERENCE.md` for exact CLI contracts. A task is complete only
when it has an executable metric or explicit contract, failure-mode and leakage
coverage, updated docs/configuration, a reproducible artifact, an appropriate
baseline comparison, and documented limitations.

## What the next handoff must contain

- task and requirement IDs;
- baseline and comparison artifact locations;
- files changed and contract versions affected;
- exact commands, seeds, resolved configuration, and results;
- source and artifact hashes plus matched per-axis deltas;
- the supported conclusion and prohibited interpretations;
- remaining blockers and the recommended next task.
