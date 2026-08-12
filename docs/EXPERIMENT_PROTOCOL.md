# Experiment protocol

## Purpose

This protocol prevents changes in data, splits, probes, or candidate sets from
being mistaken for model improvements.

## Experimental unit

A dataset run is immutable and identified by:

- resolved simulation configuration;
- dataset contract version;
- simulator version;
- random seed;
- hashes of observed source files.

An experiment is one representation/modeling attempt on exactly one dataset run
and is identified by:

- resolved embedding configuration;
- preparation metadata and cutoffs;
- code version/commit when available;
- model seed;
- checkpoint selection rule.

Use descriptive names:

```text
runs/kanto_mixed_seed20260803_u500_d14/
experiments/sv_gru_seed20260806_on_kanto_mixed_seed20260803/
```

Do not reuse a directory for materially different settings.

## Reproduction sequence

```bash
uv sync --locked --extra dev
uv run pytest

uv run geoembed simulate \
  --config configs/simulation/kanto_v1.yaml \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14

uv run geoembed validate \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14

uv run geoembed prepare \
  --config configs/embedding/single_vector.yaml \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed train \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed export \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed baseline \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed evaluate --kind learned \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed evaluate --kind baseline \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference

uv run geoembed compare \
  --run-dir runs/kanto_mixed_seed20260803_u500_d14 \
  --experiment-dir experiments/sv_reference
```

## Comparison levels

### Level A: plumbing smoke test

- 50 users, 7 days, one seed.
- Purpose: schemas, paths, MPS/CPU execution, artifacts, and tests.
- Never use it to select a scientific model.

### Level B: development comparison

- 500 users, 14 days, fixed reference seed.
- Compare statistical baseline, current GRU, and one change at a time.
- Use identical preparation and frozen probe splits.

### Level C: robustness and counterfactual suite

- At least three generator/model seeds.
- Matched latent users and world across scenario interventions.
- Report mean, standard deviation, and per-seed values.
- Evaluate all requirements targeted by the change plus regression axes.

### Level D: recommendation suite

- Freeze request/candidate records for model comparisons.
- Separate seen-region, held-out-region, seen-POI, unseen-POI, and first-arrival
  slices.
- Compare frozen encoder plus small ranker and end-to-end fine-tuning separately.

## Matched-seed simulator interventions

Scenario comparison is meaningful only when unchanged latent objects retain
stable identities. When extending the simulator, use independent named random
streams for at least:

- world/POI creation;
- user latent creation;
- episode generation;
- choice noise;
- observation/dropout/GPS noise.

Changing only exposure or observation parameters should not silently regenerate
user preferences, POI qualities, or episodes. Record stream seeds in the
manifest. Until this is implemented, describe current matched-seed scenario
comparisons as approximate.

## Model selection

- Select checkpoints using validation data only.
- Do not tune on protected test probes.
- A next-event validation loss can select within one model family, but the final
  research decision must use the held-out requirement suite.
- When objective weights change, rerun training and all downstream exports and
  evaluations. Reuse the same `prepared/` artifacts only when input fields and
  split/preprocessing settings are unchanged.

### Representation-selection policy

Selection is a recorded property of a particular export lineage, not an
informal synonym for whichever representation an audit happens to load. Every
representation used by a downstream audit must have exactly one of these
roles:

- **`diagnostic_control`**: an export that may be used in a clearly labeled
  comparative or failure-analysis audit even though it did not pass, or was
  never submitted to, the applicable scientific advancement gates. Results for
  this role characterize the export; they cannot support selection-dependent
  conclusions, deployment claims, or promotion of its component names to
  established semantics.
- **`selected_candidate`**: an export that passed all applicable matched
  scientific gates for the conclusion being attempted, including the required
  task-information, collapse, separation/retrieval, and targeted requirement
  checks against the specified controls. Passing checkpoint selection on
  validation loss, producing finite vectors, or winning one downstream metric
  is not sufficient. The decision record and matched evidence index must name
  the gates and authenticate the export lineage.

The completed T2.7 decision is **do not advance**. Consequently, no current
export qualifies as `selected_candidate`. The indexed T0.4 statistical history
baseline exports, the T2.7 `capacity_matched_single` exports, and the T2.7
`factorized_pc`, `persistent_only`, `context_only`,
`factorized_no_persistent_loss`, and `factorized_no_context_loss` exports may
qualify only as `diagnostic_control` for explicitly labeled comparative audits.
This also applies to the later T3.5 `factorized_pc` combined export: its ranking
result did not reverse or rerun the T2.7 representation gate. Qualification is
conditional on authenticating the exact artifact against its immutable index;
an unindexed or mismatched copy does not inherit a role from its branch name.

Audit artifacts must record, for every representation input:

- the exact model variant and exported component identifier;
- the checkpoint SHA-256 (or an explicit `not_applicable` value and reason for
  a non-checkpoint statistical baseline), together with the export hash;
- preparation identity, including the preparation-definition/metadata hash;
- observed source-file hashes and any evaluator-only source hashes opened by a
  protected audit;
- the complete cutoff set;
- parameter count, including an explicit zero/non-parametric declaration for
  the statistical baseline; and
- `selection_role`, using one of the two values above, plus the evidence-index
  and decision-record identities that justify it.

Reports must treat `factorized_pc` and component names such as `persistent`,
`context`, `p`, or `c` as configuration/branch identifiers with intended or
hypothesized meanings. A report must state the failed gate and diagnostic role
where those names first appear; it must not silently describe them as
established persistent-preference or current-context semantics.

When no `selected_candidate` exists, any conclusion whose prerequisite is a
selected representation is **unavailable**; absence of a candidate must not be
resolved by silently selecting the least-bad diagnostic. Work may still produce
explicitly labeled comparative audits of the statistical baseline, the
capacity-matched single vector, and factorized diagnostic variants, provided
results remain per-representation and do not make the unavailable
selection-dependent conclusion. T4.1, T4.2, and T4.3 apply this policy.

## Required baselines

Keep at least:

1. popularity/majority targets for predictive tasks;
2. the statistical history vector;
3. the current single-vector GRU;
4. a last-N-events or recent-only representation for context tasks;
5. persistent-only and context-only ablations for factorized models;
6. nearest-POI, popularity, and category-preference rankers for recommendation.

## Ablation discipline

Change one conceptual factor at a time. For a factorized model, minimum
ablations are:

- no persistent loss;
- no context loss;
- no routine branch;
- no cross-window consistency;
- no event dropout;
- no fine geohash target;
- shared versus separate projection heads;
- with and without object ID;
- full history versus recent window.

## Report contents

Every result report should include:

- command and date;
- code version/commit;
- run and experiment directories;
- simulator and model seeds;
- source hashes and temporal cutoffs;
- device and runtime versions;
- train/validation/test counts and known-label coverage;
- full configuration or path to the resolved copy;
- metric values by requirement and baseline deltas;
- runtime and peak memory when available;
- interpretation, limitations, and next decision.

## Rerun dependency table

| Change | Simulate | Validate | Prepare | Train | Export | Evaluate/compare |
|---|---:|---:|---:|---:|---:|---:|
| Simulator YAML or generator logic | Yes | Yes | Yes | Yes | Yes | Yes |
| Observed schema/contract | Yes | Yes | Yes | Yes | Yes | Yes |
| Split, fields, vocabulary, normalization | No | No | Yes | Yes | Yes | Yes |
| Model architecture or objective | No | No | Usually no | Yes | Yes | Yes |
| Training optimizer/epochs/seed | No | No | No | Yes | Yes | Yes |
| Export format/cutoffs | No | No | No | No if checkpoint compatible | Yes | Yes |
| Evaluator/probe only | No | No | No | No | No unless new export needed | Yes |
| Documentation only | No | No | No | No | No | No |

## Failure handling

- If MPS fails, first run the same batch/test on CPU and inspect the first
  asynchronous operation, not merely the synchronization line in the traceback.
- Validate categorical IDs and target ranges on CPU before accelerator work.
- If a comparison rejects source hashes or cutoffs, do not bypass the guard;
  reproduce both representations under one preparation contract.
- If a requirement cannot be measured from existing exports, extend exports or
  mark it unavailable. Do not infer it from unrelated metrics.
