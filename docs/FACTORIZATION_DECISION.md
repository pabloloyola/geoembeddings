# T2.7 persistent/context factorization decision protocol

**Requirements:** R1, R4, R5, R6, and R7. This change affects the observed-only
model/training/export path and the protected evaluator; it does not change the
observed dataset contract or expose simulator truth to training.

## Matched experiment matrix

Run `prepare` once, then use immutable experiment roots with
`factorized_pc.yaml`, `capacity_matched_single.yaml`, `persistent_only.yaml`,
`context_only.yaml`, `factorized_no_persistent_loss.yaml`, and
`factorized_no_context_loss.yaml`. Run cutoff and dense export, base evaluation,
episode evaluation, robustness, matched temporary-change evaluation, and
`compare` for every control. The training report and checkpoint record resolved
configuration, seed, trainable/total parameter count, preparation hash, observed
source hashes, cutoffs, and checkpoint lineage.

## Per-axis decision (no aggregate winner)

| Output | Required intended evidence | Mandatory collapse/failure evidence |
|---|---|---|
| `persistent` | persistent-trait/preference information and matched-intervention invariance | same/different-user separation, temporal retrieval, centered effective rank, task information |
| `context` | episode-boundary response and temporary-change adaptation/recovery | same/different-user separation, temporal retrieval, centered effective rank, persistent leakage |
| `combined` | frozen downstream task information | same/different-user separation, temporal retrieval, centered effective rank, branch-ablation deltas |

The decision is **advance** only when the factorized model beats the
capacity-matched single-vector and relevant branch/loss-routing controls on each
intended axis without an unacceptable regression on the mandatory diagnostics.
Otherwise record **do not advance** with per-axis deltas and coverage. There is
no aggregate score and no routine branch may be added following a failed or
unmeasurable decision.

## Current decision (2026-08-12 replacement identity)

**Do not advance to the routine branch.** A new immutable 50-user, 14-day
replacement run (`t2.7-factorization-20260812-s20260812-u50-d14`, simulation
seed 20260812) prepared the observed data once and trained all six matrix
variants with training seed 20260806. The comparison authenticated source hash,
preparation definition, three cutoffs, export keys, and the 49-user mask.

The factorized model had 531,779 trainable parameters versus 530,899 for the
capacity-matched single control (0.165% difference). Its persistent branch mean
held-out probe R2 was -0.651 versus -0.538 for the capacity control (delta
-0.113), while centered effective rank was 5.37 versus 9.92 and
same/different-user separation was 0.437 versus 0.707. Its combined mean probe
R2 was -0.957 versus -0.538 (delta -0.419), despite temporal retrieval of 0.929
versus 0.908. The context branch retained non-collapsed signal (effective rank
6.82, retrieval 0.684), but this cannot rescue failed persistent and combined
gates and is not evidence of semantic disentanglement.

Cutoff/dense exports and base, episode, robustness, and temporal-routine reports
exist for every control under `experiments/t2.7-factorization-20260812/`.
Matched paired/change evaluation was not needed to overturn the decision after
mandatory task-information and collapse gates failed; consequently no causal
invariance claim is made. This pilot has only seven held-out probe users and
sparse travel observations, so its estimates are coverage-qualified. Routine
work remains closed.
