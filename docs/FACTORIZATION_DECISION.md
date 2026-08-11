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

## Current decision

**Unmeasurable; do not advance to the routine branch.** No new matched experiment
artifacts were generated in this source change, and the historical T0.2 artifact
bytes are unavailable. This is an implementation and executable-contract
change, not scientific evidence of factorization.
