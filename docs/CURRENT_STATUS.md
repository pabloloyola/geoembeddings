# Current status

## Release

- Package/repository version: `0.5.0`
- Behavioral lineage: tested v0.4.x simulator and modeling code
- v0.5.0 changes: agent handoff, research protocol, roadmap, command reference,
  task backlog, literature guide, and locked dependencies
- Dataset contract: `geoembeddings-dataset/1.0`

## Working capabilities

- Semi-synthetic Kanto simulation across five controlled scenarios
- Structural and deep simulator validation
- Observed/truth information boundary
- Leakage-safe global temporal preparation
- Statistical history-vector baseline
- MPS-safe single-vector GRU training
- Learned embedding export at three cutoffs
- Protected latent probes and next-event evaluation
- Fair frozen baseline-versus-learned comparison

Release verification is recorded in `docs/VERIFICATION.md`: 13 tests passed and
a 50-user, 7-day learned pipeline plus baseline comparison completed end to end.

## Pending scientific capabilities

- Dense and episode-aligned embedding exports
- Direct persistent/context and routine/context metrics
- Controlled corruption and missing-service robustness
- Explicit geographic holdout experiments
- Matched counterfactual exposure/opportunity invariance
- Factorized encoder
- Recommendation request/impression/interaction data contract
- Candidate-aware ranking and Tokyo-to-Hakone evaluation
- Representation uncertainty, nonstationarity, privacy, and efficiency audits

## Immediate instruction

Run the comparison on the same 500-user dataset already used for learned
training. Preserve `embedding_comparison.json` and `.md` as the pre-factorization
reference. Then begin Phase 1 of `docs/ROADMAP.md`.

## T1.2 episode response (R1, R4)

The protected evaluator joins observed-event dense embeddings to truth episodes in memory and writes `episode_response.json` or `baseline_episode_response.json`. Statistical and learned dense exports remain observed-only. Coverage, response curves, coherence, boundary change, drift/recovery, a held-out-user intent probe, different-user separation, and effective rank are reported. These are single-vector diagnostics, not evidence of disentanglement.

### Matched smoke evidence (seed 20260811)

A 20-user, 3-day matched smoke run (`/tmp/t12-run`, `/tmp/t12-exp`) produced 186 unique dense rows for each representation. Learned-minus-baseline deltas were -0.02373 within-episode consecutive cosine, +0.01754 boundary-change magnitude, and +0.07824 post-episode recovery cosine. Coverage was 18/20 users, 53/60 episodes, and 6/10 bins. This tiny run is executable evidence only: deep simulation validation failed at this scale, two users had no observed dense history, and no scientific model-improvement claim is warranted.
