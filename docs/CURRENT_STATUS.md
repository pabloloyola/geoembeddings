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
