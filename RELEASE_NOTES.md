# v0.5.0 release notes

The repository has continued beyond the original agent-handoff snapshot. New
simulations use `geoembeddings-dataset/2.0`; event-only 1.0 runs remain
explicitly compatible with the legacy modeling path.

Added:

- root `AGENTS.md` with binding invariants and development protocol;
- `START_HERE.md` for remote setup and reproduction;
- complete command/input/output/storage reference;
- architecture and evidence handoff;
- R1--R13 measurable requirement matrix;
- fair experiment and rerun protocol;
- phased simulator/model/recommendation roadmap;
- executable task backlog;
- literature-to-project guide based on the supplied papers;
- release verification evidence;
- reproducible `uv.lock`.

Verified:

- 13 tests passed;
- package reports version 0.5.0;
- locked dependency resolution passes;
- 50-user, 7-day learned pipeline completed on CPU;
- baseline export, protected evaluation, and comparison completed on the same
  smoke dataset.

Known scientific limitations remain explicit in `docs/CURRENT_STATUS.md` and
`docs/REQUIREMENTS_MATRIX.md`.

Implemented since the initial handoff:

- immutable paired exposure, opportunity, observation, schedule-shift,
  temporary-trip, and sustained-preference simulation with integrity-gated
  paired/change evaluation;
- accepted, indexed T0.4 replacement reference diagnostics and their no-aggregate
  decision record;
- typed component checkpoint/export schemas, the `factorized_pc` model, a
  capacity-matched control, and branch/loss ablation configurations;
- dataset contract 2.0 public POI, request, impression, and interaction tables,
  including synthetic Hakone request-time attributes and explicit 1.0 reading;
- seeded reliability/repeatability reports and observed-only offline benchmarks.

The current gate is T2.7: factorization remains unmeasurable until matched
immutable control/factorized artifacts are compared. T3.4 observable naive
rankers are the next independent implementation option.
