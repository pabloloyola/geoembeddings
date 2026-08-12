# External validity and evidence boundaries

This is the T4.5 claim-boundary contract for R1--R13. It inventories what the
semi-synthetic harness can test, what indexed evidence is accepted, and what
would be needed to change a requirement status. Passing the static check means
that the references and claim metadata are internally auditable; it does **not**
mean that a model succeeded scientifically or that any result transfers to
people, services, or places outside the simulator.

## Simulator assumptions

- User traits, episode types, service adoption, exposure, opportunity,
  observation, utility, and change mechanisms are authored hypotheses. Their
  controlled truth is useful for falsification and recovery tests, but does not
  estimate a population effect.
- Catchments are overlapping synthetic distributions around named regional
  anchors. Movement is stop-based and straight-line; there is no transport
  network, congestion, capacity, or empirically fitted travel process.
- Matched interventions identify behavior under the simulator's structural
  equations only. Pair integrity authenticates the experiment; it does not
  authenticate those equations as a model of real behavior.
- Observation errors and missingness are configured processes rather than
  estimates from devices or production logging systems.

## Accepted indexed evidence

The historical T0.2 index is retained as an evidence-loss record. Its referenced
run and experiment bytes are unavailable, so its recorded digests cannot be
re-authenticated and it supports no result claim. The T0.4 replacement is a
separately named, locally indexed lineage accepted only for coverage-qualified
synthetic R2/R3 diagnostics. It is neither a recovery of T0.2 nor an aggregate
model verdict. The machine-readable registry below records the only
quantitative statement made by this document, together with its artifact,
cohort, seeds, source and preparation identities, and scientific scope.

## Unsupported real-world claims

This repository currently supplies no evidence that the embeddings are valid
for real residents, visitors, protected groups, devices, platforms, commercial
catalogs, or transport conditions. Simulator ranking metrics cannot establish
business lift, welfare, causal exposure correction, geographic
representativeness, calibrated uncertainty, privacy safety, or operational
service levels. Component names do not establish disentanglement. Task
completion, schema validity, reproducibility, and passing integrity checks are
engineering evidence, not scientific success.

## Licensing and data limitations

No licensed, de-identified production event corpus or independently sampled
real-world evaluation cohort is registered here. Consequently there is no
auditable consent basis, retention policy, jurisdictional assessment,
demographic coverage analysis, production logging-policy description, or
license allowing a real-data validation claim. Before accepting external data,
the project would need documented lawful access, permitted purposes,
de-identification and re-identification review, data minimization, retention and
deletion rules, geographic and temporal coverage, and publication constraints.
Public place facts would also require provenance, a compatible license, and a
dated snapshot rather than silent substitution into synthetic truth.

## Synthetic Hakone and Kanto limitations

Hakone POIs, attributes, opening and availability states, popularity, travel
times, candidate exposure, requests, interactions, and utilities are synthetic.
Regional names provide an experimental scenario, not a calibrated digital twin
of Hakone or Kanto. The harness omits rail and road topology, seasonality,
weather, crowding, closures, accessibility, multilingual needs, group travel,
merchant incentives, and platform feedback loops. Seen/unseen synthetic slices
therefore test contract behavior under designed shifts, not transfer to an
actual destination or population.

## Evidence required to change requirement status

Each status change must preserve per-axis reporting, matched identities where
applicable, uncertainty and missingness, and negative results. Evidence must be
ethically and legally usable, independently held out, time-bounded, and indexed
with immutable source/preparation identities. The registry specifies the
minimum requirement-specific addition; none may be replaced by task completion
or a simulator-only proxy.

| Requirement | Evidence needed before changing external-validity status |
|---|---|
| R1 | Held-out longitudinal real histories with defensible temporary-context and sustained-preference labels, plus collapse and leakage controls. |
| R2 | Licensed real coordinates and place hierarchy with device-error, boundary, coverage, and unseen-area analyses. |
| R3 | Real timestamps and durations spanning recurring schedules, holidays, and shifts, evaluated prospectively. |
| R4 | Independently defined episode boundaries and intents with within/between and boundary-response evaluation. |
| R5 | A justified exposure/opportunity identification design, overlap diagnostics, sensitivity analysis, and preferably randomized or natural-experiment evidence. |
| R6 | Cross-service held-out outcomes under documented identity linkage, consent, service missingness, and single-service controls. |
| R7 | Empirical device and logging missingness distributions, worst-group analyses, and prospective degradation checks. |
| R8 | Frozen prospective evaluation in genuinely unseen places, POIs, and later periods with explicit cold-start coverage. |
| R9 | Frozen candidate sets and prospective or randomized real recommendation outcomes, including utility, harms, coverage, and control rankers. |
| R10 | Held-out-user calibration and coverage-risk evidence under real shifts, with a prespecified confidence target. |
| R11 | Prospective longitudinal change evidence with censoring, temporary/sustained definitions, adaptation, recovery, and regret. |
| R12 | A deployment-specific threat model, held-out attacks, subgroup privacy/utility curves, and governance review; simulator attacks cannot certify safety. |
| R13 | Reproducible training and online-update benchmarks on the target stack and workload, including tail latency, memory, throughput, and failure behavior. |

## Static evidence registry

The registry is JSON so `scripts/check_evidence_links.py` can validate it
without importing or executing simulator/model code.

```evidence-registry
{
  "schema_version": "geoembeddings-external-validity-registry/1.0",
  "completion_is_scientific_success": false,
  "artifacts": [
    {
      "path": "docs/artifacts/t0.2-reference500.json",
      "sha256": "4aae137428df255bfabce33f4f9114da60faa4706bdded3166b5f54c76b53ac7",
      "task_id": "T0.2",
      "availability": "local_index_only",
      "accepted": false
    },
    {
      "path": "runs/reference500",
      "task_id": "T0.2",
      "availability": "unavailable_historical",
      "accepted": false
    },
    {
      "path": "docs/artifacts/t0.4-r2-r3-reference-20260811.json",
      "sha256": "b682703c58fff4c692aeaf2ee2c186fb8379f1b3845073b746919935c9326683",
      "task_id": "T0.4",
      "availability": "local_index",
      "accepted": true
    }
  ],
  "claims": [
    {
      "artifact": "docs/artifacts/t0.4-r2-r3-reference-20260811.json",
      "task_id": "T0.4",
      "requirement_ids": ["R2", "R3"],
      "cohort_size": 500,
      "seed": {"simulation": 20260811, "training": 20260806, "evaluation": 20260811},
      "source_identity": "git:a775533c99d041d3f1cf500f0b8204d6ecb82e42; manifest-sha256:5e71d69e6cc0f9825ecd2f0ab03b9e3ec20ef3c6f0bf1389aec79d2d1de419d3",
      "preparation_identity": "sha256:8d3556e59300a62a55df59cd47ed3767605424f336d819ce3c2963f178273982; geoembeddings-dataset/1.0",
      "scientific_scope": "coverage-qualified next-event and synthetic spatial/temporal diagnostics only; no aggregate winner, causal inference, factorization, or external validity",
      "evidence_kind": "simulator_only",
      "statement": "The accepted replacement lineage contains 500 simulated users under its recorded seeds and identities."
    }
  ],
  "requirements": [
    {"id": "R1", "status_change_evidence": "held-out real temporary and sustained preference evidence"},
    {"id": "R2", "status_change_evidence": "licensed real multi-scale geographic evidence"},
    {"id": "R3", "status_change_evidence": "prospective real temporal and routine evidence"},
    {"id": "R4", "status_change_evidence": "independent real episode labels"},
    {"id": "R5", "status_change_evidence": "identified real exposure and opportunity study"},
    {"id": "R6", "status_change_evidence": "consented cross-service transfer study"},
    {"id": "R7", "status_change_evidence": "empirical device and logging-noise study"},
    {"id": "R8", "status_change_evidence": "prospective unseen-place and unseen-POI study"},
    {"id": "R9", "status_change_evidence": "prospective or randomized real recommendation evaluation"},
    {"id": "R10", "status_change_evidence": "held-out real uncertainty calibration"},
    {"id": "R11", "status_change_evidence": "prospective real longitudinal change study"},
    {"id": "R12", "status_change_evidence": "deployment threat model and real held-out privacy audit"},
    {"id": "R13", "status_change_evidence": "target-stack online and training benchmark"}
  ]
}
```

Run the non-executing audit with:

```bash
uv run python scripts/check_evidence_links.py docs/EXTERNAL_VALIDITY.md
```
