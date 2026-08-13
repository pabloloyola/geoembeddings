# Privacy audit threat model

## Status, requirement, and evidence boundary

This is the pre-implementation design contract for **T4.3 / R12**. It fixes
the threat model, information boundary, split protocol, controls, and future
`audit-privacy` report before an attack harness is implemented. The change is
documentation-only: it changes neither the simulator nor the observed contract,
model, evaluator, or exports. No privacy baseline artifact currently exists.
The future audit must use matched immutable statistical, capacity-matched
single-vector, and factorized exports plus their existing utility reports.

The negative T2.7 decision means that no current representation has the
`selected_candidate` role. Current work may therefore produce only explicitly
labeled **diagnostic-control** results. No selection-dependent R12 privacy
conclusion is available until an authenticated export lineage has the immutable
`selected_candidate` role under the representation-selection policy. The audit
must not select the least-bad control or infer established semantics from a
factorized branch name.

This is a bounded empirical disclosure audit, not a proof of privacy. It does
not provide differential-privacy guarantees, bound arbitrary attackers, model
re-identification in a deployed system, or validate privacy for real people.

## Protected audit boundary

An attack may consume only:

1. an **authenticated, frozen representation export** (cutoff or dense, as
   declared by the audit specification), including its public component schema;
2. public preparation provenance needed to authenticate and stratify it,
   including cutoffs, ordered fields, train-fitted preprocessing identity,
   observed-source hashes, public per-user history/cutoff coverage, and export
   lineage; and
3. labels and partitions assembled inside the protected evaluator for the sole
   purpose of fitting or scoring the declared attack.

Before opening any evaluator-only label source, `audit-privacy` must verify the
export schema and bytes, evidence-index identity, checkpoint (when applicable),
selection role, preparation metadata/definition, observed-source hashes,
component order and dimensions, user/cutoff keys, finiteness, and all supplied
utility-report identities. Authentication failure aborts without a report.
Attack features must not include checkpoints, model parameters, gradients,
optimizer state, prediction heads, raw event rows, truth columns, or undeclared
side information. Public provenance may define matching strata but must not be
silently promoted into additional attack features; the report lists the exact
ordered attack feature schema for every attack.

Membership assignments and protected sensitive labels are evaluator-only audit
data. Sensitive labels may be joined to frozen vectors only after export and
split authentication. They must remain outside model constructors, model and
preparation datasets, collators, training APIs, checkpoint selection, ordinary
exports, ranking heads, and all observed-only commands. Ordinary exports must
never contain membership labels, sensitive labels, probe predictions, or stable
row ordering that encodes them. The protected report may contain aggregate
metrics and counts, but no per-user labels, scores, vectors, or reconstructable
small cells.

## Membership inference

### Membership unit

The primary membership unit is a **user in the target representation's encoder
training cohort**. A member contributed at least one eligible training event or
window to target-model parameter fitting. A non-member contributed none. The
label is fixed per `(target_model_lineage, user_id)` and cannot vary by cutoff,
component, or attack row. Validation users, test users, users seen only by
post-training evaluation, and users encoded only after checkpoint freezing are
non-members. Users used to fit train-only preprocessing are reported separately;
they cannot be treated as clean non-members for a claim about the whole pipeline.

The attack example is one user, not one event, window, session, export row, or
user/cutoff pair. Multiple frozen cutoffs or components for a user form one
grouped feature record (with an explicit missingness mask), preventing repeated
rows from inflating sample size or crossing attack splits. Event-, record-, and
session-membership attacks are out of scope until separately specified.

### Member and non-member construction

Construct the eligible pool before looking at attack outcomes. Members come
from authenticated target-training participation records. Non-members come
from held-out users in the same immutable dataset/preparation lineage that were
never used for target fitting or checkpoint selection but can be encoded with
the frozen encoder. A non-member may not come from another source hash, seed,
simulation scenario, preprocessing fit, or cutoff policy.

Member and non-member pools must share representation availability and be
matched or stratified using only public pre-attack variables fixed in the audit
specification, such as cutoff availability, public history-count bins, first/last
observable time bins, and observation/service coverage. Matching is without
replacement and deterministic under the audit seed. Report the eligible,
matched, unmatched, excluded, and realized counts by class and stratum. Never
discard hard examples, resample until significance, synthesize non-members, or
use sensitive labels to improve membership matching. If overlap or adequate
common support is absent, report the affected result as unavailable.

For the non-parametric statistical history vector, membership in model fitting
is not naturally defined because there are no learned target parameters.
Accordingly its primary training-membership attack is `not_applicable`; it must
still be reported as a diagnostic control for attribute disclosure and for an
explicitly labeled **pipeline-participation** attack only if that distinct
membership definition is frozen in advance. The two membership definitions
must never be pooled or compared as though identical.

## Attacker knowledge and access

The primary attacker is an **export recipient** who knows the representation
family and public preparation provenance, possesses the authenticated frozen
vectors for candidate users, knows the audit's membership base rate, and has a
separate labeled auxiliary population from the same lineage for attack fitting.
The attacker does not query the encoder, observe raw histories, parameters,
gradients, losses, truth, or the target model's validation/test outcomes.

Every result must name its access level. The required primary setting is
black-box frozen-export access. A future shadow-model, score/query, gradient,
or parameter-access attacker is a separate threat model and cannot be mixed
with this report. Attack hyperparameters and decision thresholds are selected
only on the attack validation split. Test labels are opened only for final
scoring; test outcomes cannot choose features, strata, attacks, components,
regularization, privacy transformations, or a representation.

## Split protocol

There are two distinct separation boundaries:

1. **Target-model separation.** Encoder training, encoder validation/checkpoint
   selection, and encoder test users follow the authenticated preparation
   contract. The audit cannot change these roles. Encoder-test users are not
   converted into attack-development users after results are inspected.
2. **Attack separation.** After member/non-member construction, assign whole
   users to attack-train, attack-validation, or attack-test by seeded canonical
   user hash, stratified by membership class and the frozen matching strata.
   These sets are mutually disjoint and shared across all representations and
   attacks in a matched comparison.

All rows, cutoffs, and components for a user remain in one attack split.
Preprocessing for an attack (imputation, scaling, dimensionality reduction,
class weighting) is fit on attack-train only. Attack validation selects its
hyperparameters and threshold. Attack-test is used once for frozen evaluation.
Sensitive-attribute probes use the same user-level isolation, but receive their
own versioned train/validation/test split hash if eligibility differs. No target
test label, privacy test result, or utility test result may influence target or
attack selection. Report and reject every target-role or attack-split overlap.

## Sensitive-attribute audits

An attribute is eligible only when it is named before fitting, has an explicit
privacy rationale, a versioned label derivation, sufficient support, and can be
joined inside the protected evaluator without entering the observed/modeling
contract. Initial eligible families are simulator-defined demographic groups
(for example age group and household type), service-adoption or observation
propensity groups when stored as protected truth, and discretized persistent
latent traits. Continuous latent traits must use frozen bins fitted without the
probe test labels. Exact coordinates, free text, raw identifiers, true episode
IDs, chosen flags, true utility, and sparse intersectional combinations are not
eligible sensitive-probe targets under this contract.

Observed attributes that are already explicit attack inputs are not evidence of
embedding leakage and must be marked `public_or_observed`, not sensitive-probe
success. Each audited attribute records whether it is evaluator-only, why it is
sensitive in this synthetic setting, classes/bins, missing-label policy, and
minimum class/cell count. Missing labels are never imputed. Rare classes are
merged only by a predeclared semantically defensible mapping; otherwise the
attribute or intersection is excluded with counts. Results for synthetic labels
are simulator diagnostics and do not imply the same attributes, distributions,
or harms in a real population.

The initial executable allowlist is code-versioned in
`geoembeddings.privacy.SUPPORTED_PROTECTED_ATTRIBUTES`. It contains protected
train-tertile derivations for canonical `user_latents` fields
`price_sensitivity` and `family_orientation`, plus explicit non-probe
declarations for observed `age_group` and `household_type`. Quantile boundaries
are fitted from probe-train users only. The loader reports aggregate eligibility,
missingness, unsupported, and class counts only after minimum total, class, and
class-by-split cell support passes; excluded small-cell counts are not emitted.
There is no rare-class merge in version 1 because no semantic merge mapping is
predeclared. Exact coordinates, identifiers, episode/decision IDs, chosen flags,
utility, text, and undeclared intersections fail before a truth file is opened.

## Imbalance and attack baselines

The report must preserve the natural eligible membership prevalence and class
prevalence. Primary membership AUC is prevalence-invariant, but balanced
accuracy, precision/recall, average precision, and thresholded risk must state
the natural base rate. A secondary deterministically downsampled 1:1 membership
analysis may diagnose separability; it is labeled secondary, repeats across
fixed seeds, and never replaces the natural-prevalence result. Training may use
inverse-frequency weights computed on attack-train only. Test resampling,
oversampling before user splitting, synthetic minority examples, or reporting
accuracy alone is prohibited.

At minimum, freeze and report these membership attacks:

- random and majority/base-rate controls;
- a provenance-only logistic attack using the declared public history and
  coverage variables, with no vector features;
- a regularized linear logistic attack on the frozen vector and declared
  missingness features; and
- a small nonlinear attack (for example a one-hidden-layer MLP) with parameter
  count and tuning budget frozen and reported.

Report the vector attack both alone and with the same provenance covariates;
the increment beyond provenance is a separate axis. Sensitive-attribute probes
require majority/prior, provenance-only, regularized linear, and the same
bounded nonlinear baselines. Multiclass attributes report macro-F1, balanced
accuracy, one-vs-rest macro AUC where defined, and per-class support; binary
attributes additionally report ROC AUC and average precision. A more powerful
attack succeeding is evidence of measured disclosure under this threat model;
an attack failing is not evidence that the information is absent.

## Uncertainty and comparisons

All confidence intervals use a predeclared method and confidence level. The
default is a seeded stratified **user-level bootstrap** of attack-test users,
with all of a user's features and labels resampled together. Report point
estimate, two-sided 95% percentile interval, replicate count, seed, successful
replicates, and degeneracy/exclusion counts. Do not bootstrap export rows or
cutoffs independently. For matched representation deltas, resample the same
users and compute paired bootstrap intervals. When a replicate lacks a class,
the metric is undefined for that replicate rather than coerced to zero.

Multiplicity is explicit: the report lists the number of representations,
components, attacks, attributes, and slices examined and identifies primary
endpoints fixed before evaluation. Confidence intervals are uncertainty
descriptions, not pass/fail privacy certificates. Post-hoc selection of the
largest attack or smallest leakage must be labeled exploratory.

## Diagnostic-control and utility/privacy reporting

One audit may compare only authenticated exports with identical source,
preparation, user eligibility, cutoffs, and attack splits. It must include:

- the statistical history baseline (non-parametric, with training membership
  marked as not applicable as described above);
- the `capacity_matched_single` single-vector diagnostic control; and
- every eligible factorized diagnostic variant, reporting `combined` and each
  exported component separately.

Every entry retains `selection_role: diagnostic_control`, model variant,
component identifier, parameter count, failed/unsatisfied selection gate,
checkpoint/export hashes, and evidence-index/decision identities. Factorized
component names are hypothesized branch identifiers, not established semantics.
Report membership disclosure, sensitive-probe disclosure, and utility as
separate axes, with paired deltas and uncertainty where identities match. A
utility/privacy table or Pareto plot may show each representation/component and
privacy transformation at fixed utility metrics; it must not collapse axes into
one score, rank an aggregate winner, or treat reduced utility as privacy.

Because there is currently no `selected_candidate`, the report-level field
`selection_dependent_privacy_conclusion` must be `unavailable` with reason
`no_selected_candidate`. Diagnostic controls may reveal failure modes and guide
future mitigations, but cannot complete the selection-dependent T4.3 conclusion.

## Future `audit-privacy` artifact contract

The proposed command writes immutable `audits/privacy.json` and
`audits/privacy.md` under a versioned schema such as
`geoembeddings-privacy-audit/1.0`. If either path exists, the command fails
before evaluation unless explicit `--overwrite` names and validates the exact
audit output directory; replacement is atomic for both files. It never deletes
or mutates source runs, experiments, exports, checkpoints, utility reports, or
prior audit directories. The JSON is authoritative and the Markdown is a
rendering of the same content.

The report schema must include:

| Section | Required content |
|---|---|
| `schema_version`, `command`, `created_at`, `runtime_metadata` | Audit identity, exact invocation, code/seed/runtime provenance |
| `threat_model` | This document's version/hash, membership definition, attacker access/knowledge, allowed features, prohibited inputs, primary endpoints |
| `inputs` | Canonical paths and SHA-256 for every frozen export, checkpoint or `not_applicable`, evidence index, decision record, utility report, and evaluator-only label source |
| `lineage` | Dataset contract, source-manifest identity, **observed source hashes**, **preparation metadata and definition hashes**, ordered fields, vocabulary/statistics hashes, cutoffs, component schema, model variant and parameter count |
| `splits` | Target train/validation/test role counts and hashes; attack train/validation/test **user-set hashes**; sensitive-probe split hashes; overlap checks and assignment algorithm/seed |
| `membership_population` | Unit, member/non-member derivation, natural prevalence, matching strata, requested/realized class counts, common-support diagnostics |
| `sensitive_attributes` | Eligibility rationale, provenance class, derivation/version, classes or bins, support, missingness, rare-cell action |
| `attacks` | Ordered feature schema, baseline/attack family, hyperparameter space, selected validation configuration, fit seed, parameter count, class-weighting/resampling policy |
| `membership_metrics` | Test ROC AUC and 95% interval as the primary endpoint, average precision, balanced accuracy, precision/recall, base rate, threshold rule, paired deltas, valid bootstrap counts |
| `sensitive_probe_metrics` | Per-attribute and per-component macro-F1, balanced accuracy, applicable AUC/AP, per-class support, uncertainty, provenance-only and vector-increment comparisons |
| `utility_privacy_axes` | Authenticated utility metrics and coverage beside—not averaged with—membership and attribute disclosure; paired deltas/intervals and optional Pareto membership |
| `coverage` | Eligible/evaluated users, vectors, cutoffs, components, labels, classes/strata, bootstrap replicates, and denominators for every metric |
| `exclusions` | Machine-readable reason and count for missing exports/labels, sparse classes, unmatched users, non-finite values, unavailable cutoffs/components, and undefined metrics; no user identifiers |
| `selection` | Immutable role per representation, gate/decision evidence, and `selection_dependent_privacy_conclusion` with availability and reason |
| `limitations` | Synthetic-data boundary, attacker/access limits, finite power, imbalance and common-support limits, multiplicity, attribute validity, absent formal guarantee, and untested attacks |

Source, preparation, and user-split hashes are comparability gates, not merely
descriptive metadata. A comparison aborts on any mismatch in observed sources,
preparation identity, membership definition, eligible user set, cutoff/component
schema, attack split, sensitive-label derivation, attack specification, or
utility population. Partial coverage remains explicit and is never filled from
another lineage.

## Prohibited interpretations

The audit must not claim or imply:

- that AUC near 0.5 proves privacy, anonymity, non-memorization, or absence of
  sensitive information;
- that high AUC identifies which individual users were members with certainty;
- that a confidence interval crossing 0.5 establishes equivalence or safety;
- that lower probe performance is a formal privacy guarantee or is beneficial
  when caused by collapse, missing coverage, weak attacks, imbalance, or lost
  utility;
- that one synthetic sensitive attribute is a proxy for all protected classes
  or that simulator group labels describe real people;
- that membership disclosure and attribute disclosure are interchangeable;
- that component names establish persistent/context privacy semantics;
- that the best diagnostic control is selected, deployable, fair, legally
  compliant, differentially private, or safe against stronger attackers; or
- that a simulator-only audit certifies a real deployment, production access
  controls, data retention, consent, deletion, linkage, or re-identification
  risk.

## Implementation acceptance gate

Implementation requires unit tests for membership construction, grouped user
splits, target/attack separation, matching, imbalance, undefined metrics,
bootstrap determinism, rare labels, and schema validation. A cross-stage test
must prove that only authenticated frozen exports and public preparation
provenance reach attack feature construction, while evaluator-only membership
and sensitive labels cannot enter model constructors, datasets, training APIs,
checkpoints, or ordinary exports. Integration tests must reject source,
preparation, user-set, cutoff/component, role, split, label-derivation, attack,
and utility-population mismatches and verify immutable two-file output behavior.
Completion also requires matched utility/privacy reports with coverage,
exclusions, uncertainty, limitations, and the selection-dependent conclusion
correctly marked unavailable until a genuine `selected_candidate` exists.
