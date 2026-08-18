# Thought Atlas analysis-specification coverage audit

**Specification audited:** `/Users/harsh/Downloads/THOUGHT_ATLAS_CODEX_ANALYSIS_SPEC.md`  
**Specification version:** August 6, 2026  
**Audit date:** August 9, 2026  
**Current commit:** `90bad30c8a3e0a1e0860969bc84a133e99bd170a`

## Bottom line

**Update after the staged final run:** This document audits the earlier monolithic `paper_results/` package and is retained for provenance. The project owner subsequently approved exact appendix exceptions and the staged package generated RQ1–RQ7 outputs under `data/v2/analysis/paper/`, with the authoritative report at `docs/paper_analysis/generated/ANALYSIS_REPORT.md`. Remaining limitations—especially approximate motif nulls and uncompleted human/cross-judge validation—are stated in that report and finding registry.

No—the current `paper_results` package does **not** cover the complete analysis specification.

It is a compact exploratory analysis that covers important parts of domain conditioning, amount/timing separation, adjacent motifs, contextual labeling, scale contrasts, and adaptive differentiation. It does not implement the specification's mandatory family-validation analysis, Society-of-Thought sensitivity split, exact nested outcome ladder, transfer tests, prescribed motif nulls, full robustness suite, or staged engineering architecture.

The existing numerical results remain useful, but they should be described as **interim and partially specification-aligned**, not as the final analysis package defined by the specification.

## Research-question alignment

| Specification RQ | Current status | What is covered | What is missing or materially different |
|---|---|---|---|
| **RQ1 — Are dialectical and executive-control operations empirically distinguishable?** | **Missing** | The report separates the families conceptually and compares some domain, timing, and context summaries. | No standardized behavior-profile vectors; no within- versus cross-family similarity statistic; no exact balanced 4+4 partition test; no stratified pairwise co-occurrence; no mandatory `SOT_CORE` versus questioning-only sensitivity; no incremental predictive complementarity by family. The current results cannot validate the 4+4 architecture empirically. |
| **RQ2 — How do domain and model alter amount, timing, and coupling?** | **Partially covered** | Trace-level amount, centroids, three-phase allocation, 20-bin family curves, coupling excess, domain×model variance partition, matched Qwen/Gemma comparisons, rare backward-chaining hurdle table. | Family density is reported as labels per segment rather than divided by four; no family diversity/onset/entropy; no behavior-level effect-size CIs; no Jensen–Shannon/correlation/Wasserstein shape-distance suite; no completed-only, fixed-4096, or IPW sensitivity. |
| **RQ3 — Are Society-of-Thought behaviors sufficient, or useful through executive coupling?** | **Mostly missing** | Prompt-disjoint models compare all behavior counts with timing and motif blocks. | The required M0, M1-D, M1-E, M2, M3, M4, M5 ladder was not run. There is no `ΔExecutive|Dialectical` or `ΔDialectical|Executive`; no SOT-core/questioning split; no continuous moral/idea/safety primary models; no paired bootstrap intervals for metric deltas; no model-transfer setting. The current prediction results do not answer the specification's RQ3. |
| **RQ4 — Do cross-family motifs distinguish outcomes and generalize?** | **Partially covered** | Six adjacent motifs, position-aware analytic expectation, prompt-cluster bootstrap CIs, within-domain outcome associations controlling constituent rates, prompt-disjoint prediction. | Only six of the prescribed motifs; no perspective-shift→verification or conflict→reconciliation primary pair; no length-3 motifs; no lag-2 or maximum-gap motifs; no independent circular-shift null; no decile-constrained 1,000-permutation null; no leave-one-domain-out or leave-one-model-family-out transfer. Current motif enrichment is exploratory relative to the specification. |
| **RQ5 — Which operations require trajectory context?** | **Partially covered** | Exact paired keys; disagreement, directional prevalence, retention, raw agreement, κ; behavior/family summaries; full-versus-isolated prompt-disjoint outcome models. | No model/domain/phase/outcome stratification; no prevalence-adjusted agreement; no clustered disagreement model; no SOT-core family comparison; no coefficient-stability analysis; no cross-judge detection or 1,200–2,000-segment validation sample. |
| **RQ6 — Is capability associated with adaptive differentiation?** | **Partially covered** | Repeated split-half denominator, between-domain/within-domain ratio, bootstrap interval, matched Qwen/Gemma amount and centroid comparisons, capability correlation. | Uses RMSE rather than family-wise Jensen–Shannon distance; combines the full eight-label amount profile and both family shapes rather than reporting dialectical/SOT-core/executive separately; 200 split halves and 500 interval resamples rather than the final 1,000; no successful-prototype outcome alignment; no completed/fixed-budget/IPW sensitivity. |
| **Exploratory RQ7 — K-line-inspired coalition reinstatement** | **Not run** | None. | Optional coalition windows, prototypes, held-out reinstatement scores, nulls, and interpretation gate were not implemented. This should remain future work until RQ1–RQ6 are complete. |

## Phase-by-phase implementation coverage

| Specification phase | Status | Current artifact | Main gap |
|---|---|---|---|
| **Phase 1 — Audit and manifest** | **Implemented; gate blocked** | Prescribed `00_audit/` tables, exact pair coverage, outcome coverage/direction, analysis-source/budget table, flag registry, input/output SHA256 manifest, and deterministic dev sample. | Strict-v2 still returns `ok: false`, so the manifest records `paper_analysis_ready=false` and later scientific stages remain stopped. |
| **Phase 2 — Family structure** | Missing | No direct equivalent. | Entire empirical family-separation and SOT sensitivity package is absent. |
| **Phase 3 — Amount versus shape** | Partial | Trace amount/rates/centroids, 20-bin conditional shapes, matched-scale contrasts, hurdle summary. | Required distances, 1,000 prompt bootstraps, and completion/budget/IPW populations are absent. |
| **Phase 4 — Trace orchestration features** | Partial | `cache/trace_features_full.parquet` and isolated counterpart. | Missing presence/diversity, first/last, SD, entropy, runs, recurrence, SOT-core, lag-2, within-5, transition balance/share, feature dictionary, missingness/correlation tables, and feature manifest. |
| **Phase 5 — Outcome models** | Partial | Prompt-disjoint ridge models and selected clustered motif regressions. | Wrong feature ladder and primary outcome treatment; no OOF predictions, bootstrapped metric CIs, continuous-outcome metrics, or transfer. |
| **Phase 6 — Motifs** | Partial | Adjacent motif lift and outcome tables. | Incomplete motif set, incomplete lag/gap structure, non-prescribed null, no transfer. |
| **Phase 7 — Context** | Partial | Paired behavior/family metrics and context prediction comparison. | Missing stratified disagreement analyses, clustered model, PABAK, validation sample. |
| **Phase 8 — Adaptive differentiation** | Partial | Amount/shape differentiation table and scale contrasts. | Not family-specific as specified; no successful-prototype alignment or required robustness populations. |
| **Phase 9 — K-line** | Missing | None. | Optional; correctly deferred. |

## Robustness-contract coverage

| Robustness requirement | Status | Notes |
|---|---|---|
| Full-context labels | Covered | Primary current analysis source. |
| Isolated-label replication | Partial | Context metrics and prediction comparison only; headline family/domain analyses were not rerun in isolation. |
| Cross-judge/human validation | Missing | No additional judge files detected or validation sample generated. |
| Lexical-cue baseline and cue ablation | Missing | Not run. |
| All valid versus completed-only | Missing for headline analyses | Coverage is shown, but headline estimates were not repeated on completed-only traces. |
| Fixed 4,096-token population | Missing | Segment labels are available, but the specified token-aligned sensitivity was not implemented. |
| Inverse-probability completion weighting | Missing | Not run. |
| Length-matched subsample / efficiency | Missing | Log segment length is controlled, but the other length robustness analyses are absent. |
| Prompt-disjoint prediction | Covered | Five deterministic folds grouped by `instance_id`. |
| Leave-one-domain-out | Missing | Not run. |
| Leave-one-model-family-out | Missing | Not run. |
| Continuous moral/idea primary outcome | Missing | Current prediction/inference uses the repository median split. Continuous quality appears only in capability scoring. |
| Continuous safety harm primary outcome | Missing | Current general endpoint uses absence of high harmful compliance. |
| Safety thresholds .25/.50/.75 | Missing | Not run. |
| Multiple-testing correction | Partial | BH is used for motif/timing families, but the specification's full test-family registry is absent. |
| Multi-seed validation | Missing | The release has one sample seed per prompt. |

## Engineering and artifact-contract coverage

| Requirement | Status | Difference |
|---|---|---|
| Staged `src/analysis/paper/` package | **PR1 covered** | Full prescribed module scaffold exists; data/audit modules are implemented and later scientific modules are explicitly gated placeholders. |
| `configs/paper_analysis.yaml` | **Covered** | Config is hashed in the analysis manifest and records channels, budgets, paths, comparisons, filters, and resampling settings. |
| Stage-selectable runner and dev mode | **PR1 covered** | `scripts/run_paper_analysis.py` supports every prescribed stage name plus `--dev`, `--force`, and `--jobs`; post-PR1 stages remain gated. |
| Polars lazy scan; do not materialize Track B in pandas | **Covered for PR1** | Track-B scans, joins, pairing, and coverage aggregation use Polars lazy frames; only aggregate tables are collected. |
| Prescribed output root | **Dev covered** | The validated PR1 run writes to `data/v2/analysis/paper_dev/`; final-mode outputs are intentionally not generated while the readiness gate is false. |
| Final ≥1,000 bootstrap/permutation reps | Not compliant | Current run uses 500 bootstrap resamples and 200 split-half repetitions; motif null is analytic, not 1,000 permutations. |
| Analysis amendment | Added during this alignment pass | See `prereg/ANALYSIS_AMENDMENT_2026-08-06.md`. |
| Onboarding/progress documents | Added during this alignment pass | See `docs/paper_analysis/`. |
| Paper-specific unit and integration tests | **PR1 covered** | Eleven synthetic IO, outcome, split, exception-policy, and audit-smoke tests pass; combined suite is 48/48. Later scientific modules still need their prescribed tests in their own PRs. |
| Finding registry | Missing | No row-level claim registry with status/estimand/robustness fields. |
| Exact plotted-value sidecars | Partial | Most figures map to CSVs; the atlas curve values are in the trace-feature cache rather than a dedicated plotted-values file. |
| Conceptual and robustness figures | Missing | The current eight figures do not include the required conceptual diagram, amount-versus-shape comparison panel, or consolidated robustness panel. |

## Strict-v2 audit status

The combined required audit was rerun on August 9, 2026. It found all canonical Parquet files present, valid Parquet magic bytes, valid trace manifests, 24,416 unique traces, no duplicate natural keys, and exact full/isolated segment-key coverage. However, it returned:

```text
ok: false
issues:
  - 13 invalid answer-extraction rows require retry
  - 1,622 answer-extraction prompts were truncated
  - 1 accepted row fails downstream consumer validation
  - 673 Track-B rows contain parse failures
```

`DATASET.md` describes the invalid and truncated extraction rows as accepted release exceptions. The implementation specification is stricter: Section 6 says to stop if strict-v2 fails. The next implementation must either repair these rows or record an explicit, reviewed amendment to the stop rule before treating later phases as specification-complete.

The behavior-label inputs used by the current trace analyses are present and readable. This audit failure is therefore a governance/acceptance blocker under the new specification, not evidence that all existing descriptive behavior estimates must be discarded.

## Priority order for remaining work

### P0 — Restore a valid execution baseline

1. Resolve or formally adjudicate the strict-v2 extraction exceptions.
2. Review and commit the PR1 audit artifacts; do not start scientific modeling while `paper_analysis_ready=false`.
3. Once the gate is resolved, begin PR2 and write final derived outputs to the prescribed versioned directory without deleting the interim package.

### P1 — Scientific blockers for the central paper claim

1. Run the mandatory RQ1 family-separation analysis and the `SOT_ALL`/`SOT_CORE`/questioning-only sensitivity.
2. Run the exact M0–M5 nested outcome ladder with continuous moral, idea, and safety outcomes.
3. Complete the pre-specified motif set, lag-2/length-3 features, both required nulls, and held-out domain/model-family transfer.
4. Stratify context sensitivity and generate an independent validation sample.
5. Recompute adaptive differentiation separately for dialectical, SOT-core, and executive tensors, then add successful-prototype outcome alignment.

### P2 — Robustness needed for submission

1. Completed-only, fixed-4,096-token, inverse-probability, and length-matched analyses.
2. Lexical-cue baseline and cue-removal validation.
3. Continuous/threshold sensitivity for open-ended, safety, and security outcomes.
4. 1,000-replicate final intervals/nulls and paper-specific synthetic tests.
5. Finding registry, robustness summary, and exact plotted-value sidecars.

### P3 — Optional follow-up

Run K-line-inspired coalition reinstatement only after P0–P2. It is not required for the moderate NAACL framing and cannot establish memory retrieval.

## Correct interpretation of the current package

The current results provisionally support the specification's **moderate outcome** framing:

> **Thought Atlas reveals domain- and lineage-specific organizations of visible deliberation and identifies operations whose classification changes with trajectory context.**

They do not yet meet the stronger framing, because family separability is untested, cross-family predictive value is near zero, transfer is untested, and the full robustness contract is incomplete.
