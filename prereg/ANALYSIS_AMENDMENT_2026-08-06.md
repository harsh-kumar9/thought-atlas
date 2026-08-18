# Analysis amendment: two-family paper analyses

**Specification date:** August 6, 2026  
**Amendment recorded:** August 9, 2026  
**Current commit at recording:** `90bad30c8a3e0a1e0860969bc84a133e99bd170a`

This document is an honest ledger. It does not retroactively make post-hoc analyses preregistered.

## Frozen before the current paper analysis

The repository's original `prereg/PREREGISTRATION.md` froze:

- H1: domain×behavior differences in behavior configurations;
- H2: temporal heartbeat differences across domains;
- H3: artifact robustness across think/response traces;
- H4: temporal structure relative to a within-trace label-shuffle null;
- the eight-label taxonomy;
- count/presence models and the broad domain×model×artifact formula;
- functional heartbeat and HMM intentions.

The original document also records later post-hoc additions for timing-versus-level decomposition, pooled prefix monitoring, and the safety/security domains.

## Proposed after aggregate results and therefore not preregistered

The following analyses were motivated after the original aggregate atlas and nearby-work audit were available:

- separation into dialectical and executive-control families;
- `SOT_ALL`, `SOT_CORE`, and questioning-only sensitivity;
- family separability and residual co-occurrence;
- amount-versus-conditional-shape repair;
- cross-family motifs and predictive complementarity;
- contextual monitorability from paired full/isolated labels;
- adaptive differentiation;
- completion/censoring sensitivities;
- lexical-cue robustness;
- K-line-inspired coalition recurrence.

These must be labeled post-hoc or exploratory unless separately frozen before their code is run.

## Analyses run in the interim `paper_results` package

The August 9 interim package was implemented after reading the paper-worthiness audit but before this execution specification was supplied in the current task. It includes:

- domain/model variance decomposition for family amount, centroids, and coupling;
- model-balanced family trajectory plots;
- matched-prompt Qwen/Gemma contrasts;
- backward-chaining hurdle summaries;
- an RMSE-based adaptive-differentiation ratio;
- six adjacent motif enrichments using an analytic position-aware expectation;
- domain-specific motif associations;
- prompt-disjoint prediction using counts, coarse timing, and motif blocks;
- paired full/isolated context metrics;
- observational later-versus-earlier outcome associations.

All are post-hoc relative to the frozen preregistration. Motif identities were theory-motivated by the paper audit, but the implemented set and null do not match the later execution specification exactly; they should be treated as exploratory.

## Held-out status of the interim analyses

- Prompt-disjoint five-fold evaluation: used for predictive models.
- Prompt-cluster bootstrap: used for motif enrichment and context intervals.
- Leave-one-domain-out: not used.
- Leave-one-model-family-out: not used.
- Human/cross-judge holdout: not available.
- Multi-seed holdout: not available.

## Prospective analyses that may be frozen now

Before implementing new scientific models, freeze a config and code-independent feature dictionary for:

1. RQ1 family-profile separation with `SOT_CORE` sensitivity;
2. the exact M0–M5 outcome ladder;
3. the complete pre-specified motif set, lags, and two nulls;
4. context stratification and validation sampling;
5. family-specific adaptive differentiation and successful-prototype alignment;
6. completion, fixed-budget, lexical-cue, and outcome robustness.

The K-line analysis remains exploratory even if implemented later.

## Release-exception decision recorded August 9, 2026

The project owner confirmed that the remaining upstream rows cannot be retried or repaired and approved retaining them as appendix limitations. The paper-analysis config therefore accepts only these four exact strict-audit messages:

- 13 invalid answer-extraction rows require retry;
- 1,622 extraction prompts were truncated at the configured window;
- one accepted extraction fails downstream consumer validation;
- 673 Track-B rows contain parse failures across full and isolated contexts.

This does not turn the underlying rows into valid observations. Outcome analyses retain null outcomes, behavior analyses exclude unparsed labels from denominators, and all coverage losses remain reported. Any new or textually changed audit issue remains blocking. This is a post-hoc release-governance decision, not a change to the scientific hypotheses.
