# Pre-registration — society-task-exp2 (freeze BEFORE fitting any model)

Date frozen: ____  | Commit: ____

## Hypotheses
H1. Behavior configurations differ by task domain within each generation model
    (domain × behavior interaction, rate ratios > 1.5 for ≥4 of 8 behaviors).
H2. Temporal "heartbeat" shapes differ across ≥3 domains (functional ANOVA, FDR-controlled).
H3. Effects hold for BOTH think-block and response artifacts in the reasoning model.
H4. Temporal structure exceeds a within-trace label-shuffle null.

## Behavior taxonomy (8 codes)
Kim conversational: Question_and_Answering, Perspective_Shift, Conflict_of_Perspectives, Reconciliation.
Gandhi cognitive: verification, backtracking, subgoal, backward_chaining.
Per-segment label representation: multi-label presence, two taxonomy passes, full-context primary.

## GLMM formula (freeze)
Counts: negative-binomial; Presence: logistic.
  behavior_outcome ~ domain * model * artifact_type + (1 | problem_id) + (1 | instance_seed)
Cluster-robust SE alternative reported. Per-100-token rates descriptive only (not headline).

## Temporal
Heartbeat: norm_pos in [0,1] (ThinkARM convention), per-behavior normalized frequency,
panels per behavior, lines per domain, mean±SD across models. Analysis code supports
minimum segment/token guards; cross-model similarity defaults to completed traces only.
Functional ANOVA via scikit-fda or the lightweight paper-figure proxy.
Null: shuffle behavior labels within trace (preserve counts), recompute, compare.

## HMM regimes (pre-register ranges)
math 3-5 · code 2-4 · gpqa 3-5 · planning 3-5 · moral 2-4 · idea 2-4. Select within range by BIC/held-out LL.
Test (a) regime-count differs by domain, (b) emission alignment to theory (Hotelling T² / Dir-mult LRT).

## Stratified analyses
By correctness (per domain) · by artifact (think vs response, never lumped) · by failure mode.

## Decision gate to SAE phase
H1–H4 all supported → proceed. Otherwise reframe as "routing weaker than prior work" (still publishable).

## Post-hoc additions

2026-07-02: Added the Timing vs level decomposition after inspecting the data.
It descends from H4's temporal shuffle-null motivation, but it was not part of
the frozen preregistration. Treat it as exploratory: it quantifies whether
outcome-linked behavior gaps are level-like, timing-like, both, or neither using
the dashboard heartbeat trace population and trace-clustered bootstrap covariance.

2026-07-03: Added prefix monitorability as a post-hoc predictive analysis. It
uses nested regularized logistic monitors and prompt-disjoint folds to test
whether early behavior prefixes predict final success/high-quality outcomes and
whether temporal bins add value beyond metadata, prefix length, and behavior
counts. Percentage prefixes are retrospective diagnostics, not deployable online
intervention budgets.
