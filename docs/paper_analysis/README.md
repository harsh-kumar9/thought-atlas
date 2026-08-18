# Paper analysis: specification, provenance, and reader guide

This directory documents the paper-oriented analysis of the Thought Atlas v2
release. It is the best starting point for understanding what was measured, how
the results were produced, which claims are supported, and which limitations are
still material.

The short conclusion is deliberately moderate:

> Visible deliberation differs reliably by task domain and model, and some labels
> depend on surrounding trajectory context. The data do not support a clean
> two-family separation, a universal scaling law, a general outcome benefit from
> coupling or motifs, or a causal interpretation of the observed signatures.

## Which artifacts are authoritative

- `generated/ANALYSIS_REPORT.md` is the authoritative RQ1–RQ9 result narrative.
- `generated/REGRESSION_MODEL_LOG.md` records the regression estimands, feature
  ladders, correction families, and held-out scoring rules.
- `generated/SIGNATURE_ANALYSIS_REPORT.md` gives the detailed RQ8–RQ9 domain and
  attempt-quality results.
- `../../data/v2/analysis/paper/finding_registry.csv` is the machine-readable
  claim ledger.
- `../../data/v2/analysis/paper/final_analysis_manifest.json` hashes the generated
  output bundle.
- `../../configs/paper_analysis.yaml` is the frozen analysis configuration.
- `../../prereg/ANALYSIS_AMENDMENT_2026-08-06.md` distinguishes preregistered,
  post-hoc, and exploratory work.

`../../paper_results/` is an earlier exploratory analysis package. It remains in
the repository both for provenance and because the staged implementation reuses
some of its audited feature caches and tables. Its report is not the final claim
source when it differs from the generated report in this directory.

`PR1_AUDIT_REPORT.md` is a historical stop decision written before the owner
adjudicated four exact release exceptions. `PROGRESS.md` and the full audit
manifest record the later decision and current analysis-ready status.

## Analysis population

The frozen v2 release contains:

- 24,416 traces from 3,488 prompt instances;
- eight domains: math, code, GPQA, planning, moral, idea, safety, and security;
- seven observed model conditions: Llama 3.1 8B Instruct (`anchor`), DeepSeek-R1
  Distill 8B, Gemma 4 E4B/31B, and Qwen3.5 4B/9B/27B;
- one configured but absent condition, Gemma 4 12B;
- 4,635,520 full-context and 4,635,520 isolated-context Track-B segment rows,
  paired exactly on `trace_id + seg_idx`;
- one generation seed, 42, per prompt/model condition.

The main reasoning-trace analyses use 18,624 traces from the six reasoning-model
conditions. The Llama anchor is retained in coverage and audit outputs but is not
treated as a comparable reasoning trace: it uses `answer_text` with a 4,096-token
budget, while the reasoning conditions use `think_text` with a 65,536-token
budget. Any anchor-versus-reasoner contrast therefore combines channel, budget,
training, and model differences.

Safety and security are post-hoc extension domains. Security correctness is a
hazardous-knowledge capability endpoint: higher WMDP-Cyber accuracy is not a
claim of safer behavior.

## Behavior and feature contract

The analysis groups the eight sentence-level labels into two theory-defined
families:

| Family | Labels |
| --- | --- |
| Conversational / dialectical | question-and-answering, perspective shift, conflict of perspectives, reconciliation |
| Cognitive / executive control | verification, backtracking, subgoal setting, backward chaining |

These are observable textual annotations, not direct measurements of hidden
cognition or internal agents. The prespecified grouping is itself tested in RQ1;
it is not assumed to be an empirically recovered latent structure.

Core feature definitions:

- **Amount** is the fraction of valid segments carrying a behavior. A zero is
  recorded only when a valid denominator exists; invalid judge rows stay null.
- **Conditional shape/timing** describes where a behavior occurs, conditional on
  it occurring. It is kept separate from amount so a frequent behavior cannot
  appear “late” merely because it occurs everywhere.
- **Timing contrasts** compare middle and late prevalence with early prevalence
  after normalizing within behavior.
- **Coupling** measures same-segment or cross-family co-occurrence beyond the
  marginal amount features.
- **Motifs** are specified ordered label transitions and three-step sequences.
- **Modal paths** show the most prevalent family-specific operation in each trace
  decile. Because labels may co-occur, these are summaries, not hidden states.

The exact generated feature dictionary is
`../../data/v2/analysis/paper/03_trace_features/feature_dictionary.csv`.

## Outcome contract

| Domain | Primary outcome | Type and direction |
| --- | --- | --- |
| Math, GPQA, planning | deterministic success | binary; higher is better |
| Code | sandboxed execution success | binary; higher is better |
| Moral, idea | answer-only rubric quality | continuous; higher is better |
| Safety | harmful-compliance score | continuous; lower is safer; reversed only in the common higher-is-better field |
| Security | WMDP-Cyber correctness | binary; higher means greater hazardous capability, not greater safety |

Missing or invalid outcomes remain missing. Completed traces without a valid
score are never silently relabeled as failure.

## What was analyzed

| RQ | Question and method | Result | Claim boundary |
| --- | --- | --- | --- |
| RQ1 | Test the prespecified 4+4 family split using behavior-profile similarity, prompt-composition bootstrap, and an exact balanced-partition comparison. | Partial support. Separation = -0.016, 95% CI [-0.051, 0.018], exact p = 0.514. | The two label groups remain useful descriptive vocabularies, but they are not cleanly recovered as two empirical systems. |
| RQ2 | Compare behavior amount and conditional timing across domains and models, with completion and fixed-budget sensitivities. | Supported. Domain and model organize both amount and shape across all 48 reasoning-model/domain cells. | There is one seed and unequal completion; intervals describe prompt composition, not decoding variability. |
| RQ3 | Fit the prompt-disjoint M0–M5 outcome ladder: metadata, each family, both families, timing, coupling, and motifs. | Not supported as a general complementarity claim. Mean coupling increment is approximately 0; effects vary by domain. | Held-out association is not causal benefit from adding a behavior. |
| RQ4 | Test ordered motifs, approximate null enrichment, outcome association, and held-out incremental value. | Partial support. Structure is enriched, but mean held-out motif increment is -0.0016. | The implemented nulls are count-level Monte Carlo approximations, not exact label-level permutations; motif confirmation is deferred. |
| RQ5 | Pair full-context and isolated labels on exact segment keys and measure disagreement/recoverability. | Supported as context sensitivity: 3.9% disagreement for conversational labels and 3.5% for cognitive labels. | The same judge produced both views; disagreement does not establish which view is valid. |
| RQ6 | Compare between-domain profile distance with repeated within-domain prompt-half distance. | Partial support. All reasoning models differentiate domains beyond prompt-composition variability; ratios span 5.43–16.74. | The ratio is not a capability score, and Qwen/Gemma do not show one universal scale trend. |
| RQ7 | Explore recurrence of visible behavior coalitions and apply a five-part interpretation gate. | Not supported. Only 2 of 5 gate criteria pass; mean outcome correlation is 0.085. | Visible recurrence is not evidence of memory retrieval or K-lines. |
| RQ8 | Within each model, predict held-out domain from amount and then conditional timing; localize all 28 domain pairs and use exact four-behavior Shapley attribution. | Supported. Amount helps in 12/12 model-family tests and 294/336 pairwise cells; timing helps in 94 pairwise cells. | Shapley allocates held-out predictive information, not causal influence or latent cognition. |
| RQ9 | Within model/domain, compare high/low sensitivity and stricter native-outcome prediction, controlling for length and available prompt strata. | Partial sensitivity only. Two high/low amount cells survive correction; native-outcome amount and timing survive in 0 of 88 estimable cells. | The two dichotomized findings are not confirmed by native outcomes; incomplete attempts are too sparse for a three-class analysis. |

Benjamini–Hochberg false-discovery control uses `q = 0.05`. Signature models use
five deterministic prompt-disjoint folds grouped by `instance_id`, train-fold-only
imputation and scaling, fixed-L2 ridge models, 1,000 paired stratified bootstrap
resamples, and paired sign-flip tests. Categorical gain is reported as held-out
bits per trace; continuous gain is the held-out change in R². Exact four-player
Shapley attribution fits all 16 subsets of the four labels in a family.

## Assumptions and non-claims

The analysis depends on the following assumptions, all of which constrain the
language used in the reports:

1. The production judge labels are useful operational measurements of visible
   text. They are not assumed to reveal private cognition faithfully.
2. `instance_id` is the correct leakage unit. All responses to a prompt stay in
   the same held-out fold.
3. Prompt resampling is a meaningful uncertainty source. With one decoding seed,
   it cannot estimate generation variance.
4. Valid labeled segments are representative enough for descriptive analysis
   after explicit coverage reporting. Missingness may still be informative,
   especially for lower-completion model/domain cells.
5. Full/isolated disagreement measures context dependence, not annotation truth,
   because there is no independent human or cross-judge validation set.
6. Outcome associations may reflect difficulty, selection, or response length
   even after controls. No observational result licenses “inducing this behavior
   improves the answer.”
7. Family, scale, and anchor contrasts are descriptive unless channel, token
   budget, lineage, and missing cells are jointly controlled.

Consequently, the repository does not claim a universal reasoning heartbeat,
two recovered inner agents, a universal scaling law, deployable safety monitoring,
or a causal benefit from motifs, timing, or behavior families.

## Data-quality decisions

The literal upstream strict-v2 audit returns false. On August 9, 2026, the owner
approved exactly four non-repairable issues as appendix limitations:

- 13 invalid answer-extraction rows;
- 1,622 extraction prompts clipped at the configured context window;
- one accepted extraction rejected by the downstream consumer contract;
- 673 Track-B parse-failure rows across full and isolated contexts.

The config stores the exact accepted strings. Any new or changed issue blocks the
analysis. Acceptance is a governance decision, not imputation: invalid outcomes
remain null, invalid labels are excluded from denominators, and coverage loss is
reported. The full effective audit is ready because all reported issues match the
frozen exception list and there are no additional issues.

The deterministic dev audit is retained under `../../data/v2/analysis/paper_dev/`.
It samples at most 200 traces per model/domain cell and was used to verify joins,
schemas, output contracts, and memory behavior before the full analysis. It is
not the result population.

## Engineering challenges and current mitigations

- **Large sentence-label tables.** The paired Track-B inputs contain 9.27 million
  rows. Audit and pairing use lazy Polars scans, explicit key validation, and
  streaming collection where available.
- **Exact pairing.** Full and isolated labels are joined only on
  `trace_id + seg_idx`; duplicate and unmatched keys are reported before any
  inner join can hide loss.
- **Leakage risk.** A SHA256-based fold registry assigns every shared prompt to
  one fold and tests enforce no `instance_id` overlap.
- **Rare and missing labels.** Backward chaining uses hurdle-style summaries;
  parse failures remain null; timing is undefined when a behavior does not occur.
- **Mixed outcome scales.** A registry fixes each domain's type, native direction,
  and interpretation before model fitting.
- **Compute cost.** Some intended 1,000-replicate operations are capped at 200 or
  500 in specific stages, and motif nulls use 1,000 count-level draws. Each such
  deviation is named in the final report.
- **Interim-package dependency.** `src/analysis/paper/final_analysis.py` currently
  consumes `paper_results/cache/trace_features_full.parquet` and selected interim
  tables for variance, motif, and context summaries. Run the interim builder
  first in a clean checkout. Removing this dependency is future refactoring work.
- **Stage granularity.** Apart from `audit` and the dedicated signature runner,
  named scientific stage entry points currently delegate to the cached full
  `run_all` build. Use `--stage all`; treat the other names as conceptual entry
  points rather than isolated jobs. `--jobs` is validated but the current final
  implementation should be run with `--jobs 1`.
- **Generated provenance.** The checked-in generated manifests record base commit
  `90bad30...` because the analysis code and outputs were produced while the
  release changes were still uncommitted. File/config SHA256 values identify the
  exact analyzed artifacts; the commit containing this document identifies the
  complete checked-in source snapshot. A clean rerun will record its new `HEAD`.
- **Absolute manifest paths.** Existing manifests contain the local generation
  path. Paths are descriptive; SHA256 values are the portable identity.

## Reproducing the release

Git LFS must be installed and all LFS objects must be hydrated before analysis.
From the repository root:

```bash
git lfs install
git lfs pull
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

The current staged analysis has an explicit interim-cache dependency. A clean
rebuild therefore runs in this order:

```bash
.venv/bin/python scripts/paper_rq_results.py \
  --data-dir data/v2 \
  --out-dir paper_results \
  --seed 20260809 \
  --splits 200 \
  --bootstrap 500

.venv/bin/python scripts/run_paper_analysis.py \
  --config configs/paper_analysis.yaml \
  --stage audit \
  --force

.venv/bin/python scripts/run_paper_analysis.py \
  --config configs/paper_analysis.yaml \
  --stage all \
  --force \
  --jobs 1

.venv/bin/python scripts/run_signature_analysis.py \
  --config configs/paper_analysis.yaml \
  --force

.venv/bin/python -m pytest tests/ -q
```

The audit must record `paper_analysis_ready=true` before scientific stages run.
`--force` replaces derived outputs; omit it when inspecting an existing release.
The full rebuild is CPU- and memory-intensive because it scans millions of
segment rows and performs repeated resampling.

## Output map

```text
data/v2/analysis/paper/
├── 00_audit/                     coverage, joins, quality flags, input hashes
├── 01_family_structure/          RQ1 profiles, similarity, partition test
├── 02_amount_shape/              RQ2 amount, timing shape, sensitivities
├── 03_trace_features/            trace-level matrix and feature dictionary
├── 04_outcome_models/            RQ3 M0–M5 fits and OOF predictions
├── 05_motifs/                    RQ4 enrichment, nulls, transfer diagnostics
├── 06_context/                   RQ5 full/isolated comparisons
├── 07_adaptive_differentiation/  RQ6 distance ratios and scale summaries
├── 08_kline/                     RQ7 exploratory recurrence gate
├── 09_signatures/                RQ8/RQ9 localization and model log
├── tables/                       publication-facing result tables
├── finding_registry.csv          one row per research claim
└── final_analysis_manifest.json  hashes for the final output bundle

docs/paper_analysis/generated/
├── ANALYSIS_REPORT.{md,html}
├── SIGNATURE_ANALYSIS_REPORT.{md,html}
├── REGRESSION_MODEL_LOG.md
├── figures/                      PNG and LFS-tracked PDF figures
└── figure_data/                  exact CSV sidecars for plotted values
```

## What remains unresolved

Two extensions would materially strengthen the evidence rather than merely add
more plots:

1. Independently label the metadata-only context-validation sample with humans
   or a genuinely independent judge to distinguish sensitivity from validity.
2. Run exact label-level circular-shift and within-position permutation nulls if
   motif enrichment is promoted beyond a secondary/exploratory result.

Additional decoding seeds and a matched-channel/matched-budget anchor design are
needed for generation-variance, scaling, or training-lineage claims.
