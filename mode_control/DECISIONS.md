# Decisions

Status: proposed = my default, running with it until you say otherwise; ratified = you agreed.

- D-01 (proposed) Phenotype window. Baseline traces run to 65k tokens; steered generation with HF
  generate cannot. We phenotype everything on the first W reasoning tokens, W = 4096 for the
  pilot (median trace is 7-8k tokens), and refit the frozen score on the same window of the
  baseline traces. Alternative: W = 8192 for the powered run (doubles generation and judge cost).
- D-02 (proposed) Frozen score. Ridge logistic (C = 1, same fixed-L2 family as the paper) on
  standardized amounts, positive = Idea-like. Three variants: cog, conv, all. Out-of-fold on the
  paper's SHA256 prompt folds for the baseline; full-fit coefficients for new traces.
- D-03 (proposed) Strategy prompts go in a system message, four paraphrases per condition
  (`configs/strategy_prompts.json`), with no literal label words. The user task text is unchanged.
- D-04 (proposed) Direction extraction: whitened (Ledoit-Wolf) mean of paired differences from
  teacher-forcing the baseline trace's own first 32 tokens under both prompts. Symmetrization with
  condition-native continuations comes after the prompt-condition generations exist.
- D-05 (proposed) Pulse = 64 residual positions (last prompt position plus the first 63 generated
  tokens) at one layer chosen by Gate 2; sustained steering is a secondary condition.
- D-06 (proposed) Pilot sample: 40 Math (8 per MATH level) + 40 Idea tasks with valid baseline
  labels, one seed; seed 20260903. Powered run: all valid tasks (439 Math, 344 Idea), two seeds.
- D-07 (proposed) Judge: the production judge and prompts unchanged (gemma-4-31B-it, Track A and
  Track B full-context); the isolated pass is skipped for the intervention traces.
- D-08 (proposed) Sampling for steered generation: the paper's shared decode (T 0.6, top-p 0.95,
  top-k 20) with one torch seed per batch shared across conditions (common random numbers).
- D-09 (decided, method fix) Whitening. Full-covariance (Ledoit-Wolf) whitening with 783 traces in
  4096 dimensions is rank deficient (shrinkage 0.01, most eigenvalues at the floor) and threw the
  direction into the null space (domain AUROC 0.50). The default direction is now the plain mean
  paired difference; a PCA-64-subspace-whitened variant (`prompt_pw_r1`) is kept as a robustness
  condition. Files with the `_v2` tag are the corrected ones.
- D-10 (proposed) Steering layer = 19, picked by the pre-registered rule "argmax of the mean
  within-domain Spearman correlation (cog and conv, Math and Idea) over layers 6-26 at the mean-32
  phase". Layers 15-22 are all similar; 19 also maximizes the out-of-fold R² gain. Wrong-layer
  controls at 2 and 29.
- D-11 (note) The condition `rawdom_gm1` in pilot1 was generated with the discarded whitened
  direction (a near-null-space axis at layer 16). It is kept as an extra "null-space direction"
  control and labeled that way in the analysis; the real raw-domain control is `rawdom_r1_gm1`.
- D-12 (proposed) Completion budget for RQ3. With max 4096 new tokens only 42% of Math and 10% of
  Idea pilot traces close their think block, so task outcomes are mostly missing in the pilot. For
  the powered run, generate to a larger budget (8192 or 16384) but keep the phenotype on the first
  4096 reasoning tokens; the pulse ends at token 64, so the extra length only costs decode time.
- D-13 (note, repo change) `src/judge/vllm_engine.py` now passes an `hf_overrides` callable that sets
  `allow_global_per_layer_attribute_access=True` on the HF config. transformers was upgraded to
  5.15 in the `sote` env on 2026-08-13 (after the v2 judge runs) and it refuses vLLM 0.22's global
  `head_dim` read on gemma-4's mixed sliding/global attention config. The flag restores the
  pre-5.15 value (256), which is what worked for the v2 labels. No other behaviour changes.
- D-14 (decided) Common random numbers. pilot1 seeded the sampler with the condition name included,
  so paired conditions did not share a sampling stream. From the next run the torch seed depends only
  on the batch composition; conditions then share the stream until the intervention changes logits.
- D-15 (proposed) Noise band. With one seed and 40 tasks, the discarded null-space direction shifted
  the Math combined score by +0.75 (CI 0.19 to 1.34). Treat paired shifts under about one logit as
  uninterpretable in pilot1; the powered run needs two seeds and the full valid task set.
- D-16 (proposed) Powered run on Qwen3.5-9B: 150 tasks per domain (detects a 10% attenuation of the
  gap), one seed, generation budget 12288 new tokens, phenotype and judge window 8192 reasoning
  tokens (frozen score refit on the same window of the baseline traces), common random numbers.
  Conditions pruned after the pilot gamma readout. Jobs split by condition so they run on
  whichever GPU frees first.
- D-17 (proposed) R1-Distill-Llama-8B as second model, not a replacement. Per trace it is about
  three times cheaper (99% of baseline traces complete, median 1.9k / 1.1k tokens, dense
  architecture), but its cognitive signature is weaker (frozen cognitive score AUROC 0.84 vs
  0.999; conversational 0.97), so it needs about three times the tasks for the same attenuation
  power. It gives measurable task outcomes (RQ3) and a less templated Idea opening (15 distinct
  twelve-token openings vs 1). Its readout stages run now; a steering run follows if the Qwen
  powered run shows an effect.
- D-18 (proposed) Late readout phase. Add activations averaged over generated tokens 65 to 192
  (past the Idea boilerplate) to test RQ1 within Idea; used for readout only unless it wins the
  layer-selection rule.
- D-19 (decided) Do not power the steering arms yet. The pilot1 gamma sweep (64-token pulse, layer 19,
  |gamma| <= 2) moved the phenotype no more than the random controls, while the prompts attenuated the
  gap by 19-34%. Before spending ~30 GPU-hours, run a strong-intervention pilot (pilot2, 40 + 40 tasks,
  4096 window): sustained steering, gamma = -3, a seven-layer band (12-24), additive shifts of one
  full domain gap, a centroid swap, a matched random band, and the raw-domain direction as the
  topic control. The powered run starts now with the design-independent arms only (baseline,
  checking prompt, exploratory prompt; 150 + 150 tasks, 12288 budget, 8192 window).
- D-20 (proposed) R1 direction. The prompt contrast fails Gate 1 in R1, so R1's steering pilot uses
  the supervised readout direction (`readout_r1`), with the raw-domain direction and a matched random
  band as controls. The Qwen strong pilot keeps the prompt direction; `readout_r1` is added to the
  Qwen map for comparison.
- D-21 (proposed) Magnitude scales. Additive steering along the prompt direction is scaled in units
  of the per-layer Idea-minus-Math projection gap (pilot2: ±1 gap). The supervised readout direction
  has no domain gap by construction, so it is scaled in pooled within-domain SDs of its projection
  (pilot2r1: ±3 SD). Random-band controls use the same scale key as the arm they match.
- D-22 (decided) Powered steering arms (power1, 150 + 150 tasks, 12288 budget, 8192 window), all
  sustained: reflect γ = −1 and γ = −3 at layer 19 (dose), additive −1 gap across the band, matched
  random band, centroid swap, additive +0.5 gap (Idea-ward at a dose below the loop threshold), plus
  the already-running baseline and two prompt arms. Split into three two-arm jobs.
- D-23 (decided) Idea-ward dose pilot (pilot3, 40 + 40, 4096 window): band +0.25 and +0.5 gap; layer
  19 alone at +1, +2 (and −2 as the matching Math-ward dose); raw-domain −0.25 gap as a low-dose
  topic control. Purpose: find any Idea-ward dose that changes the signature without loops.
- D-24 (decided) Report paired shifts twice: all traces, and restricted to pairs with no loop flags.
  Reversal claims require the loop-free table.

## D-25 (proposed): equal-dose order-swap run (`sched1`), reviewer design
Arms, all at layer 19 on the prompt direction, additive, every generated position steered, |alpha| = 2 gap
(the largest single-layer dose that stayed coherent in both signs in pilot3: +2 gap moved Math +1.4 with no
loops, -2 gap moved Idea -4.0 with no loops): unsteered; sustained idea (+); sustained math (-); idea->math;
math->idea; random direction + then - (same |alpha|, same switch). With additive steering |delta h_t|^2 =
alpha^2 at every position, so the integrated dose is alpha^2 x positions in every steered arm and the two
ordered arms differ only in sign order. Realized dose (steered_positions, late_positions, switch_at) is
logged per trace.

## D-26 (proposed): switch point
Primary: half of the task's own unsteered generation length (same instance and seed, common random numbers),
clipped to [512, 6144] tokens. Grounding: Math lengths span 1.4k-12k (10th-75th pct) and Idea 4.4k-7.6k, so a
single fixed switch would give short Math traces almost no late phase. Robustness arms: fixed 2048-token
switch for the two ordered schedules only.

## D-27 (proposed): sample and budget
200 tasks per domain (pool with valid baseline labels: 439 Math, 344 Idea), one seed, 12288-token budget,
8192-token judge window, same decode as power1. Second seed only if the primary contrast is borderline.

## D-28 (proposed): pre-registered endpoints
Primary (Idea): originality and feasibility of the final answer (atlas rubric judge). Frontier expansion =
idea->math has feasibility above unsteered AND originality not below unsteered (paired, 95% CI), while
sustained math trades one for the other. Order effect = (idea->math) - (math->idea) on both outcomes.
Primary (Math): exact-match accuracy; hypothesis: math->idea... no, idea->math ("explore early, verify late")
does not lose accuracy vs unsteered and beats sustained idea. Secondary: early-half vs late-half behavior
rates confirm the schedule took hold; loop and completion rates reported per arm.

D-25..D-28 status (2026-09-04 20:00 UTC): running as proposed. Run `sched1`, 8 arms x 200 tasks per domain,
SLURM 351329 (baseline) 351330-351331 (sustained) 351332-351336 (scheduled, gated on baseline) 351337
(answer-quality judge, gated on all). Smoke test `sched_smoke` confirmed the switch: ordered arms share the
early text with the matching sustained arm and diverge after the switch; late-position counts match.
Submission gotcha: sbatch options must precede the script path (a first submission passed --job-name as
the stage argument and was cancelled).

## D-29 (proposed, contingency): half-dose replicate of the order-swap run
Interim (2026-09-05, baseline + sustained-idea judged, 200 tasks/domain): at |alpha| = 2 gap the sustained
Idea-ward arm is coherent on Math (+1.6 Idea-likeness, accuracy -0.125, loops 13%) but degenerates on Idea
tasks over the 12k budget (loops 68% vs 24% unsteered, answers -43 pts); the phase split shows the excess
builds in the late half. If the ordered arms' Idea-task loop rates exceed ~35%, rerun the six core arms at
|alpha| = 1 gap (`sched1_d1`) so the frontier question is answered on coherent text, and report both doses.

D-29 status (2026-09-05): running as `sched1_d1` (six core arms, |alpha| = 1 gap, same tasks, seed and switch rule; fixed-switch arms skipped). Trigger: idea->math Idea-task loops 0.35 vs 0.24 unsteered.

## D-30 (proposed, running): multi-seed sub-run for solution and process diversity (`sched1_div`)
Same five core arms (unsteered, sustained explore, sustained check, explore->check, check->explore) at
|alpha| = 2 gap, 40 tasks per domain (stratified as before, fresh task seed) x 5 sampling seeds, 12k budget.
Rationale: diversity is a per-input property (Kirk et al. 2024 separate per-input from across-input
diversity); one sample per task cannot measure whether a schedule yields the same solution every time or
different ones. Pre-registered metrics per arm:
- Outcome-level (Math): pass@k for k = 1..5 (Chen et al. 2021 unbiased estimator), majority-vote accuracy
  (self-consistency, Wang et al. 2022), number of distinct final answers per task.
- Strategy-level (Math): a judge labels each trace's solution method (short category + one-line summary);
  distinct methods per task and method entropy per arm. The same judge model as the atlas (Gemma-4-31B).
- Lexical (both domains, per input): self-BLEU (Zhu et al. 2018), distinct-n and expectation-adjusted
  distinct (Li et al. 2016; Liu et al. 2022) over reasoning text and over answers.
- Semantic (both domains, per input): mean pairwise cosine distance of embeddings and the Vendi score
  (Friedman & Dieng 2023) on the embedding kernel; embeddings from a sentence-embedding model if one can be
  installed offline, otherwise mean-pooled hidden states of the generation model (reported as such).
- Process-level (both domains): entropy of behavior-label transitions and distinct behavior 3-grams per trace
  from the atlas labels; across-seed variance of the Idea-likeness trajectory.
Across-input versions of the lexical and semantic metrics are also reported on the 200-task single-seed run.

D-28 outcome (2026-09-05): Math endpoint met (explore->check keeps accuracy, +0.10 vs the reverse order;
random schedule null). Idea frontier-expansion endpoint not met (originality +0.09 ns, feasibility -0.22;
random schedule feasibility -0.21). The claim to carry forward is the order effect on Math and the
asymmetric persistence of the exploratory regime, not a creativity frontier gain.

D-28 outcome, revised (2026-09-06): the Math order effect (+0.10, both doses) decomposes as check->explore
below the random schedule by 0.105 at both doses, with explore->check equal to random. The claim to carry:
"late exploration after early verification costs accuracy; the reverse schedule does no harm", not
"explore-early/verify-late helps". The random schedule is the mandatory comparator for any schedule claim.

D-30 outcome (2026-09-06): per-input diversity measured. Explore->check: reasoning-path diversity across
seeds up (self-BLEU -0.028, Vendi +0.042, CIs exclude 0) at unchanged pass@1/pass@5; sustained explore gives
the same diversity at -0.08 pass@1; sustained check reduces diversity. Distinct answers unchanged in all arms.
The claim: "early exploration buys path diversity for free; late verification is what keeps accuracy."

## D-31 (2026-09-06): repertoire distillation deferred
Step 0 (complementarity on sched1_div) found no portfolio gain from schedule-steered samples on Math (mixed
pass@K <= homogeneous, error correlation 0.7-0.8) or Idea (no feasible-coverage gain from explore->check).
Distillation (reviewer Experiment 3) is deferred until a controller produces routes that fail differently.
Next priority: the low-rank signature-subspace controller with leave-one-domain-out extraction (reviewer
Experiment 2), evaluated first by complementarity, not by signature movement.

## D-32 (proposed, running): robust CoT similarity and more seeds
Measure: per task, mean distance over all pairs (the limit of "sample n, repeat N times"), within each arm
and between arms, on three representations (bge-m3 embedding of the reasoning; behavior transition
profile, Jensen-Shannon; token 3-gram Jaccard); excess = between - mean within, paired over tasks, with a
within-task permutation null and a leave-one-task-out classifier AUROC. Reported for all traces and for
correct traces only. To give the correct-only cells enough pairs, `sched1_div2` adds seeds 6-10 for
unsteered, explore->check and check->explore on the same 40 Math and 40 Idea tasks (10 samples per task).

D-32 outcome (2026-09-08): measure adopted (all-pairs within/between excess with permutation null +
leave-one-task-out AUROC, correct-only). Result: successful explore->check traces are only weakly
distinguishable from successful unsteered traces (AUROC 0.55-0.58; process organisation identical);
sustained explore is distinguishable (0.67) at an accuracy cost. Together with D-31, the rank-one
schedule does not create a new region of successful reasoning.

## D-33 (proposed): low-rank signature control subspace (reviewer Experiment 2)
Model: stay on Qwen3.5-9B. Grounding: 4B's atlas traces average 27k tokens (9B: 17k), 67% complete, and
only 32% of its Idea traces were labellable, so per trace it is no faster and its labels are thinner; every
existing direction, phenotype and result is 9B.
Stage 1, corpus: unsteered, 4 samples per prompt, 200 prompts per domain, four domains (math, idea, gpqa,
planning; the two with the highest label coverage and verifiable answers besides our two), 12k budget,
8k judge window. Reuses the existing 10-seed Math / 5-seed Idea baselines on 40 tasks.
Stage 2, signature and activations: teacher-force each trace; residual stream at every judge-segment
boundary for layers 8-24 step 2; target phi_future(t) = behavior rates over the remaining segments (8)
plus over the next 4 segments (8); phi_sofar(t) for the closed-loop estimate.
Stage 3, fit: per layer, reduced-rank ridge h_t -> phi_future(t), rank in {1, 2, 4, 8}, leave-one-domain-out;
choose layer and rank by held-out R^2 per component and a pilot causal check, not by variance explained.
Stage 4, control: minimum-norm additive correction toward a target signature at every generated token,
delta h = A^T (A A^T + lambda I)^-1 (phi* - A h - b), gain-scaled; targets = k-medoid archetypes (k = 4 per
domain, whitened phi over high-quality traces) and, as a sanity check, the two rank-one endpoints.
Controls: norm-matched random subspace of the same rank; the rank-one prompt direction at the same norm.
Stage 5, evaluation, pre-registered: gap closure per component; selectivity (intended vs off-target
movement); naturality (loops, completion); utility (accuracy, originality, feasibility); complementarity
(mixed pass@K, unique solves) as the decision metric; generality = fit on three domains, steer the fourth.
Budget: corpus 2-3 GPU-days, activations + fit 1 day, steering pilots 2 days.

## D-34 (proposed, running): signature-controller pilot
Fit findings (corpus1, 4 domains): the residual state at a segment boundary predicts the domain-relative
future signature with within-domain R^2 0.08 (correlation 0.27; rank 16 0.085, rank 8 0.080, rank 2 0.06),
best for perspective shift, conflict and backtracking, nil for backward chaining; every layer 10-22 is
equivalent (16 marginally best); transfer to a held-out domain is correlation 0.06, i.e. the readout is
domain-specific. Note the pooled-fold R^2 of 0.18 reported first was inflated by domain identity.
Pilot map: layer 16, rank 8, lambda 1e5, domain-demeaned targets, fit on all four domains. Pilot runs
sigpilot_math and sigpilot_idea: 40 tasks per domain held out from the corpus prompts, one seed, 12k
budget; conditions = unsteered, two archetype targets per domain (the medoids with the largest and smallest
late-verification rise: math_A4/math_A1, idea_A3/idea_A4), gains 1 and 2, and a norm-matched random
subspace at each gain. Evaluation: sigeval (gap closure, selectivity), loops, accuracy/quality, then
complementarity if any arm moves the signature.

D-34 outcome, Idea (2026-09-09): under-dosed; gap closure negative and equal to the random subspace.
## D-35 (proposed): second controller pilot with calibrated dose
Controller ridge set relative to the readout scale (rel_lam = 1e-3 x mean eigenvalue of W^T W) so the
per-token correction closes the readout gap; per-token cap |delta h| <= 0.25 |h|; gains chosen so the
median correction is ~5-15% of the residual norm. Added an open-loop variant: a constant direction per
trace, G (phi_archetype - phi_domain_mean), scaled to a fixed norm of 7 units (the rank-one dose), which
makes the comparison with the earlier steering a pure change of direction. Random-subspace controls for
both variants at each gain. Same tasks, targets and evaluation as D-34.

D-35 outcome, Idea (2026-09-10): dose fixed (open-loop at 4.5-6.75 units matches rank-one strength), but
the induced shift vectors are collinear with the rank-one axis (cosine 0.73-0.87) and the orthogonal
component is not target-specific. Replaced gap closure with fraction-of-gap-covered along the target
direction (medoid targets are too close to the domain mean for closure to be meaningful). Await Math.

D-35 outcome, Math (2026-09-10): the A2 archetype (subgoal up, verification/Q&A down) is off-axis and the
calibrated controller covers 0.4-0.6 of its gap with target-specific components; accuracy +0.225 vs
unsteered and +0.20 (0.075, 0.35) vs the matched random subspace at gain 2 closed-loop, through more
traces finishing within budget (conditional accuracy unchanged). Idea: no off-axis control.
## D-36 (proposed, running): powered replication of the Math A2 controller
sigpilot3_math: 200 held-out Math tasks (excluding corpus and pilot tasks), arms = unsteered,
closed-loop A2 gain 2, open-loop A2 gain 3, matched random subspaces for both, and the rank-one sustained
check arm (|alpha| = 2 gaps at layer 19) as the axis comparator. Pre-registered: primary = accuracy of the
closed-loop A2 arm minus its random control (paired); secondary = answered within budget, subgoal and
verification shifts, conditional accuracy, loops; and complementarity of A2 traces vs unsteered (unique
solves) on the same tasks. Jobs 353184 / 353185 (two condition groups) -> 353186 (sigeval).

D-36 outcome (2026-09-10): replicated. Closed-loop A2 +0.216 (0.141, 0.296) vs unsteered, +0.312 vs the
matched random subspace, +0.161 vs the rank-one check arm; behavior shifts as pre-registered. The claim:
"a low-rank readout of the reasoning signature yields a controllable direction (subgoal up, verification
and Q&A down) that the explore/check axis does not contain, and steering along it raises MATH pass@1 at a
fixed budget by 21 points, mainly by finishing more traces, with conditional accuracy up 5."
## D-37 (proposed, running): pass@K and complementarity of the A2 controller
Generate the closed-loop A2 arm (gain 2) on the sched1_div Math tasks (40 tasks x 5 seeds, same seeds),
grading only, and pool with sched1_div's unsteered samples: pass@k, mixed-portfolio pass@K vs homogeneous,
per-input diversity, and the between/within similarity of correct traces. Decides whether the A2 regime
is a different way of solving or a faster way of reaching the same solution.

D-37 outcome (2026-09-11): A2 dominates unsteered (pass@1 +0.19 at 5 seeds, pass@K higher at every K) but
is not complementary (mixed pool < all-A2; r = 0.77). Repertoire/distillation framing stays deferred; the
claim is a controllable efficiency direction, not a new solver.

## D-38 (proposed, running): Math thinking applied to Idea tasks
Question: does the Math A2 direction (subgoals up, verification/Q&A down) do anything on Idea tasks, and
what does it do to originality/feasibility? Two applications on 200 held-out Idea tasks (one seed): (a)
the open-loop vector exactly as used on Math (reference mean = Math, norm 6.75 units, gain 3); (b) the
closed-loop controller driving the Idea readout toward the absolute A2 profile relative to the Idea mean
(gain 2, cap 0.25). Matched random subspaces for both. Endpoints: signature movement (subgoal,
verification, Q&A, shift, conflict), loops/answered/length, originality and feasibility (paired), and
the same-vector arm's shift vector compared with the Math result.

## D-39 (proposed, running): the fourth cell, Idea directions on Math tasks
Same 200 held-out Math tasks as the A2 replication (task seed 20260911, same exclusions). Arms: the
open-loop vectors of idea_A3 (high shift) and idea_A4 (low shift) exactly as applied on Idea (reference
mean = Idea, norm 6.75, gain 3); closed-loop toward the absolute idea_A3 profile (gain 2, cap 0.25);
matched random subspaces. Endpoints as D-36 (accuracy, answered, loops, behaviour shifts). With D-36,
D-35 (Idea on Idea) and D-38 (Math on Idea) this completes the 2x2 of direction x task domain, all with
the same readout, dose calibration and controls.

D-38 / D-39 outcome (2026-09-12): 2x2 complete. Math A2 on Idea reproduces its organisation (cos 0.80) but
costs originality (-0.25 / -0.79) with no feasibility gain; Idea A4 (low shift) on Math gives +0.11 accuracy
(+0.20 vs random) by inducing the same subgoal-up / verification-down change as A2; Idea A3 on Math is
harmful and its absolute profile collapses Math. Claim: the beneficial Math direction is a property of the
Math task, reachable from either domain's archetypes; no direction improves Idea answers.

## D-40 (proposed): fixed-vector steering in vLLM
Open-loop A2 matched closed-loop A2 on the 199 held-out Math tasks (0.623 vs 0.628), so A2 can be one constant
vector. A constant edit does not need HF generate. `mc/vsteer.py` adds the vector with a forward hook on the vLLM
decoder layer (eager mode, no torch.compile), which keeps continuous batching. HF generate at a 32k budget would
take about 45 minutes per batch of 32, because every batch runs to its longest trace.
Gate before any run (`vprep` stage): teacher-forced log-probs in HF (original Steerer) and vLLM must agree on the
per-token steering effect (correlation > 0.9, norm ratio 0.8-1.25), and the SEAL position rule must steer exactly
the delimiter tokens fed back during decoding. Built-in replication: at the 12k truncation, vLLM unsteered and A2
should land near the HF numbers (0.41 and 0.62).
Sampling seeds depend only on (run, task, seed), so conditions share sampling streams (as D-14). Jobs run at
`ashton_low` (lab priority policy, 2026-09-23), write finished traces every 4 minutes, and resume from the done set
after a requeue.

## D-41 (proposed): P1, is A2 an efficiency effect, and is it SEAL?
Run `p1_budget`: the 199 held-out Math tasks of D-36, 2 seeds, 32k budget. Accuracy at 4k, 8k, 12k, 16k, 24k and 32k
is read by truncating each trace (a trace cut at B tokens is what a B-token run gives with the same sampling stream).
Arms:
- unsteered
- A2 open-loop, the D-36 vector (layer 16, norm 6.75, all positions)
- random direction, same layer and norm
- SEAL as published (Chen et al. 2025, arXiv 2504.07986, and their released code): mean residual at "\n\n" tokens
  that precede reflection or transition thoughts minus the mean at those that precede execution thoughts (their
  keyword rules), from our 800 unsteered corpus Math traces; added with coefficient -1 at delimiter tokens during
  decoding. Layer picked on a validation split (below).
- the SEAL direction at A2's layer and dose (all positions, norm 6.75), to separate direction from dose
- concise instruction: "Be concise." as the system message
SEAL layer: the paper uses layer 20 of 28 (7B) and 55 of 64 (32B) and picks it by ablation. We try the inputs to
layers 17, 20, 23 and 26 at coefficient -1 on the 101 Math tasks that are in neither the corpus nor the evaluation
set (run `p1_sealval`, one seed, 12k budget), and keep the highest accuracy (ties: fewer tokens).
Reading, written before the run: if A2's gain over unsteered goes to about zero by 32k, A2 is an efficiency result.
If it stays, it is more than budget. If SEAL or the SEAL direction at A2's dose matches A2 at equal tokens, A2 adds
nothing over SEAL; if A2 beats both, the signature readout found a better direction. Track B on seed 1 compares the
behaviour shifts (does SEAL also raise subgoal setting?).
Deviations from SEAL: they stop steering after </think>; our hook has no request identity, so delimiters in the
final answer are steered too. Their vector used 1000 MATH-train traces; ours uses 800 traces on 200 MATH-test
problems, disjoint from the evaluation tasks.

## D-42 (proposed): P2, A2 and each domain's own archetype on GPQA and planning
Step 1, offline (`mc/success_dir.py`): in each domain's unsteered corpus (200 tasks x 4 seeds), which behaviours
separate a successful sample from an unsuccessful sample of the same problem (task fixed effects), for correct,
answered, and correct among answered. A2's measured Math shift pushed through each domain's within-task regression
gives a predicted accuracy change per domain, written down before the steering run. The same rule picks each
domain's own archetype (largest predicted gain).
Step 2 (`p2_gpqa`, `p2_planning`): 200 held-out tasks per domain (not in the corpus), 2 seeds, 32k budget, read at
12k and 32k. Arms: unsteered; A2, the same vector as on Math; random direction; the chosen own-domain archetype
(open-loop, relative to that domain's mean, norm 6.75).
What it separates: at 12k GPQA answers 41% of tasks (Math 65%, planning 78%), and planning has the lowest subgoal
rate (0.035 vs Math 0.11). If A2 is an efficiency direction, GPQA gains through answering. If it works through
subgoal structure, planning gains in accuracy among answered traces.

D-42 step 1 outcome (2026-09-23, before any P2 steering; data/sigctrl/success_dir.csv, success_pred.csv):
Within-task correlates of correctness (tasks whose outcome varies over the 4 seeds; 95% task-bootstrap CI):
- Math (119 tasks): subgoal +0.38, perspective shift +0.29, verification +0.26; backtracking -0.10 (ns).
- GPQA (57 tasks): backtracking -0.33, conflict -0.23, Q&A -0.21; verification +0.29, subgoal +0.23.
- Planning (47 tasks): conflict -0.39, backtracking -0.19, backward chaining -0.19; verification +0.27,
  reconciliation +0.17; subgoal +0.05 (ns).
Check on known data: A2's measured Math shift through the Math regression predicts accuracy +0.17 (0.10, 0.27) and
answered +0.25; D-36 observed +0.21 and +0.26. The rank-one check shift predicts -0.02; observed +0.055 (ns).
Predictions for step 2, written now (accuracy change, same vector and shift as on Math):
- GPQA: A2 +0.35 (0.07, 0.60); own archetype gpqa_A4 +0.58 (0.39, 0.76).
- Planning: A2 +0.10 (-0.33, 0.58), i.e. no confident prediction; own archetype planning_A2 +0.23 (0.14, 0.36).
The linear predictions assume the vector moves each domain's behaviours as it moved Math's; take the sign and
ordering, not the size. My earlier guess that planning would gain through subgoals is not supported: subgoal does not
separate planning successes from failures. Own archetypes chosen by the rule: gpqa_A4 (more verification and
subgoals, less Q&A, conflict and backtracking; vector cosine with A2 0.79) and planning_A2 (less conflict,
perspective shift and backtracking; cosine with A2 0.42).

D-40 gate outcome (2026-09-23, jobs 362599 and 362602; data/vsteer/check_unscaled.json, check_seal_x8.json):
4 corpus Math traces, prompt + first 1024 reasoning tokens, teacher-forced log-probs in HF and vLLM.
- Unsteered: mean |HF - vLLM| 0.005 per token (bf16 engine noise).
- A2 (all positions, layer 16): per-token steering effect correlation 0.998, norm ratio 1.00. Passes.
- SEAL at its real size (layer 22 output, norm 12.7, 54 delimiter positions): correlation 0.73, norm ratio 1.05.
  The effect (mean 0.005) is the same size as engine noise, so the test has no power. With the vector scaled 8x:
  correlation 0.986, norm ratio 1.00. Passes; the hook edits the right layer and positions.
- SEAL decode rule: steered events equal the delimiter tokens fed back (34 of 34, 36 of 36). Passes.
SEAL vectors (800 corpus Math traces; 3,129 reflection, 74 transition, 28,247 execution thoughts): |S| at the output
of layers 16, 19, 22, 25 = 3.2, 6.5, 12.7, 16.9. Cosine of the SEAL direction with A2 at layer 16: 0.09, i.e. A2 is
not SEAL's direction.

D-41 SEAL layer outcome (2026-09-23, run p1_sealval, job 362604; 101 Math tasks outside the corpus and the
evaluation set, one seed, 12k budget, paired against unsteered):
| SEAL layer (input) | accuracy at 12k | change (95% CI) | answered | correct among answered | tokens |
|---|---|---|---|---|---|
| unsteered | 0.426 | | 0.58 | 0.73 | 8055 |
| 17 | 0.465 | +0.04 (-0.05, 0.13) | 0.62 | 0.75 | 7820 |
| 20 | 0.505 | +0.08 (-0.01, 0.18) | 0.71 | 0.71 | 6973 |
| 23 | 0.485 | +0.06 (-0.03, 0.16) | 0.62 | 0.78 | 7351 |
| 26 | 0.564 | +0.14 (0.05, 0.23) | 0.68 | 0.83 | 7088 |
Chosen by the rule: layer 26 (output of layer 25). The best layer is at the edge of the tested range; the paper's
32B setting (55 of 64) is at the same relative depth. SEAL is a strong baseline here, and unlike A2 its gain is
mostly accuracy among answered traces. p1_budget launched with seal_L26.

D-41 / D-42 outcome (2026-09-24; RESULTS.md "P1 and P2"): A2 is an efficiency direction. Its gain shrinks with budget
(Math +0.24 at 12k, +0.09 at 32k from non-finishing unsteered traces; GPQA +0.23 to -0.04; planning +0.16 to +0.02),
with accuracy among answered traces unchanged. It is not SEAL (published SEAL +0.03 ns; SEAL direction at A2's dose
+0.13; A2 minus that +0.11 at 12k). Each domain's own archetype beats the Math vector on its own domain (GPQA +0.36,
planning +0.19 at 12k), and the pre-registered within-task predictions got every sign and ordering right. The claim
to carry: signature-derived steering gives the same accuracy for a third to a half of the tokens across Math, GPQA
and planning, beyond SEAL and a concise prompt; it does not raise accuracy when budget is not binding.

## D-43 (proposed, running): behaviour check for P2
Track B on seed 1 of the GPQA and planning runs, for unsteered, A2 and the own-domain archetype (200 traces each;
random skipped). Questions: does the Math A2 vector move GPQA and planning behaviours the way it moves Math (shift
cosine), and does the D-42 linear model predict the observed accuracy change when fed each domain's observed shift
instead of the Math shift. Jobs: GPQA first (`judgeB p2_gpqa`), planning after it (one slot each, in turn).

## D-44 (proposed, running): P3, steering the solution method
Harsh's point: for Math, different kinds of solutions may be worth more than one solution. So far the behaviour modes
change how the model reasons, not which method it uses (7 of 40 tasks show two or more method categories over five
seeds; every schedule arm 10-11 of 40). Step 1 asks whether an instruction can change the method at all, and whether
that adds solves. Run `p3_strat`: the 199 held-out Math tasks, 4 seeds, 16k budget, four arms: unsteered and three
general heuristics from Polya (How to Solve It, 1945) as the system message:
- cases: "Before solving it in general, try small cases or specific values, look for a pattern, and then confirm it."
- backward: "Work backwards: start from what the question asks for and reason back to what is given."
- figure: "Represent the problem as a picture (a diagram, a number line, or coordinates) and reason from that picture."
Measures: method mix per arm and methods per task when arms are pooled (method judge, D-30); accuracy; the D-31
complementarity test (unique solves, mixed-pool pass@K vs a homogeneous pool of the same size, error correlation).
Gate for step 2: an arm that changes the method on a sizeable share of tasks and adds solves in a mixed pool.
Step 2 then extracts that prompt's direction the way the first direction was built (same tokens teacher-forced with
and without the instruction, difference of layer-16 states) and steers without the prompt.

## D-45 (proposed): P4a, train only a steering bias with RL and compare it with A2
Question: is A2 what RL would learn? The model stays frozen; one vector b at the output of layer 16 (A2's placement)
is trained with RL on Math. Rollouts in vLLM with b applied, gradient in HF with b as the only parameter
(`mc/rlbias.py`). Reward 1 if correct within an 8k budget (where A2's gain is largest). Dr. GRPO loss (Liu et al.
2025): group-mean advantage, no std scaling, token log-probs divided by a constant, so the loss itself does not favour
short or long answers; only the budget does. 16 prompts x 8 samples per step, Adam lr 2e-3, 60 steps (about 3 passes
over 301 train problems: the MATH-500 tasks outside the 199 evaluation tasks). Recent work fits test-time RL into
bias-only vectors (arXiv 2609.18587), so a bias should have enough capacity.
Written before the run: if RL finds A2's behaviour, the learned b cuts tokens and loops, raises subgoal setting and
lowers verification and Q&A (behaviour-shift cosine with A2 high); its raw cosine with A2 can still be low, since
SEAL's direction reached A2's behaviour at cosine 0.09. Evaluation: export b at steps 20/40/60 and run it on the 199
held-out tasks with the P1 settings (same seeds, so P1's unsteered and A2 arms are the comparison), then Track B.
Next runs, each only after the previous one is read: b initialised at A2 (does RL keep, grow or turn it?), and a
generous 32k budget (prediction: little pressure toward efficiency).

D-45 correction (2026-09-26): the RL runs zero8k and a2init8k computed token log-probs at temperature 1 while rollouts
were sampled at temperature 0.6 (top-p 0.95, top-k 20). TRL and verl divide logits by the sampling temperature; without
it the update is not the policy gradient of the sampled policy. Symptom that exposed it: started at A2, reward fell
steadily (0.65 at steps 1-10, 0.41 at steps 51-60) while b drifted in the same direction the zero-start run took (cosine
0.83 between b60 - A2 and the zero-start b60). Both runs and the p4_eval numbers from zero8k are withdrawn as evidence.
Fixed in mc/rlbias.py (logits / 0.6; config records logprob_temperature). Rerun order: a2init8k_T first as a sanity
check (with a correct gradient, RL started at A2 should not steadily lose reward), then zero8k_T.

D-45 outcome (2026-09-27): the pre-registered test was the behaviour level, and it is met. Bias-only RL on a budgeted
correctness reward, started from zero, learns a vector that moves behaviour in A2's direction (cosine 0.94: more subgoal
setting, fewer Q&A turns; verification mostly kept), removes loops, halves tokens at 32k and nearly matches A2's
accuracy, while being orthogonal to A2 in the residual stream and a third of its norm. Started from A2, RL keeps A2 and
strengthens its subgoal component. Reading: the efficient reasoning profile the signature readout found is also what
RL finds when the budget binds, and several unrelated residual directions reach it (A2, SEAL's direction, the RL bias).
What RL finds is A2's behaviour, not A2's vector.

## D-46 (proposed, running): re-grade math with the repo's validated answer extraction
Found 2026-09-27 while pulling examples for the story page: the direct grader used by mc.analyze, mc.budget and
mc.complementarity (repo `grade_math_answer` on the raw answer, bold stripped) misreads many correct answers. It takes
the last math span in the answer, which is often not the answer (e.g. "the area of $\triangle ACD$ is 28" -> reads
"\triangle ACD"; "h + k + a + b = ... = 16" -> reads "h + k + a + b"), and it rejected "\frac{5}{9}" against the
reference "\frac 59". In a random sample of 8 flagged baseline answers, all 8 were correct answers graded wrong. A loose
flag (reference string appears near the end of an answer graded wrong) fires on 21-30% of answered math traces, and the
rate differs by arm (A2 21%, unsteered 26%, random 30%), so arm contrasts can be biased, not only levels. The RL runs of
D-45 used this grader as their reward, so RL may partly have learned grader-friendly formatting.
The main atlas pipeline does not have this problem: it runs the reference-blind LLM extractor (RUNBOOK section 4) and
grades the extracted answer (section 5). mode_control skipped that step.
Fix, following the RUNBOOK: run src.judge.run_answer_extraction (gemma-4-31B-it, configs/exp.yaml) over every trace in
p1_budget, p1_sealval, p2_gpqa, p2_planning, p3_strat, p3_steer, p4_eval2, p4_eval3 and sigpilot3_math (14,847 traces,
staged at data/extract_in; jobs mcextract0/1), then src.perf.grade, then recompute every accuracy, budget curve, pool and
RL comparison from those grades. Budget truncation: a trace that stopped within the budget takes its extracted grade; a
cut trace counts as unanswered unless its thinking closed before the cut. The story page waits for this.

D-46 finding (2026-09-27): the repo's answer extractor is itself broken, for the atlas as well as mode_control. Over the
14,847 mode_control traces it returned status "no_answer" for 11,639 of 11,647 math traces (8,725 of which contain an
answer) and for almost all GPQA and planning traces; the canonical atlas file data/v2/judge/answer_extractions__gemma
shows the same (3,958 of about 4,000 math rows "no_answer"). src.perf.grade therefore fell back to the deterministic
parser for nearly every row, in the atlas too, and that parser reads markdown bold "**" as the answer.
Diagnosis (jobs xdiag 366743, 366771; 16 + 24 traces): without the JSON grammar, gemma-4-31B extracts every answer
correctly (e.g. "45", "15x - 80", "-2 + 7i") but writes the answer first and calls its status "success". The schema forces
the "status" field first, with enum ok/no_answer/ambiguous, so the model must label before it has found the answer and
picks "no_answer". It is not the doubled BOS (single-BOS token input gives the same result). Moving "status" to the end
fixes 8 of 12 answered math traces; the rest fail the schema (missing fields), and on GPQA the model extracts the right
letter but still labels it "no_answer".

## D-47 (proposed, needs Harsh): how to grade
Options:
(a) Repair the repo extractor for everyone: status last, clearer status names ("answered" / "no_answer" / "ambiguous"),
    shorter evidence, check on a hand-labelled sample, then rerun extraction for the atlas and mode_control. Right fix;
    changes the main paper's grading pipeline.
(b) Keep the extractor prompt and its verbatim-evidence validation, but drop the JSON grammar (free text, parse the JSON,
    map "success" to "ok"). In the diagnostic this extracted 16 of 16 answered math traces correctly. Smallest change to
    the validated contract; still needs a hand-checked sample.
(c) mode_control only: a stricter deterministic reader ("Answer:" / "Final Answer" line, last boxed, last bold value, then
    math_verify), checked against (b) on a sample. Fast, but bespoke.
Default proposal: (b) now for mode_control, checked on 100 hand-labelled traces, and (a) for the atlas as a separate
decision. The story page waits for this.

D-47 ratified (2026-09-27, Harsh: "free-text extraction"): mc/extract_free.py keeps the repo prompt (build_prompt) and
row format but generates without the JSON grammar and normalises the reply. The repo's verbatim-evidence rule rejected
about half of correct extractions (gemma paraphrases its evidence quote, or quotes the reasoning when the final answer
is a bare letter), so grading uses a grounding rule instead: the extracted answer itself must appear in the final answer
section (after </think>), as text or as an equivalent expression among the repo parser's candidates; MCQ letters must
be the choice read from the final answer or stand alone in its last 200 characters. Unfinished generations grade 0.
Hand check (100 traces; 40 math where the direct grader and the extraction disagree, 30 math where they agree, 10
unanswered math, 20 MCQ; mode_control/data/extract_free/check_sample.parquet): the extracted answer matched what the
model committed to in all 90 answered traces, and all 10 unfinished traces graded 0. Grading errors: 5 of 70 answered
math traces were correct answers graded wrong because of reference formats the repo matcher cannot read (\dfrac{17}{50}
vs 17/50; "x \in [-2,7]" vs [-2, 7] twice; \frac9{19} vs 9/19; \$32,\!348 vs 32,348). Fixed by a formatting-only
normalisation of both sides (norm_math / math_correct), unit-tested on those five plus near-miss wrong answers, which
stay wrong. Of the 40 disagreements, 39 were correct answers the direct grader missed and 1 was the \dfrac case.
The MCQ grader agrees with the extraction on every checked trace, so GPQA and planning results stand as they were.

D-47 outcome (2026-09-28): re-grade done for p1_budget, p1_sealval, p2_gpqa, p2_planning, p3_strat, p3_steer, p4_eval2,
p4_eval3, sigpilot3_math (numbers in RESULTS.md, "Re-graded math results (D-47)"). What changed in the conclusions:
1. A2 is not "equally accurate when both answer". At 32k it is right on 93% of the problems it answers, against 97%
   unsteered; it answers 99% vs 88%. The efficiency claim stands (A2 at a 16k cap, 87%, matches unsteered at 32k, 85%,
   with 6.0k tokens vs 13.2k).
2. RL from zero is the most accurate arm at 32k (95%, +3.5 over A2, CI 1 to 6). It does not cut verification (-0.4, CI
   -1.3 to 0.4; A2 -3.3), and it is right on 97% of what it answers.
3. P3 complementarity does not survive: mixed pool +1.7 (-0.6 to +4.1). "Work backwards" helps (+9, 6 to 13) mostly by
   finishing more often, not by a new method.
4. D-36 replication re-graded: unsteered 57%, A2 81%, rank-one check 63%, random 46%.
Also corrected (not a grading issue): the "off the usual method" share picked each task's usual method with a polars
group_by().first() after a sort, which does not keep the sort order, so ties and even clear modes were picked
arbitrarily. With a tie-aware rule (every tied mode counts as usual): unprompted 5%, "work backwards" 14%, "try small
cases" 18%, "draw a picture" 29% (was reported as 7/17/21/31%); steering: unsteered 5%, picture direction x2 9%, x4 13%,
random 8%. Qualitative reading unchanged.
Still open: re-grade of the older rank-one runs (power1, sched1, sched1_d1, pilot3, sigx_math; 5,624 math traces staged
in data/extract_in2), job 367279 held for the two-job cap. Until then the "exploring costs math 13 points" claim is
stated without a size. The RL training rewards used the old grader (trend only).

## D-48 (ratified 2026-10-06): paper frame and order of work
Harsh: "all of this sounds good, go ahead and implement. we will do audit during paper writing. we can schedule these
jobs with LOW priority." The paper is built on three findings:
- F1. Reasoning models deliberate differently by task demand, and the state that sets this is readable before the
  first thinking token. New test: minimal pairs (D-50).
- F2. Each domain's signature is tuned to its own demand. Porting it into another domain moves behaviour, and the
  target domain's own behaviour-outcome slopes predict what it does to the outcome (D-52), including the Society of
  Thought crossover (D-51).
- F3. The behaviour profile, not the residual vector, is the unit of control, and what it buys is efficiency (D-53).
Order: blockers (push mode_control to GitHub, re-grade the old runs, re-grade the atlas), then F1, Countdown,
transplant matrix, then the second model (D-54). The human audit of the behaviour labels moves to paper writing.
All jobs from here run at --qos=ashton_low (Harsh, 2026-10-06), so every stage must checkpoint well inside 15 minutes:
vsteer flushes every 180 s, extraction writes every 128 traces, Track B uses exp_mode_chunk20.yaml.
Blocker status 2026-10-06: mode_control pushed as branch `mode-control` (95690cc; code, decisions, results, figures,
story sources, archived run builders; data stays on ada). The ada checkout is NOT pulled: GitHub main tracks data/v2 as
LFS pointers and ada has no git-lfs, so a pull would replace the real parquets. Old-run re-grade resubmitted (freex2,
jobs 378510/378511).

## D-49 (ratified with D-48): re-grade the atlas with the free-text extraction
This settles D-47 option (a) for the main pipeline by reusing the validated mode_control extraction rather than
changing src/judge/run_answer_extraction.py. `mc/atlas_regrade.py stage` copies the math, gpqa, planning and security
traces of all 8 models (16,000) to data/v2/extract_free_atlas/in; `mode.sbatch freexa <shard> 6` extracts (jobs
378517-378522); `atlas_regrade grade` writes data/v2/perf/success_grades_freex.parquet next to the canonical file (never
over it) and a model x domain comparison. The paper analysis switches to the new file only after the comparison is read.
The D-47 hand check was on Qwen3.5-9B traces only; before the switch I check 10 changed grades per model family.

## D-50 (proposed, ratified in outline by D-48): F1 minimal pairs, demand vs topic vs format
Question: does a model's deliberation follow what the task asks for (solve it, generate options, judge an answer), or
the topic, or the answer format? CoT Encyclopedia (arXiv 2505.10185) finds format matters far more than domain, and
Sahoo et al. (arXiv 2606.02907) find reasoning-type probes detect format. Our within-format separation (MCQ trio 68-85%
vs chance 33%) suggests more than format, but topic and demand are still confounded with domain.
Design. Content: 80 MATH-500 problems (16 per level) and 80 GPQA questions, seed 20261006. Four prompts per item:
- solve_free: math as in the atlas; GPQA stem without options, "end with your final answer".
- solve_mcq: GPQA as in the atlas; math stem with four options (the reference plus three wrong final answers other atlas
  models gave for that problem; numeric perturbations of the reference if fewer than three exist), same MCQ instruction.
- generate: "Do not solve this problem. List several genuinely different ways someone could approach it, and say
  briefly what each would involve." (no options shown)
- evaluate: the problem (no options) and a proposed final answer, correct for half the items and a wrong one for the
  other half (fixed by item hash); "Decide whether the proposed answer is correct ... end with Correct or Incorrect."
So within an item, solve_free vs solve_mcq changes only the format, solve_free vs generate/evaluate changes only the
demand, and math vs GPQA at the same prompt type changes the topic.
Models: Qwen3.5-9B, Gemma-4-12B, R1-Distill-Llama-8B; one sample, 32k budget, atlas decoding. Track B on the first 8,192
reasoning tokens (exp_mode_chunk20.yaml, full context only), same as every mode_control run.
Analyses, fixed now: (1) per behaviour and for the 8-rate profile, the share of between-trace variance explained by
demand, format and topic (item as a blocking factor); (2) paired distances within item: format change vs demand change
vs topic change, in the atlas log-ratio metric; (3) the model's own atlas domain classifier (8 domains, retrained on
atlas labels cut to the same 8,192-token window) applied to each prompt type; (4) Qwen and R1: residual stream at the
last prompt token, decode demand trained on math items and tested on GPQA items, and topic trained on solve prompts and
tested on generate prompts, per layer.
Predictions: demand changes move the profile more than format changes in all three models; generate prompts on math
content are classified as idea or moral more often than as math; demand is decodable across topics from early layers.
If format moves the profile as much as demand, F1 is restated as a format result and the paper says so.

## D-51 (proposed, ratified in outline by D-48): Society of Thought crossover on Countdown
Kim et al. (arXiv 2601.10825) steer toward more conversational behaviour (Q&A, perspective shift, conflict,
reconciliation) and double Countdown accuracy. Our gains on Math, GPQA and planning come from fewer Q&A and conflict
turns. Prediction: a crossover. Conversational-up helps Countdown and hurts Math at tight budgets; A2 (structure) helps
Math and helps Countdown less or hurts it.
Design: 200 four-number Countdown tasks (Jiayi-Pan/Countdown-Tasks-3to4, seed 20261006), TinyZero-style prompt with the
equation in <answer> tags, deterministic grader (each number once, value equals target). Qwen3.5-9B, vLLM fixed vectors
at layer 16, norm 6.75 (A2's), 2 seeds, 32k budget read at 4k-32k by truncation. Arms on Countdown: unsteered, A2,
conv_up, conv_down, random. Arms on Math (the 199 p1_budget tasks and seeds, so unsteered/A2/random are shared):
conv_up, conv_down. conv_up = the A2 recipe with the target "+1 within-domain SD on the four conversational rates",
same map (map_corpus1_dm8_L16_r8). Manipulation check: Track B on seed 1 of every arm.
Pre-registered contrast: (conv_up - A2) on Countdown minus (conv_up - A2) on Math, at 12k, > 0.

## D-52 (proposed, ratified in outline by D-48): the transplant matrix
Domains: math, gpqa, planning (converge), idea (generate), moral (judge). Code dropped: 175 tasks cannot supply a
readout corpus and 200 held-out tasks, and its grading needs sandboxed execution.
Signature vector of a domain S: its mean behaviour profile minus the mean of the five domain profiles (the 16 readout
targets: future and next-4 rates over the first 8k tokens), pushed through the same readout map as A2
(map_corpus1_dm8_L16_r8, rel_lam 1e-3) and scaled to norm 6.75. Moral is not in the readout corpus; its profile is
computed the same way from the atlas Qwen3.5-9B moral traces (300 tasks; the other 200 are held out). The map stays the
one A2 came from, so every vector in the paper lives in one readout.
Targets and tasks: math = the 199 p1_budget tasks; gpqa and planning = the p2 tasks (same seeds, so their unsteered,
A2, own-archetype and random arms are reused); moral = 200 held-out atlas tasks; idea = 200 LiveIdeaBench keywords
never used before (the 500 atlas keywords are spent on the corpus and pilots). 2 seeds, 32k budget, 8k Track B window.
Arms per target: the five signature vectors (own = amplify), random, two prompt ports as system messages ("Think about
this the way you would think about a math problem." / "... the way you would brainstorm a research idea."), and on math
and gpqa "idea + commit" (the idea vector plus half of A2, renormalised to 6.75). Dose: 0.5x and 1.5x of each signature
vector on the first 100 tasks, seed 1.
Outcomes: accuracy at 12k (primary) and 32k for math/gpqa/planning (free-text extraction for math); originality,
feasibility and their mean for idea; MoReBench rubric score for moral; answered, loops and tokens everywhere.
H2a, written down before any outcome is graded (data/transplant/prereg.json, hashed): each target's within-task
behaviour-outcome slopes (unsteered corpus, task fixed effects, re-graded math) times (i) the profile shift the vector is
built to make and (ii) the shift Track B measures. Tests: sign agreement over the 20 off-diagonal cells; Spearman of
predicted vs observed within target, averaged; pooled correlation in outcome-SD units.
H2b: loops and unanswered rise more when divergent signatures (idea, moral) enter convergent targets than the reverse;
"idea + commit" removes most of those loops while keeping the shift and conflict increase.

## D-53 (proposed, ratified in outline by D-48): F3 beyond Math
On the gpqa and planning targets, add the two vectors whose Math behaviour shift matched A2 from orthogonal residual
directions: the SEAL direction at A2's dose and the zero-start RL bias (step 60). Prediction: both reproduce A2's GPQA
and planning behaviour shift (cosine > 0.8) and land near its accuracy, ordered by shift size. Then one regression over
every vector we have run on Math, GPQA and planning: accuracy change at 12k on the Track B shift vs on the residual-space
cosine with A2. Expectation: the behaviour shift carries the prediction, the residual cosine adds nothing.

## D-54 (proposed): second model = Gemma-4-12B, after the Qwen matrix
Other lineage, already in the atlas with 500 labelled traces per domain, and our vLLM plugin serves it. Readout corpus =
its atlas traces on 300 tasks per domain (no new generation); held-out = the other 200. Needs HF loading and hooks for
the unified checkpoint, the map fit, its own efficient archetype, and then the core cells: Math<->Idea transplant, its A2
analogue on Math and GPQA, random, and the Countdown crossover. Starts when the Qwen matrix has been read.
D-50 build note (2026-10-06, before any generation): two item rules added after inspecting the tasks. (1) Math items are
restricted to integer answers (311 of 500 eligible), and the three MCQ distractors are other atlas models' wrong integer
answers, then nearby integers. The first build used the canonical predicted answers, which carry the bold-parser debris
("**", "x", prose) and order-swapped lists equal to the reference. (2) GPQA items whose stem already lists the option
texts are skipped (453 of 500 eligible), since their "free" version would still show the options. Final set: 80 math
(16 per level) and 80 GPQA items, 640 prompts; mode_control/data/f1/tasks. Generation config mode_control/configs/exp_f1.yaml
(configs/exp.yaml with a 32k budget and 32-trace checkpoints).
