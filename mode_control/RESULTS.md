# Results (interim, 2026-09-03)

Model Qwen3.5-9B, domains Math vs Idea, phenotype window 4096 reasoning tokens.

## RQ1: is there an early, low-dimensional state tied to the eventual signature?

Gate 1 (spontaneous domain gap on the prompt-derived direction): passes. The direction learned
only from paired strategy prompts (exploratory minus checking, same tokens teacher-forced)
separates unprompted Math from Idea traces at AUROC 0.99 to 1.00 from layer 3 onward, both at the
last prompt token and averaged over generated tokens 1 to 32. Its cosine with the raw
Idea-minus-Math axis is 0.15 to 0.30 (mean-32 phase) and 0.5 to 0.7 (last prompt token), so it is
related to, but not the same as, the topic axis.

Gate 2 (within-domain prediction of the later signature): partial.
- Within Math, a more exploratory early projection predicts a more Idea-like cognitive signature
  later: out-of-fold Spearman 0.34 to 0.38 at layers 13 to 19 (mean-32 phase). This survives
  controlling for prompt length and MATH level (partial 0.27 at layer 19). A 64-PC ridge readout
  of the same target reaches 0.44 to 0.46, so the single direction carries most of the linearly
  readable within-Math signal. Conversational score: weaker (0.06 to 0.21).
- Within Idea, nothing at this phase. The reason is structural: every Idea prompt is the same
  template with one keyword swapped, and all 344 valid Idea traces open with the identical twelve
  tokens ("Thinking Process: 1. Analyze the Request"), so the early projection has a third of the
  Math variance. Testing RQ1 within Idea needs a later phase (after the keyword-specific content
  starts) in the powered run.
- Caveat: at the last-prompt-token phase the within-Math sign flips (negative), driven by prompt
  length and difficulty (rho with MATH level -0.48); the partial correlation is about zero. The
  mean-32 phase is the one to use.

Figures: `data/directions/figures/layer_phase_map.png`, `rep_space_mean32.png`.

Method note: Ledoit-Wolf whitening was a bug (n = 783 < d = 4096) and is replaced by the plain
mean paired difference (DECISIONS D-09). Steering layer 19 (D-10).

## RQ2 (steering), RQ3 (task performance), RQ4 (prompt mediation)

Pilot `pilot1` (40 Math + 40 Idea tasks, one seed, 19 conditions) is generating and judging.
Prompt-derived direction at layer 19: baseline projections Math -1.5, Idea +2.0 (SD 1.9);
the strategy prompts move the projection by 2.0, about 57 percent of the spontaneous gap.

## RQ4 premise: what the strategy prompts do to the phenotype (pilot1, 40 + 40 tasks, one seed)

Paired shifts against the in-run baseline, combined score (positive = more Idea-like), 95% CI:

| condition | Math | Idea |
|---|---|---|
| checking prompt | +0.11 (−0.40, 0.63) | −2.56 (−3.96, −1.14) |
| exploratory prompt | +1.29 (0.54, 2.12) | −3.55 (−4.85, −2.25) |
| additive pulse +u, layer 19 | +0.41 (−0.14, 0.97) | +0.94 (−0.24, 2.11) |
| null-space direction control | +0.75 (0.19, 1.34) | +0.05 (−1.00, 1.13) |

In Math the exploratory prompt works as intended: more perspective shift (+0.029) and conflict
(+0.013), less verification (−0.026), longer traces (+654 tokens), and a drop in answered-and-correct
(0.18 to 0.06 of answered traces, small n). In Idea both prompts push traces toward the Math profile,
and the exploratory prompt does so most: backtracking −0.059, perspective shift −0.038,
verification −0.030. An explicit instruction to explore makes the model enumerate alternatives in a
planned way, which the judge does not label as backtracking (abandon and retry). The Idea signature
therefore measures unplanned revision, and the prompt contrast (exploratory minus checking) has
opposite signs in the two domains (+1.2 in Math, −1.0 in Idea).

The additive pulse moves Idea in the expected direction but by less than the noise band set by the
null-space control (D-15). The gamma sweep with RMS-matched random controls is the decisive test.

## RQ2, pilot1 gamma sweep (64-token pulse at layer 19, 40 + 40 tasks, one seed): null

Idea-minus-Math gap on the combined score per condition, and attenuation relative to the in-run
baseline gap (14.4 logits):

| condition | gap | 95% CI | attenuation |
|---|---|---|---|
| baseline | 14.4 | 13.4 to 15.5 | 0 |
| amplify (γ = 2) | 14.4 | 13.3 to 15.5 | 0.00 |
| neutralize (γ = 0) | 14.2 | 13.0 to 15.6 | 0.01 |
| reflect (γ = −1) | 14.4 | 13.0 to 15.9 | 0.00 |
| additive +u (prompt shift) | 14.9 | 13.7 to 16.2 | −0.04 |
| PCA-whitened variant, reflect | 14.6 | 13.3 to 16.0 | −0.02 |
| random direction a / b (RMS-matched) | 14.8 / 14.5 | | −0.03 / −0.01 |
| null-space direction | 13.7 | 12.4 to 15.0 | 0.05 |
| raw domain direction, reflect | 13.2 | 12.2 to 14.3 | 0.08 |
| checking prompt | 11.7 | 10.5 to 13.2 | 0.19 |
| exploratory prompt | 9.6 | 8.3 to 11.0 | 0.34 |

Per-domain paired shifts for every steered condition are within the random-control band (about
±1 logit). Reading: at this locus and strength (|Δh| about 3.5, roughly 8 percent of the residual
norm, for 64 positions), the state that predicts the signature is not a control knob, or the
intervention is too weak or too local. pilot2 tests sustained steering, γ = −3, a seven-layer
band, one-gap additive shifts, a centroid swap, and a matched random band.

## RQ1 refined with the late phase (tokens 65 to 192) and a general linear readout

A 64-PC ridge readout (out of fold on the prompt folds), Spearman with the later frozen score:

| phase | within Math, cognitive | within Math, conversational | within Idea, cognitive | within Idea, conversational |
|---|---|---|---|---|
| tokens 1-32 | 0.41 to 0.48 | 0.13 to 0.19 | −0.03 to 0.07 | 0.18 to 0.23 |
| tokens 1-64 | 0.45 to 0.48 | 0.16 to 0.18 | −0.04 to 0.05 | 0.17 to 0.23 |
| tokens 65-192 | 0.48 to 0.51 | 0.27 to 0.30 | 0.02 to 0.06 | 0.12 to 0.20 |

So an early state predicts the later signature within Math for both families, and within Idea for
the conversational family only. The Idea cognitive signature (backtracking) is not linearly
readable early at any phase up to token 192; the prompt-derived direction alone reads nothing
within Idea. Layers 16 to 24 are best throughout. Full table: `data/directions/readout_by_phase_k192.csv`.

## R1-Distill-Llama-8B: RQ1 readout (baseline traces, 498 Math + 493 Idea valid)

The prompt-derived direction fails Gate 1 in R1: exploratory-minus-checking is unrelated or
anti-aligned to R1's spontaneous Math–Idea axis at the early phases (cosine with the raw domain
axis −0.2 to 0.05; projection domain AUROC 0.0 to 0.4), turning weakly positive only at the late
phase in layers 20 to 28 (AUROC 0.78 to 0.93). The strategy-prompt contrast is therefore not the
domain axis for this model.

A general linear readout (64-PC ridge, out of fold) is strong within both domains and, unlike
Qwen, includes the cognitive signature within Idea, growing with phase:

| phase | Math cog | Math conv | Idea cog | Idea conv |
|---|---|---|---|---|
| last prompt token | 0.47 | 0.32 | 0.07 to 0.12 | 0.02 to 0.09 |
| tokens 1-32 | 0.45 to 0.49 | 0.45 to 0.48 | 0.24 to 0.25 | 0.19 to 0.20 |
| tokens 1-64 | 0.45 to 0.49 | 0.44 to 0.49 | 0.32 | 0.22 to 0.26 |
| tokens 65-192 | 0.45 to 0.48 | 0.52 to 0.53 | 0.44 to 0.48 | 0.26 to 0.31 |

R1 steering therefore uses a supervised readout direction (`readout_r1`, ridge on the top 64 PCs
to the within-domain-centered combined score, fit on training folds for the gates and on all
baseline traces for steering) rather than the prompt contrast. Held-out tasks and the frozen score
keep this from being optimization against the judge.

## RQ2, pilot1 extra arms: the first sign of life is sustained, not pulsed

Combined-score paired shifts vs the in-run baseline (40 tasks per domain, one seed):

| condition | Math | Idea | gap attenuation |
|---|---|---|---|
| reflect, sustained at layer 19 (γ = −1 at every position) | +0.45 (−0.07, 1.02) | **−4.41 (−5.47, −3.30)** | **0.34** |
| reflect, rank-4 pulse | +0.53 | +0.24 | 0.02 |
| reflect pulse at layer 2 / layer 29 | −0.04 / +0.27 | +0.87 / −0.17 | −0.06 / 0.03 |
| centroid swap pulse | +0.11 (n = 16) | +1.05 | −0.02 |
| additive −u pulse | +0.51 | +0.76 | −0.02 |
| checking prompt + neutralize pulse | −0.09 | −2.15 | 0.14 |
| exploratory prompt + neutralize pulse | +1.50 | −2.99 | 0.31 |

Sustained reflection is the only steering arm that moves anything, and it moves Idea toward the
Math profile by as much as the exploratory prompt does: backtracking −0.047, perspective shift
−0.051, conflict −0.052 (baseline rates 0.15, 0.31, 0.14), verification unchanged; Idea traces
close their think block within the window in 48% of cases instead of 10%, with no increase in
loop flags and coherent text. Math moves little (perspective shift +0.019). Neutralizing during
the pulse does not remove the prompt effect, as expected given the pulse does nothing on its own.

Reading: an early 64-token pulse does not set a persistent mode at this layer; persistent
steering does change the Idea signature. Whether this is specific to the direction (versus any
sustained perturbation making Idea converge sooner) is what the matched random sustained band in
pilot2 tests. The asymmetry (Idea moves, Math barely) mirrors the prompts.

## RQ2, pilot2 strong arms: preliminary from Track A counts (Track B pending)

Sustained steering, 40 + 40 tasks, one seed. Text-quality flags and Idea paired count shifts vs
baseline (whole-trace counts; baseline Idea: backtracking 11.0, perspective shift 18.7, conflict 10.8):

| condition | hard loops | closed think | Idea backtracking | Idea persp. shift | Idea conflict |
|---|---|---|---|---|---|
| baseline | 0.02 | 0.30 | | | |
| reflect γ = −3, layer 19 | 0.10 | 0.55 | −6.7 ± 0.9 | −7.4 ± 0.9 | −5.0 ± 0.5 |
| additive −1 gap, band 12-24 | 0.18 | 0.46 | −4.2 ± 1.3 | −3.6 ± 1.8 | −4.7 ± 0.6 |
| reflect γ = −1, band 12-24 | 0.01 | 0.29 | −0.0 ± 1.0 | −1.1 ± 1.1 | −1.7 ± 0.5 |
| centroid swap, band | 0.04 | 0.20 | −0.4 ± 1.3 | −2.2 ± 1.2 | −2.6 ± 0.6 |
| random band, matched | 0.00 | 0.28 | +2.0 ± 1.1 | +0.2 ± 0.9 | +0.8 ± 0.4 |
| additive +1 gap, band (toward Idea) | 0.74 | 0.01 | degenerate | | |
| raw-domain −1 gap, band | 0.26 | 0.00 | degenerate (traces collapse to a third of their length) | | |

Reading so far: moving the state toward the Math pole reduces Idea's revision and dialogue
behaviors in a direction-specific, dose-dependent way (the matched random band does not), at the
cost of some loop risk at the highest doses. Moving Math toward the Idea pole does not produce
Idea-like behavior at usable strength; at full strength it degenerates. Track B rates and the
frozen scores follow.

## RQ2, pilot2 with Track B (sustained arms, 40 + 40 tasks, one seed): one-directional control

Combined-score paired shifts vs the in-run baseline, and gap attenuation:

| condition | Math | Idea | attenuation | hard loops |
|---|---|---|---|---|
| reflect γ = −3, layer 19, sustained | −0.5 (−1.2, 0.2) | **−7.9 (−9.4, −6.4)** | **0.50** | 0.10 |
| additive −1 gap, band 12-24, sustained | −0.7 (−1.2, −0.2) | **−6.0 (−8.1, −3.7)** | 0.36 | 0.18 |
| centroid swap, band, sustained | +0.3 (n = 16) | −3.0 (−4.8, −1.4) | 0.26 | 0.04 |
| reflect γ = −1, band, sustained | −0.1 | −1.9 (−3.3, −0.5) | 0.12 | 0.01 |
| random band, matched to ±1 gap | −0.2 (−0.7, 0.3) | +0.4 (−1.2, 1.9) | −0.04 | 0.00 |
| additive +1 gap, band (toward Idea) | +9.3 | +16.9 | −0.52 | 0.74 (degenerate) |
| raw-domain −1 gap, band | traces collapse to about 5 segments | | | 0.26 |

Under sustained γ = −3 at layer 19, Idea traces acquire the Math profile rather than merely losing
behaviors: backtracking 0.16 → 0.08, perspective shift 0.32 → 0.22, conflict 0.14 → 0.07, and
verification 0.12 → 0.17. The effect is dose dependent (γ = −1 in pilot1 gave −4.4, γ = −3 gives −7.9),
direction specific (matched random band null), and reproduces with an additive shift across a
seven-layer band. The apparent reversal under +1 gap is loop text ("Wait, no...") scored as
backtracking and conflict and is not a valid result; reflecting Math's state does not make Math
Idea-like. So the prompt-derived axis is a control knob toward convergent, verification-heavy
reasoning, at least in this model, and not (at these doses) a knob toward exploratory reasoning.
pilot3 scans lower Idea-ward doses (0.25 and 0.5 gap band, 1 and 2 gap at one layer).

Figures: `data/runs/pilot2/analysis/figures/`. Loop-free shifts: `paired_shifts_noloop.csv`.

## RQ3, task performance under steering (math correctness so far; Idea quality judge queued)

Math accuracy counting an unanswered task as wrong, paired against the same tasks unsteered (mean ± SE):

| run | condition | accuracy (all tasks) | change vs baseline |
|---|---|---|---|
| Qwen powered, 12k budget, 150 tasks | baseline | 0.28 (0.41 among answered) | |
| | checking prompt | 0.33 | +0.05 ± 0.05 |
| | exploratory prompt | 0.40 | +0.12 ± 0.05 |
| | sustained reflect γ = −1, layer 19 (138 tasks so far) | 0.28 | +0.01 ± 0.05 |
| Qwen pilot2, 4k cap, 40 tasks | sustained reflect γ = −3, layer 19 | 0.15 | −0.03 ± 0.06 |
| | additive −1 gap, band 12-24 | 0.28 | +0.10 ± 0.09 |
| | random band (matched) | 0.20 | +0.03 ± 0.06 |
| | additive +1 gap, band (Idea-ward) | 0.00 | −0.18 ± 0.06 |
| | raw-domain −1 gap, band | 0.00 | −0.18 ± 0.06 |
| R1 pilot, 4k cap, 40 tasks (traces complete) | baseline | 0.62 (0.83 among answered) | |
| | readout −3 SD, band (Math-ward) | 0.62 | 0.00 ± 0.04 |
| | readout +3 SD, band (Idea-ward) | 0.62 | 0.00 ± 0.06 |
| | random band (matched) | 0.68 | +0.05 ± 0.05 |
| | raw-domain ±1 gap, band | 0.00 | −0.63 ± 0.08 (100% loops) |

Reading: at the doses that move the reasoning signature, Math accuracy does not change in either
model (differences within ±1 SE of zero). Accuracy collapses only where generation degenerates
(Idea-ward at full strength, the raw-domain direction), and those arms are excluded from behavior
claims anyway. At the 12k budget the exploratory prompt raises Math accuracy (+0.12 ± 0.05, with
82% of traces finishing vs 69%), which reverses the small-n 4k-cap impression from pilot1. Idea
quality (the atlas originality/feasibility rubric) is not measured yet; quality-judge jobs are
queued for the R1 pilot and the powered Qwen run. Answered-only accuracy is selection-biased
(steering changes who finishes), so the all-tasks number is primary.

## RQ2, pilot3 Idea-ward dose scan: preliminary from Track A counts (Track B pending)

Sustained additive steering toward the exploratory pole at lower doses (40 + 40 tasks, one seed).
Paired whole-trace count shifts vs baseline (baseline Math: backtracking 0.2, perspective shift 3.2,
conflict 2.1; baseline Idea: 11.0, 18.6, 10.9):

| condition | Math loops | Math persp. shift | Math conflict | Math backtracking | Idea loops | Idea backtracking | Idea persp. shift |
|---|---|---|---|---|---|---|---|
| +0.25 gap, band 12-24 | 0.00 | +0.8 ± 0.2 | +0.8 ± 0.2 | −0.0 | 0.10 | +3.9 ± 0.9 | +3.8 ± 1.0 |
| +0.5 gap, band 12-24 | 0.00 | +1.3 ± 0.2 | +1.8 ± 0.2 | +0.0 | 0.38 | +6.3 ± 0.9 | +8.3 ± 1.1 |
| +1 gap, layer 19 only | 0.00 | +0.7 ± 0.2 | +0.8 ± 0.2 | +0.1 | 0.05 | +2.0 ± 1.0 | +2.1 ± 1.0 |
| +2 gap, layer 19 only | 0.00 | +1.0 ± 0.2 | +1.1 ± 0.2 | +0.1 | 0.20 | +4.5 ± 0.8 | +6.0 ± 1.1 |
| −2 gap, layer 19 only (Math-ward) | 0.05 | +0.3 | −0.3 | +0.1 | 0.00 | −2.1 ± 0.7 | −3.5 ± 0.9 |
| raw-domain −0.25 gap, band | 0.02 | +2.0 | +1.0 | +0.4 | 0.80 | degenerate | |

Reading: below the loop threshold the exploratory direction moves Math toward the Idea profile on the
conversational behaviors (perspective shift and conflict rise by 30 to 85 percent, traces lengthen by
1 to 2k characters, no loops), but not on backtracking. On Idea tasks the same doses amplify the
Idea signature in a dose-dependent way, at rising loop cost because Idea is already at that end.
So the axis is bidirectional for the dialogue-like behaviors and one-directional for revision.
The raw-domain (topic) direction degenerates even at a quarter gap. Track B rates and frozen
scores follow when the judge finishes.

### RQ3 correction and powered math accuracy (2026-09-04)

The repo's deterministic math grader reads a markdown-bold "**Answer:**" line as the answer "**"
and marks it wrong; the production pipeline avoids this through the LLM extractor, but our direct
grading did not. `mc.analyze` now strips bold markers before grading. This deflated every
condition equally, so the differences above stand, but the levels were wrong. Corrected, at the
powered scale (150 Math tasks, 12288-token budget; unanswered counts as wrong):

| condition | accuracy, all tasks | among answered | change vs baseline (paired, 95% CI) |
|---|---|---|---|
| baseline | 0.49 | 0.72 | |
| checking prompt | 0.52 | 0.71 | +0.03 (−0.07, 0.11) |
| exploratory prompt | 0.58 | 0.71 | +0.09 (−0.01, 0.17) |
| sustained reflect γ = −1, layer 19 (99 tasks so far) | 0.42 | 0.67 | −0.06 (−0.18, 0.05) |
| sustained reflect γ = −3, layer 19 (94 tasks so far) | 0.50 | 0.72 | +0.04 (−0.07, 0.17) |

Accuracy among answered traces is flat across conditions; the all-tasks differences track how
many traces finish within the budget (the exploratory prompt finishes 82% vs 69%).

### RQ3, Idea quality (R1 pilot; atlas originality/feasibility rubric, 0-1)

Paired against the same tasks unsteered (R1-Distill, 40 Idea tasks, readout direction, sustained):

| condition | quality | change vs baseline (mean ± SE) |
|---|---|---|
| baseline | 0.60 | |
| toward Math (−3 SD, band) | 0.57 | −0.024 ± 0.018 (n = 33) |
| toward Idea (+3 SD, band) | 0.56 | −0.033 ± 0.017 (n = 26) |
| random band (matched) | 0.59 | +0.003 ± 0.016 (n = 32) |
| raw-domain ±1 gap | no answers (100% loops) | |

Reading: a small, borderline decrease in either direction (about 1.5 to 2 SE) against no change for
the random band, on a small sample. The powered Qwen quality judge (150 Idea tasks) is running.

### RQ3, Idea quality (Qwen powered run, 150 Idea tasks; atlas rubric: originality and feasibility, 1-10, quality = mean/10)

Among answered traces, paired against the same tasks unsteered (mean ± SE):

| condition | answered | originality | feasibility | quality | change vs baseline |
|---|---|---|---|---|---|
| baseline | 0.81 | 8.5 | 7.2 | 0.79 | |
| checking prompt | 0.83 | 8.5 | 7.4 | 0.79 | +0.009 ± 0.006 |
| exploratory prompt | 0.79 | 8.5 | 6.8 | 0.77 | −0.014 ± 0.006 |
| sustained reflect γ = −1, layer 19 | 0.92 | 8.3 | 7.5 | 0.79 | −0.001 ± 0.005 |
| sustained reflect γ = −3, layer 19 | 0.85 | 8.0 | 7.3 | 0.77 | −0.019 ± 0.005 |
| additive −1 gap, band 12-24 (partial n) | 0.59 | 6.6 | 8.0 | 0.73 | −0.060 ± 0.009 |
| centroid swap, band (partial n) | 0.82 | 5.9 | 8.5 | 0.72 | −0.056 ± 0.008 |
| random band, matched (partial n) | 0.71 | 8.7 | 7.1 | 0.79 | −0.001 ± 0.007 |
| additive +0.5 gap, band (Idea-ward) | 0.07 | | | | traces rarely finish |

Reading: moving Idea reasoning toward the Math mode makes the resulting ideas more feasible and
less original, in proportion to how far the behavior signature moved (γ = −1: no change; γ = −3:
originality −0.5; band shift and swap: originality −1.9 to −2.6, feasibility +0.8 to +1.3). The
random band changes neither. So the differentiation is functional in the direction the behaviors
suggest: convergent, verification-heavy deliberation yields safer, more feasible ideas at the cost of
originality. Counting unanswered traces as zero, γ = −1 improves all-task quality (+0.08 ± 0.03)
because more Idea traces finish within budget, while the strongest band arms lose on completion.
Math accuracy is unchanged under the same interventions (above).

## R1-Distill-Llama-8B steering pilot (readout direction, ±3 SD, sustained, layers 12-24; 40 + 40 tasks)

R1's frozen scores have a smaller Math–Idea gap (7.8 logits). Paired shifts on the combined score:

| condition | Math | Idea | note |
|---|---|---|---|
| readout direction toward Idea (+3 SD) | **+2.1 (1.3, 3.0)** | +1.6 (0.3, 3.0) | conversational family carries it; Idea traces +1,060 tokens |
| readout direction toward Math (−3 SD) | −1.1 (−2.1, −0.1) | −0.9 (−2.3, 0.5) | Idea: backtracking −0.038, conflict −0.043, but verification −0.037 too; −700 tokens |
| random band (matched) | +0.6 (−0.3, 1.5) | +0.0 (−1.3, 1.3) | |
| raw-domain ±1 gap | degenerate (100% loops, all rates 0) | | |

Reading: in R1 the supervised readout direction is a bidirectional but weak knob that mostly
changes dialogue-like behaviors and trace length in both domains at once (the Math–Idea gap
barely changes: attenuation −3% and +6%). It does not give Idea the Math profile (verification
falls rather than rises), so it behaves more like a verbosity/dialogue axis than a domain mode.
Math accuracy and Idea quality are unchanged (above). The strong, profile-specific control seen in
Qwen along the prompt-derived direction therefore does not transfer to R1 through this direction;
whether R1 has an equivalent direction elsewhere (e.g. a checking/exploratory contrast that does
separate its domains) is open.

## RQ2 powered (Qwen3.5-9B, 150 + 150 tasks, 12288-token budget, 8192-token judge window)

Combined-score paired shifts vs the in-run baseline (95% CI), loop-free subset in brackets:

| condition | Math | Idea | gap attenuation |
|---|---|---|---|
| checking prompt | −0.54 (−0.88, −0.19) | −5.1 (−6.7, −3.6) | 0.30 |
| exploratory prompt | +1.13 (0.74, 1.51) | −4.8 (−6.5, −3.4) | 0.39 |
| sustained reflect γ = −1, layer 19 | −0.33 (−0.68, 0.05) | **−6.1 (−7.6, −4.8)** [−4.3] | 0.37 |
| sustained reflect γ = −3, layer 19 | −0.88 (−1.27, −0.48) | **−9.2 (−10.7, −7.9)** [−7.4] | 0.54 |

Under γ = −3, Idea: verification 0.12 → 0.16, backtracking 0.17 → 0.07, perspective shift 0.30 →
0.22, conflict 0.13 → 0.05; traces 2,350 tokens shorter; loop rate 16% vs 19% unsteered; answers
+3%. Math accuracy unchanged (0.00 ± 0.04). Band, swap, and random arms at this scale are still
being judged.

## RQ2, pilot3 Idea-ward dose scan with Track B (40 + 40 tasks, 4096 window)

| condition | Math loops | Math shift | Idea loops | Idea shift [loop-free] |
|---|---|---|---|---|
| +0.25 gap, band 12-24 | 0.00 | +0.33 (−0.51, 1.09) | 0.10 | +3.0 (1.6, 4.5) [+2.2] |
| +0.5 gap, band 12-24 | 0.00 | **+2.20 (1.48, 2.90)** | 0.38 | +9.2 (6.8, 12.0) [+5.3, n = 25] |
| +1 gap, layer 19 | 0.00 | +0.31 (−0.25, 0.80) | 0.05 | +1.6 (0.1, 3.1) [+2.3] |
| +2 gap, layer 19 | 0.00 | **+1.43 (0.64, 2.21)** | 0.20 | +6.5 (4.2, 8.8) [+4.8] |
| −2 gap, layer 19 (Math-ward) | 0.05 | −0.52 (−1.16, 0.12) | 0.00 | −4.0 (−5.6, −2.6) |
| raw-domain −0.25 gap, band | 0.02 | +1.25 (verification −0.065) | 0.80 | degenerate |

Math under +0.5 gap band: backtracking 0.013 → 0.029, perspective shift 0.13 → 0.17, conflict
0.03 → 0.06, verification unchanged; no loops. Math accuracy (all tasks, 4k cap) falls 0.38 →
0.18 and answers 0.45 → 0.28, i.e. Idea-ward steering makes Math reasoning longer, more
revision-heavy, and less often correct within budget.

Updated reading: the prompt-derived axis is bidirectional at coherent doses, with asymmetric reach:
Idea can be moved about half of the way to Math (−9 of a 15-logit gap) while Math moves about a
tenth to a seventh of the way to Idea (+1.4 to +2.2) before the Idea-ward dose starts to loop on
Idea tasks. Mismatched steering is costly in the Math direction (accuracy) and in the Idea
direction (originality), and neutral otherwise.

### RQ2 powered, band arms (150 + 150 tasks, 12288 budget, 8192 window)

| condition | Math | Idea [loop-free] | attenuation | Idea loops | Math answers | Math accuracy change |
|---|---|---|---|---|---|---|
| additive −1 gap, layers 12-24, sustained | −1.65 (−2.09, −1.21) | −8.5 (−10.0, −7.1) [−6.3] | 0.44 | 0.46 (baseline 0.19) | −0.23 | −0.13 (−0.23, −0.04) |
| random band, matched | +0.04 (−0.30, 0.40) | +1.2 (−0.5, 2.9) [+1.1] | −0.08 | 0.27 | +0.06 | +0.03 (−0.05, 0.11) |
| sustained reflect γ = −3, layer 19 (for comparison) | −0.88 | −9.2 [−7.4] | 0.54 | 0.16 | −0.01 | 0.00 |

Reading: at the longer budget the seven-layer shift at one full gap is too strong; loops accumulate
over 12k tokens, answers drop in both domains, and Math accuracy falls. The single-layer reflection
at γ = −3 gives a larger, cleaner effect with no outcome cost, so it is the reference intervention.
The random band's small positive Idea shift comes with a mild loop increase (27% vs 19%), i.e. the
generic-perturbation floor at this magnitude is about +1 logit on Idea and zero on Math. The band
swap and +0.5-gap arms are still being judged.

### RQ2 powered, remaining arms (150 + 150 tasks)

| condition | Math | Idea [loop-free] | attenuation | loops (Math / Idea) | Math accuracy change |
|---|---|---|---|---|---|
| centroid swap, layers 12-24, sustained | −0.01 (−0.36, 0.34) | −4.6 (−6.0, −3.4) [−2.9] | 0.30 | 0.04 / 0.19 (unchanged) | +0.02 |
| additive +0.5 gap, layers 12-24, sustained (Idea-ward) | **+2.4 (1.9, 3.0)** [+2.0] | +17 (93% loops; degenerate) | | 0.13 / 0.93 | **−0.13 (−0.23, −0.04)** |

The swap is the cleanest Math-ward control of the band family (no loop or answer cost) but weaker
than the single-layer reflection. The Idea-ward half-gap band replicates the pilot on Math at
150 tasks (backtracking 0.015 → 0.031, perspective shift +0.03, conflict +0.034) and confirms
its cost: fewer answers within budget and lower accuracy. On Idea tasks the same dose is beyond
the coherent range at a 12k budget.

## Summary of findings (2026-09-04)

1. RQ1. A direction learned only from paired strategy prompts on the same task separates unprompted
   Math from Idea traces (AUROC ≈ 1) and predicts each trace's later signature within Math
   (Spearman 0.3-0.5) and, for the conversational family, within Idea (about 0.2). Qwen's Idea
   openings are templated, which hides within-Idea cognitive variance; R1 shows both families.
2. RQ2. A 64-token pulse at the start does nothing at any layer, rank or strength. Sustained steering
   along the direction moves the later signature in both directions: Idea traces move about half
   of the way to the Math profile (γ = −3 at layer 19: −9.2 of a 15-logit gap, verification up,
   backtracking, shifts and conflicts down, text coherent), Math traces move a tenth to a seventh
   of the way toward Idea before generation starts to loop. Matched random directions do nothing;
   the raw topic direction only breaks generation. Multi-layer additive shifts reach similar
   sizes but degrade text at long budgets; the single-layer reflection is the reference.
3. RQ3. Math accuracy is unchanged under Math-ward steering and falls (−0.13) under Idea-ward
   steering. Idea originality falls and feasibility rises under Math-ward steering in proportion
   to the behavior change, while the random control changes neither. Each domain's own mode is the
   one that serves its outcome.
4. RQ4. The strategy prompts move Math as intended but push Idea toward Math as well: an explicit
   "explore" instruction produces orderly enumeration, not the unplanned revision the Idea
   signature measures. Prompt effects (30-39% attenuation) are of the same size as γ = −1
   steering; neutralizing the direction during a 64-token pulse does not remove them (a sustained
   necessity test has not been run).
5. R1-Distill: the prompt contrast is not R1's domain axis; a supervised readout direction gives
   only a weak, bidirectional verbosity-like effect. The strong, profile-specific control is so
   far a one-model result.

Limits: one seed; the pilot arms use 40 tasks per domain; loop and completion rates differ by
arm and are reported alongside every effect; the judge is the same model-based judge as the
atlas.

### sched1 interim (baseline + sustained idea, 200 + 200 tasks)
- Baseline at 200 tasks: Idea loops 0.24, answers 0.76; Math loops 0.10, answers 0.64, accuracy 0.49.
- Sustained Idea-ward (+2 gap, L19): Math +1.6 (1.2, 2.0), accuracy -0.125 (-0.20, -0.06); Idea +11.4 but
  loops 0.68 and answers 0.33, i.e. degenerate on Idea tasks at this budget. Early/late split: Idea +5.0 early,
  +19.4 late; Math +1.2 early, +2.2 late.
- Phase pipeline check: unsteered early vs late halves are close (Idea 9.7 vs 8.4, Math -6.4 vs -7.0).
- Sustained Math-ward (-2 gap, L19): Idea -4.1 (-5.1, -3.1), loops 0.12 (vs 0.24), answers +12 pts, backtracking
  -0.048; Math -0.68 (-0.95, -0.38), accuracy +0.045 (-0.02, 0.10), verification -0.025. Early/late: Idea -3.7 /
  -3.9, Math -0.5 / -0.7. So at this dose the Math-ward sign is clean in both domains and the Idea-ward sign is
  clean only on Math.
- idea->math (+2 then -2 at the task's own midpoint; switch median 3.2k Idea / 3.7k Math tokens):
  Idea +4.1 (3.0, 5.4) overall, +1.1 (0.3, 1.8) loop-free; loops 0.35 (baseline 0.24), answers -10.5 pts.
  Phases: early +4.9 (= sustained-idea early +5.0), late +4.5 (sustained-math late is -3.9): the late
  Math-ward push does not undo the early exploratory regime on Idea tasks.
  Math +0.65 (0.25, 1.04): early +1.1 (= sustained-idea early), late +0.07 (back to baseline); accuracy
  +0.03 (-0.04, 0.10), answers +7 pts, loops 0.105 (baseline 0.10). Explore-early/verify-late costs Math
  nothing, whereas sustained explore costs 12.5 points.

### sched1: order contrast at equal dose (200 + 200 tasks, |alpha| = 2 gap, L19, switch at the task's own midpoint)

| | idea->math | math->idea | order effect (paired, idea->math minus math->idea) |
|---|---|---|---|
| Math accuracy change | +0.03 (-0.04, 0.10) | -0.07 (-0.14, 0.00) | **+0.10 (0.03, 0.18)** |
| Math answered within budget | +0.07 | -0.085 | **+0.155 (0.08, 0.23)** |
| Math Idea-likeness | +0.65 | +0.14 | +0.51 (0.09, 0.95) |
| Idea Idea-likeness | +4.1 (loop-free +1.1) | -0.7 (loop-free -1.0) | +4.8 (3.0, 6.7) |
| Idea loops / answered | 0.35 / 0.65 | 0.26 / 0.73 | |

Phase check (Idea-likeness vs unsteered, same split for every arm):
- idea->math: Idea early +4.9, late +4.5; Math early +1.1, late +0.1.
- math->idea: Idea early -3.5, late +5.7; Math early -0.4, late +1.6.
Reading: the early phase always takes hold. A late Idea-ward push also takes hold in both domains; a late
Math-ward push does not pull Idea tasks back (the exploratory regime is sticky) and only returns Math tasks
to baseline. Explore-early/verify-late is the schedule that costs Math nothing; the reverse order costs
7 points and 8 points of answering. Idea originality/feasibility pending the quality judge.
- explore->check with a fixed 2048-token switch (robustness arm): Math accuracy +0.025 (-0.05, 0.10), answers
  +5.5 pts, Idea-likeness -0.09 (same accuracy result as the midpoint switch). Idea tasks: Idea-likeness -0.9
  (-2.0, 0.1), loops 0.13 (unsteered 0.24, midpoint-switch arm 0.35), answers +11.5 pts. So how long the
  exploratory phase lasts decides whether it becomes sticky: ~2k tokens of exploring followed by checking
  ends Math-like and coherent; 3-4k tokens of exploring does not return. The Math benefit of explore-early /
  verify-late does not depend on the switch rule.
- random direction + then - (same |alpha|, same switch): Idea +0.07 (-1.1, 1.3), Math -0.04 (-0.31, 0.24),
  accuracy +0.035 (-0.02, 0.09), loops and answers unchanged. The schedule effects are direction-specific.
- check->explore with a fixed 2048-token switch: Math accuracy -0.07 (-0.135, -0.005), answers -9 pts (same as
  the midpoint-switch version); Idea +3.2 (1.7, 4.7), loops 0.38, answers -15 pts. Under both switch rules the
  reverse order costs Math accuracy, and the arm with the longest exploratory tail loops most on Idea tasks.
  All eight full-dose arms judged; answer-quality judge pending.

### sched1 complete: answer quality on Idea tasks (rubric judge, paired vs unsteered, tasks answered in both)

| arm | originality | feasibility | answered |
|---|---|---|---|
| sustained explore | **+0.30** (0.17, 0.43) | -0.28 (-0.57, 0.00) | 0.33 |
| sustained check | **-0.30** (-0.44, -0.18) | +0.12 (-0.08, 0.32) | 0.88 |
| explore -> check (midpoint) | +0.09 (-0.03, 0.21) | **-0.22** (-0.41, -0.03) | 0.65 |
| explore -> check (switch 2k) | -0.06 (-0.17, 0.06) | -0.11 (-0.30, 0.09) | 0.87 |
| check -> explore (midpoint) | -0.12 (-0.26, 0.03) | +0.03 (-0.24, 0.28) | 0.73 |
| check -> explore (switch 2k) | -0.03 (-0.17, 0.11) | 0.00 (-0.22, 0.23) | 0.61 |
| random + then - | -0.02 (-0.14, 0.09) | -0.21 (-0.41, -0.01) | 0.78 |

Pre-registered test (D-28): frontier expansion would need explore->check to raise feasibility without lowering
originality. Not observed: originality unchanged, feasibility down by about the random-schedule amount (the
feasibility noise floor at this dose is ~0.2). The sustained arms reproduce the trade-off (explore: +0.30 /
-0.28; check: -0.30 / +0.12). The rubric is near its ceiling (unsteered 8.6 and 7.3 of 10), which limits
sensitivity. Verdict: at |alpha| = 2 gap, temporal composition helps Math (accuracy kept, more answers,
+10 pts vs the reverse order) but does not move Idea answers off the originality-feasibility trade-off.
Half-dose replicate and the multi-seed diversity run are still to come.

### Diversity, across-input (sched1, one sample per task, 60 texts per arm and domain; no CIs yet)
Lexical and embedding measures barely move between arms (self-BLEU of reasoning 0.50-0.55, mean cosine
distance of answers 0.48-0.51 everywhere): with one sample per task they measure task variety, not the
arm. Process-level measures do separate the arms. Behavior-transition entropy (bits): Math unsteered 4.06,
sustained explore 4.60, sustained check 3.71, explore->check 4.31, check->explore 3.95, random 4.07; Idea
unsteered 4.73, explore 4.97, check 4.43, explore->check 4.80. The solution-method judge (all 1600 Math
traces parsed) gives a near-identical category mix across arms and 1.1-1.3 approaches tried per trace;
per-input strategy diversity needs the multi-seed run (sched1_div).

### Order effect decomposed with the random schedule (Math, paired, 200 tasks; both doses)

| contrast | accuracy, |alpha| = 2 | accuracy, |alpha| = 1 | answered, 2 | answered, 1 |
|---|---|---|---|---|
| explore->check minus check->explore | +0.100 (0.03, 0.18) | +0.105 (0.03, 0.19) | +0.155 | +0.100 |
| explore->check minus random schedule | -0.005 (-0.08, 0.07) | 0.000 (-0.07, 0.07) | +0.03 | -0.03 |
| check->explore minus random schedule | **-0.105 (-0.18, -0.03)** | **-0.105 (-0.18, -0.04)** | -0.125 | -0.130 |
| random schedule minus unsteered | +0.035 (-0.03, 0.10) | +0.060 (-0.01, 0.13) | +0.04 | +0.07 (0.01, 0.14) |

Corrected reading. The order effect is real and replicates at half dose, but it is the cost of verifying first
and exploring late, not a gain from exploring first and verifying late: explore->check sits exactly on the
matched random schedule at both doses, and the random schedule itself is a few points above unsteered
(fewer loops, more answers within budget), which is a generic-perturbation effect, not a mode effect.
Check->explore loses about 10 points against both comparators at both doses. Half-dose Idea traces under
explore->check are clean (loops and answering equal to unsteered); the half-dose Idea quality judge is pending.

### sched1_d1 complete (|alpha| = 1 gap, same 200 + 200 tasks, same switch rule)
Idea answer quality, paired vs unsteered: explore->check originality +0.07 (-0.03, 0.18), feasibility -0.19
(-0.40, 0.01); check->explore -0.04 / +0.08; random schedule -0.03 / -0.07; sustained explore +0.20 (0.08,
0.32) / -0.26 (-0.47, -0.05); sustained check -0.12 (-0.24, -0.02) / +0.07. Loops: explore->check 0.24 (=
unsteered), sustained explore 0.415, sustained check 0.11. Math accuracy: sustained check +0.015 (ns),
sustained explore -0.01 (ns); ordered arms and random as in the decomposition table above.
Dose replication verdict: (i) the exploratory regime's originality gain and feasibility loss scale with dose
and never decouple; (ii) no schedule expands the Idea frontier at either dose; (iii) on Math, late
exploration after early verification costs ~10 points at either dose, and every other arm equals the random
schedule. Multi-seed diversity run still in progress.
Across-input diversity at half dose matches full dose: self-BLEU 0.51-0.56 and answer cosine distance
0.48-0.51 in every arm; behavior-transition entropy Math: unsteered 4.19, explore 4.36, check 3.77,
explore->check 4.06, check->explore 4.04, random 4.16; Idea: 4.64 / 4.98 / 4.47 / 4.74 / 4.65 / 4.67.
Method judge: 1.1-1.2 approaches per Math trace, 2-5% switched, in every arm.

### sched1_div: per-input diversity (40 tasks per domain x 5 seeds, |alpha| = 2 gap; paired vs unsteered, 95% CI)

Math (reasoning text across the 5 samples of a task):

| arm | self-BLEU (lower = more varied) | Vendi (bge-m3) | methods per task | pass@1 | pass@5 | distinct answers (grader-equivalent) |
|---|---|---|---|---|---|---|
| unsteered (level) | 0.714 | 1.33 | 1.18 | 0.51 | 0.78 | 1.4 of 3.6 answered |
| sustained explore | **-0.029** (-0.043, -0.016) | **+0.043** (0.023, 0.065) | +0.175 (0.00, 0.35) | -0.08 (-0.17, 0.01) | -0.08 (-0.20, 0.05) | -0.13 (ns) |
| sustained check | **+0.022** (0.014, 0.032) | +0.001 | +0.075 (ns) | -0.03 (ns) | -0.03 (ns) | +0.23 (-0.03, 0.50) |
| explore -> check | **-0.028** (-0.040, -0.014) | **+0.042** (0.021, 0.062) | +0.175 (0.00, 0.375) | -0.015 (-0.08, 0.06) | -0.05 (-0.18, 0.08) | +0.08 (ns) |
| check -> explore | +0.010 (0.00, 0.02) | +0.012 (ns) | +0.10 (ns) | -0.06 (-0.15, 0.03) | -0.05 (ns) | -0.25 (-0.50, 0.00) |

Process (behavior-transition entropy, Math): sustained explore +0.52, sustained check -0.47, both ordered arms
about 0 (the halves cancel). Idea: explore->check is the only arm whose five ideas per task become more
distinct from each other (answer cosine distance +0.022 (0.002, 0.041)); the check-first arms make the
reasoning text more varied across seeds (+0.009 to +0.012 cosine) without changing the ideas.

Reading. Exploring first and verifying late gives the exploratory regime's path diversity (as large as
sustained explore on both the lexical and the semantic measure, and a borderline increase in distinct
solution methods) at unsteered accuracy, where sustained explore loses about 8 points of pass@1. The
variety is in the route, not the answer: distinct final answers do not change in any arm. Check-first
schedules make paths more alike. Caveat: 40 tasks; the method-count and pass@k contrasts are not powered.

### Step 0 of the repertoire line: complementarity (sched1_div, 40 tasks x 5 seeds, mc/complementarity.py)

Math. Per-task success rates correlate strongly between arms (r = 0.69-0.83); explore->check is the arm
most correlated with unsteered (r = 0.83). Unique solves (tasks solved at least once by one arm and never
by the other): unsteered 4 vs explore->check 2, and 5 vs 5 against check->explore. Mixed portfolios never
beat a homogeneous unsteered pool at the same K: K = 2, half unsteered + half arm, delta -0.017 to -0.029
(all CIs include 0); K = 4, delta -0.003 to -0.019. The different routes fail on the same problems.

Idea. Coverage of feasible ideas (feasibility >= 7; Vendi of bge-m3 embeddings; 4-sample pools, paired):
2 unsteered + 2 explore->check vs 4 unsteered -0.06 (-0.21, 0.09); + 2 sustained check +0.23 (0.11, 0.35),
which is a feasibility-count effect (check-mode ideas pass the threshold more often); + 2 sustained explore
-0.31.

Verdict: with the current rank-one direction, the extra path diversity is not complementary. It neither
adds unique solves nor extends feasible-idea coverage, so repertoire distillation from this vector is not
motivated. The limit is the controller, not the idea.

### Complementarity, pooled 10 seeds (sched1_div + sched1_div2; 40 Math tasks, unsteered / explore->check / check->explore at 10 samples each)
Mixed portfolio (half unsteered, half arm) vs homogeneous unsteered at the same K: explore->check K = 2
-0.001 (-0.023, 0.022), K = 4 -0.006 (-0.027, 0.014); check->explore K = 2 -0.033 (-0.060, -0.006), K = 4
-0.023 (-0.048, 0.003). Sustained arms: -0.05 to -0.09 (lower pass@1). Unique solves at 10 samples:
unsteered 1 task explore->check never solves, explore->check 1 task unsteered never solves. Decisive: the
schedule-steered routes are not complementary to unsteered ones; a mixed pool is at best equal.

### CoT similarity, between vs within arm, pooled 10 seeds (40 Math tasks; mc/similarity.py)

Per task: mean distance over all pairs within the unsteered set, within the arm, and between the two;
excess = between - mean within (0 = arms indistinguishable); ratio = between / within-unsteered; paired
over tasks, task-bootstrap CI, within-task permutation p. Correct traces only (tasks with >= 2 correct in
both arms), three representations:

| arm | semantic ratio (excess, CI) | process ratio (excess, CI) | lexical ratio | AUROC semantic | AUROC process |
|---|---|---|---|---|---|
| explore -> check | 1.10 (0.0030, 0.0012-0.0049) | 1.01 (0.0026, -0.0017-0.0070; p = 0.03) | 1.013 | 0.55 (0.51-0.60) | 0.58 (0.53-0.64) |
| check -> explore | 1.10 (0.0033, 0.0010-0.0060) | 0.99 (p = 0.38) | 1.015 | 0.51 (0.47-0.56) | 0.52 (0.46-0.58) |
| sustained explore | 1.21 (0.0078, 0.0041-0.0125) | 1.02 (0.019, 0.009-0.030) | 1.028 | 0.67 (0.62-0.74) | 0.68 (0.60-0.76) |
| sustained check | 1.06 (0.0039, 0.0018-0.0062) | 1.02 (p = 0.03) | 1.006 | 0.58 (0.52-0.66) | 0.61 (0.56-0.66) |

AUROC = leave-one-task-out logistic classifier, arm vs unsteered, correct traces (chance 0.5). Semantic =
bge-m3 embedding of the reasoning (cosine); process = behavior transition profile (Jensen-Shannon);
lexical = token 3-gram Jaccard. Using all traces (not only correct) raises every AUROC by 0.03-0.07.

Reading. A successful explore->check trajectory is about 10% farther from a successful unsteered one than
unsteered successes are from each other, and its behavior organisation is the same (process ratio 1.01,
classifier at 0.58). The reverse order is indistinguishable from unsteered among successes. Only sustained
explore produces successes that are recognisably different (AUROC 0.67, semantic ratio 1.21), and those come
with the pass@1 loss. Consistent with the complementarity null: the schedule spreads samples out within the
arm (earlier self-BLEU / Vendi result) without moving the successful ones into new territory. Measurement
note: averaging over all pairs and pooling 10 seeds halves the CI width relative to the 5-seed summary;
the remaining width is between-task variance over 40 tasks.

### D-33 stage 3, first corpus fit (corpus1: 4 domains x 800 traces, ~250k segment states, layers 10-22)

Reduced-rank ridge from the residual state at a judge-segment boundary to the future signature (16 targets:
rates over the remaining segments and over the next 4). Within-domain 5-fold CV by trace (folds mix
domains): mean R^2 0.18-0.19 at every layer, rank 8 > 4 > 1 (0.186 / 0.174 / 0.136 at layer 16), lambda
irrelevant from 10 to 1e4. Per target (layer 16, rank 8): perspective shift 0.42, conflict 0.35,
backtracking (next 4) 0.27, Q&A 0.19, verification 0.10, subgoal 0.09, backward chaining 0.01.
Leave-one-domain-out, uncentered: negative for every held-out domain (-0.1 to -2), driven by level shifts
between domains (the held-out domain's mean signature is not predicted). Whether the within-domain
variation transfers is being measured with a centered LODO R^2 and a domain-demeaned fit (v2, v3).
Reading so far: the state carries the conversational organisation and backtracking of what follows, not
verification; the rank-one explore/check axis we used before spans about a third of what rank 8 captures.

### D-34 pilot, Idea (sigpilot_idea: 40 held-out tasks, layer 16, rank 8, domain-relative archetype targets)

Gap closure (1 - d_steered/d_unsteered over the 8 behavior rates, paired): idea_A3 gain 1 -0.51 (-0.83,
-0.21), gain 2 -0.13 (-0.31, 0.04); idea_A4 gain 1 -0.53, gain 2 -0.26 (-0.52, -0.04); random subspace
-0.21 / -0.20. Selectivity (movement along the target direction / orthogonal) below 1 in every arm,
random included. Behavior shifts: toward A3 (high shift) perspective shift +0.04 (0.007, 0.076) and
conflict +0.025 at gain 1, nothing at gain 2; toward A4 conflict -0.014 at gain 2. Coherence improved:
loops 0.20 -> 0.00 (A4 gain 1), 0.05, 0.10 (A3 gain 2), 0.125 (random gain 1); answers 0.80 -> 1.00.
Diagnosis (mc controller diagnostic on corpus states): the correction was ~2% of the residual norm per
token at gain 1 (4% at gain 2) because the controller ridge (1e-3) was large relative to the ridge-shrunk
readout (W^T W ~ 1e-4), and it closed the readout gap by only 45-55%. The rank-one steering that worked
moved the state by ~13% of its norm (|alpha| = 2 gaps = 7 units against |h| = 54 at layer 19). Verdict:
this pilot tested a controller five times weaker than the reference dose; its signature movement is not
distinguishable from the random subspace, and the loop reduction is a generic-perturbation effect.

### D-34 pilot, Math (sigpilot_math; same under-dosed controller)
Gap closure: math_A2 +0.02 / +0.08 (gain 1 / 2), math_A4 -0.07 / -0.04, random -0.10 / -0.11; all CIs
overlap; selectivity < 1 everywhere. Behavior shifts within +-0.03 (verification -0.026 (-0.054, -0.002)
at A2 gain 2, nothing else clear of zero). Accuracy (40 tasks): A2 +0.10 (-0.08, 0.28) at both gains,
A4 -0.125 (-0.25, -0.03) at gain 1 and -0.05 at gain 2, random 0.00 / +0.025; answers: A4 -0.125 at
gain 1. Same verdict as Idea: at 2-4% of the residual norm the controller does not move the signature
beyond the random subspace; the accuracy hints (A2 up, A4 down) are worth watching in the calibrated
pilot but are not powered.

### D-35 pilot, Idea (sigpilot2_idea: calibrated rank-8 controller, 40 held-out tasks)

Two controllers: closed-loop (per-token correction toward the archetype's domain-relative signature,
capped at 25% of the residual norm) and open-loop (a constant direction G(phi_archetype - phi_domain),
norm 4.5 / 6.75 units at gains 2 / 3), plus norm-matched random subspaces.

| arm | shift | conflict | verif. | Idea-likeness | loops | answered | fraction of target gap covered |
|---|---|---|---|---|---|---|---|
| open-loop -> A3 (high shift), g2 / g3 | +0.084 / +0.155 | +0.042 / +0.070 | -0.012 / +0.006 | +5.1 / +9.6 | 0.75 / 0.83 | 0.23 / 0.05 | 0.88 / 2.0 |
| open-loop -> A4 (low shift), g2 / g3 | -0.03 / -0.05 | -0.013 / -0.023 | +0.011 / -0.005 | -3.6 / -4.4 | 0.00 / 0.00 | 1.00 / 1.00 | 0.62 / 0.85 |
| closed-loop -> A3, g2 / g3 | +0.045 / +0.067 | -0.002 / +0.027 | -0.005 / +0.023 | +1.6 / +3.5 | 0.10 / 0.20 | 0.90 / 0.80 | 0.29 / 1.23 |
| closed-loop -> A4, g2 / g3 | -0.003 / -0.011 | -0.013 / -0.010 | ~0 | -0.4 / -1.2 | 0.03 / 0.05 | 0.98 / 0.95 | 0.13 / 0.20 |
| random subspace, open / closed | +0.02 / ~0 | +0.017 / +0.009 | +0.023 / +0.003 | +1.8 / +1.4 | 0.15-0.30 / 0.23 | 0.7-0.8 | 0.5-0.7 / 0.2-0.3 |

(unsteered: shift 0.29, conflict 0.124, verification 0.115, loops 0.20, answered 0.80; 95% CIs in
analysis/paired_shifts.csv and sigctrl_eval.csv.)

Reading. The calibrated controller moves the signature with dose response and in the intended
direction for both targets, and the open-loop constant direction is as strong as the rank-one sustained
steering was (A3: +9.6 Idea-likeness at 6.75 units vs +11 for sustained explore at 7 units; A4: -4.4 vs
-4 to -9 for sustained check). But the movement is the same axis: the 8-behavior shift vectors have
cosine 0.73-0.87 with the rank-one explore/check shift vector, and the component orthogonal to that axis
is 0.02-0.07 in rate units with no target-specific sign (verification stays within +-0.02 for every
arm, including the A3 target whose gap asked for +0.03). Gap closure against the medoid is negative
because the medoids sit within 0.03 of the domain mean on most behaviors and the doses overshoot them;
fraction of gap covered along the target direction is the usable metric (0.6-0.9 at gain 2, overshoot at
gain 3). The random open-loop subspace also "covers" 0.5-0.7 of the A3 gap because a generic
perturbation itself pushes Idea traces toward more shifting and looping.
Verdict so far: on Idea, the rank-8 readout buys no control beyond the explore/check axis; the
archetypes differ from the mean mostly along that axis and the controller cannot move verification or
backtracking independently of it. Math (math_A4 = late-verification archetype) pending.
Idea answer quality under the calibrated controller (paired vs unsteered, tasks answered in both):
closed-loop -> A3 originality -0.32 (-0.68, 0.00) / -0.54 (-0.81, -0.27) at gains 2 / 3, feasibility
-0.04 / +0.12; closed-loop -> A4 originality -0.19 / -0.37 (-0.67, -0.07), feasibility +0.23 / -0.13;
open-loop -> A4 originality -0.12 / -0.06, feasibility -0.22 / -0.50 (-0.91, -0.03); open-loop -> A3
answered too rarely to score (0.23 / 0.05); random arms -0.03 to -0.26 / -0.30 to +0.12. No arm improves
either axis; the sustained explore/check trade-off is not recovered here because the medoid targets are
small displacements, and the strong open-loop arms either loop (A3) or lose feasibility (A4).

### D-35 pilot, Math (sigpilot2_math: calibrated rank-8 controller, 40 held-out tasks)

Targets: math_A2 (gap from unsteered: subgoal +0.147, verification -0.047, Q&A -0.053, conflict -0.036)
and math_A4 (gap within +-0.025 on every behavior, i.e. no real target). The A2 direction is nearly
orthogonal to the rank-one explore/check axis (cosine 0.13-0.26 with its shift vector).

| arm | subgoal | verification | Q&A | gap covered | loops | answered | tokens | accuracy (vs 0.425) |
|---|---|---|---|---|---|---|---|---|
| closed-loop -> A2, g2 | +0.089 (0.065, 0.113) | -0.041 (-0.068, -0.014) | -0.029 | 0.60 (0.44, ..) | 0.00 | 0.875 | 5767 | 0.65, +0.225 (0.075, 0.40) |
| closed-loop -> A2, g3 | +0.081 | -0.034 | -0.021 | 0.53 | 0.05 | 0.85 | 5939 | 0.65, +0.225 (0.05, 0.40) |
| open-loop -> A2, g2 / g3 | +0.059 / +0.067 | -0.025 / -0.036 | -0.013 / -0.021 | 0.38 / 0.46 | 0.05 / 0.03 | 0.80 / 0.85 | 5867 / 5630 | 0.625 / 0.65 |
| random closed g2 / g3 | ~0 | -0.021 / -0.009 | | -0.34 / -0.18 | 0.08 / 0.00 | 0.65 / 0.70 | 7334 / 6903 | 0.45 / 0.55 |
| random open g2 / g3 | ~0 | -0.010 / -0.020 | | -0.03 / -0.51 | 0.00 / 0.00 | 0.65 / 0.775 | 6938 / 6071 | 0.50 / 0.60 |
| -> A4 closed g2 / g3 | | +0.005 / +0.006 | | 0.78 / 0.64 (tiny gap) | 0.13 / 0.18 | 0.43 / 0.48 | 9192 / 9245 | 0.325 / 0.35 |

Paired against the matched random arm: closed-loop A2 g2 accuracy +0.20 (0.075, 0.35), answered +0.225
(0.075, 0.375); g3 +0.10 (-0.075, 0.25); open-loop g2 +0.125 (-0.05, 0.30), g3 +0.05 (-0.125, 0.225).
Accuracy among answered traces is unchanged in every arm (0.74-0.78 vs 0.77 unsteered): the gain is
traces finishing within the 12k budget. The controller moves subgoal setting up and verification/Q&A
down together, a combination the rank-one axis cannot produce (its Math shift vector is +0.01 on both),
and the readout's off-axis component is target-specific here (orthogonal residual 0.06-0.09 in the
intended signs). Random subspaces also shorten traces and lift answering a little, which is why the
random comparator matters.
Reading: on Math the rank-8 readout gives control that the rank-one axis does not (subgoal-structured,
less verification-heavy reasoning), and at gain 2 closed-loop it turns into +20 points of pass@1 at a
fixed budget on 40 held-out tasks, beyond the matched random subspace. Needs the powered replication
(D-36) before it is a claim; the A4 (late-verification) target has no usable gap and is dropped.

### D-36 replication: Math A2 controller on 199 held-out tasks (sigpilot3_math)

| arm | accuracy | vs unsteered (paired) | answered | loops | tokens | subgoal | verification | Q&A |
|---|---|---|---|---|---|---|---|---|
| unsteered | 0.412 | | 0.62 | 0.055 | 8315 | | | |
| closed-loop -> A2, gain 2 | **0.628** | **+0.216 (0.141, 0.296)** | 0.88 (+0.26) | 0.015 | 5756 (-2559) | +0.082 (0.072, 0.091) | -0.022 (-0.033, -0.011) | -0.033 |
| open-loop -> A2, gain 3 | **0.623** | **+0.211 (0.126, 0.291)** | 0.86 (+0.24) | 0.010 | 5633 | +0.075 | -0.020 | -0.025 |
| random subspace, closed g2 | 0.317 | -0.095 (-0.166, -0.025) | 0.49 (-0.13) | 0.111 | 8830 | ~0 | +0.033 | |
| random subspace, open g3 | 0.322 | -0.090 (-0.171, -0.010) | 0.49 (-0.14) | 0.101 | 8915 | ~0 | +0.028 | |
| rank-one sustained check (|alpha| = 2 gaps, L19) | 0.467 | +0.055 (-0.015, 0.131) | 0.68 (+0.06) | 0.065 | 7546 | +0.009 | -0.015 | -0.024 |

Pre-registered primary: closed-loop A2 minus its random subspace, paired: accuracy **+0.312 (0.231,
0.392)**, answered +0.392 (0.322, 0.467). Open-loop A2 minus its random: +0.302 (0.221, 0.382). A2 minus
the rank-one check arm: +0.161 (0.080, 0.236). Accuracy among answered traces: 0.71 / 0.72 (A2) vs 0.66
(unsteered) vs 0.64 / 0.66 (random) vs 0.68 (rank-one). Unique solves (one sample each): A2 solves 56-59
tasks unsteered does not, unsteered solves 13-17 that A2 does not; union 0.69-0.71 vs 0.41. Fraction of
the A2 gap covered 0.51 (0.45, 0.56) and 0.47; random subspaces cover -0.17. Shift-vector cosine with the
rank-one axis 0.18-0.22.

Reading. On held-out MATH tasks, steering along a readout-derived direction that raises subgoal setting
and lowers verification and Q&A lifts pass@1 at a fixed 12k budget from 0.41 to 0.63. Most of the gain
is traces finishing within budget (2.6k tokens shorter, loops 5.5% -> 1%), with a further 5-6 points of
conditional accuracy. The norm-matched random subspace at the same dose is harmful (-9 points, more loops,
longer traces), so this is not a generic perturbation effect, and the rank-one check direction at its
reference dose gives a quarter of the gain and one tenth of the subgoal shift, so it is not the explore /
check axis either. Same model, judge and labels as everything above; one seed; the readout was fit on
four domains including Math, and the tasks are held out from both the readout corpus and the pilots.

### D-37: A2 controller with 5 seeds on the 40 sched1_div Math tasks (sigdiv_math pooled with sched1_div)
pass@1 0.70 vs 0.51 unsteered (tokens 4582 vs 6686); solved at least once in 5: 34/40 vs 31/40; A2 solves
5 tasks unsteered never does and 4 the rank-one check arm never does, the reverse 2 and 0. Error
correlation with unsteered r = 0.77 (sample-level phi 0.52). Portfolios at K = 2: unsteered x2 0.665,
half unsteered + half A2 0.748 (+0.083, 0.029-0.154), A2 x2 0.780; at K = 4: 0.755 / 0.812 / 0.835.
Reading: A2 dominates rather than complements. The mixed pool beats all-unsteered but not all-A2, and
errors stay correlated; A2 is a faster route to mostly the same solutions plus a few extra ones, not a
different solver. For post-training this means A2 traces are a better single source, not a repertoire.
Caveat for D-37 only: 23 of these 40 Math tasks are among the 400 prompts the readout was fit on (the
sched1_div task sample predates the corpus). The headline replication (sigpilot3_math, 199 tasks) and the
pilot (sigpilot2_math) have zero overlap with the readout corpus.
Similarity of correct traces (pooled 5 seeds, 27 tasks with >= 2 correct in both arms): a correct A2 trace
is 29% farther from a correct unsteered trace than unsteered successes are from each other (semantic
excess 0.017, CI 0.012-0.023), 14% in behaviour organisation (process excess 0.020, 0.008-0.034), 3%
lexically; leave-one-task-out AUROC 0.65 (semantic) / 0.69 (process) against 0.53-0.57 for the schedule
and rank-one arms. A2 successes are a recognisably different regime even though they mostly solve the
same problems. Slides: board G (controller) added to the canvas.
Per-input diversity of the A2 arm across 5 seeds (paired vs unsteered, 40 tasks): pass@1 +0.19 (0.12, 0.28),
pass@5 +0.075 (-0.05, 0.20); reasoning-path variety slightly lower (Vendi -0.032 (-0.062, -0.006), cosine
distance -0.007, self-BLEU -0.007 ns); behaviour-transition entropy -0.25 (-0.35, -0.14); distinct
behaviour 3-grams +0.04 (0.01, 0.07). A2 is a more stereotyped regime: each task is solved the same
structured way across seeds, which is why it dominates unsteered at pass@1 but adds little at pass@5.

### D-38 / D-39: the 2x2 of direction x task domain (same readout, dose, controls; one seed)

Math A2 direction on Idea tasks (sigx_idea; 104 held-out Idea tasks, all that remain outside the readout
corpus and the pilots):

| arm | subgoal | shift | conflict | backtracking | loops | answered | tokens | originality | feasibility |
|---|---|---|---|---|---|---|---|---|---|
| unsteered | | | | | 0.22 | 0.78 | 7385 | 8.63 | 7.14 |
| same vector as on Math (open-loop, g3) | +6.1 | +0.3 | -0.1 | -1.9 | 0.00 | 1.00 | 3015 | -0.25 (-0.42, -0.09) | -0.35 (-0.62, -0.07) |
| absolute A2 profile (closed-loop, g2) | +14.4 | -4.2 | -4.6 | -8.6 | 0.00 | 1.00 | 1624 | -0.79 (-1.01, -0.59) | -0.09 (-0.38, 0.21) |
| random open g3 / closed g2 | -0.9 / -1.5 | +12.6 / +1.4 | +17.7 / +6.6 | +11.0 / +2.7 | 0.89 / 0.66 | 0.04 / 0.34 | 12017 / 9831 | n<5 / +0.24 | n<5 / -0.52 |

(shifts in rate points x 100, paired; the same-vector arm's shift vector has cosine 0.80 with the shift the
same vector produced on Math.)

Idea directions on Math tasks (sigx_math; the 199 held-out tasks of the A2 replication):

| arm | subgoal | verification | Q&A | shift | conflict | loops | answered | tokens | accuracy | vs matched random |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | | | | | | 0.055 | 0.62 | 8315 | 0.412 | |
| idea_A4 (low shift) same vector, g3 | +6.1 | -3.3 | -1.9 | -0.1 | -1.4 | 0.01 | 0.82 | 6481 | 0.523, +0.111 (0.04, 0.19) | **+0.20 (0.13, 0.28)** |
| idea_A3 (high shift) same vector, g3 | +2.7 | -1.0 | -0.1 | +0.9 | -0.5 | 0.23 | 0.46 | 8910 | 0.271, -0.141 | -0.05 (-0.13, 0.03) |
| absolute idea_A3 profile, closed-loop g2 | -6.0 | +0.9 | -0.5 | +2.5 | +7.1 | 0.48 | 0.03 | 12156 | 0.025 | -0.30 |
| random open g3 / closed g2 | 0 / -2.2 | +1.0 / -0.6 | | +1.4 / +0.6 | +0.6 / +2.0 | 0.12 / 0.09 | 0.50 / 0.47 | | 0.322 / 0.322 | |

(cosine of the same-vector shifts with their Idea-domain effect: A4 0.55, A3 0.26.)

The 2x2 in one line each:
- Math A2 -> Math (D-36): +0.22 accuracy; structured, shorter, coherent.
- Math A2 -> Idea: the same organisation appears (subgoals up, cosine 0.80), traces become short and never
  loop, but the ideas lose originality (-0.25 same vector, -0.79 absolute profile) and gain no feasibility.
  Unlike the rank-one "check" direction (originality -0.30, feasibility +0.12), this direction buys nothing
  on Idea: it is a Math-good, Idea-bad direction.
- Idea A4 (low shift) -> Math: +0.11 accuracy (+0.20 vs random) through the same functional change as A2
  (subgoal up, verification and Q&A down) at about two thirds of A2's strength. Idea A3 (high shift) -> Math:
  -0.14, and forcing the absolute A3 profile onto Math collapses it (48% loops, 3% answered).
- Idea A3/A4 -> Idea (D-35): the explore/check axis only; no quality gain.
Regularity: on Math every helpful direction, whatever its origin, is "more subgoal structure, less
verification and Q&A", and every harmful one is exploratory; on Idea, structure-inducing directions
always cost originality. Random subspaces at these norms are destructive on Idea (0.66-0.89 loops) and
mildly harmful on Math (-0.09), so none of the cross-domain effects is a generic-perturbation artefact.

## P1 and P2 (2026-09-24): is A2 only budget, is it SEAL, and does it carry to other domains?

Setup (D-40 to D-42): fixed steering vectors run in vLLM (`mc/vsteer.py`, checked against the HF path), budget 32k,
accuracy at smaller budgets read by truncating each trace (`mc/budget.py`). Unanswered counts as wrong. Paired over
tasks after averaging the 2 seeds; 95% task-bootstrap CI. Math: the 199 held-out tasks of D-36. GPQA and planning:
200 tasks each outside the readout corpus. All arms at layer 16, norm 6.75, every position, unless noted.

### Answers
1. The signature directions buy efficiency, not better reasoning. Gains are large at tight budgets and shrink as the
   budget grows. At 32k they are zero on GPQA and planning. On Math +0.09 remains at 32k, and it comes from traces
   that never finish without steering (unsteered: 12.6% hit the 32k cap, 9.3% loop; A2: 2.0% and 0.5%). Accuracy
   among answered traces is the same or a little lower in every domain.
2. At equal accuracy they use a third to a half of the tokens. Math: A2 at a 12k cap matches unsteered at 32k (0.63
   vs 0.63) with 5.6k mean tokens against 13.2k. GPQA: gpqa_A4 matches unsteered at 32k (0.74 vs 0.73) with 5.2k
   against 16.0k. Planning: planning_A2 0.90 vs 0.89 with 4.0k against 10.2k.
3. A2 is not SEAL. SEAL run as published (their vector recipe and position rule, layer picked on a validation split)
   gives +0.03 at 12k on held-out Math, not significant; its +0.14 on the validation split looks like a winner's curse
   from picking the best of four layers. SEAL's direction at A2's layer and dose gives +0.13; A2 beats it by +0.11
   (0.05, 0.16) at 12k and +0.06 (0.00, 0.11) at 32k. The two directions are nearly orthogonal (cosine 0.09).
   A plain "Be concise." instruction gives +0.06 at 12k (not significant) and raises loops to 16%.
4. The Math direction carries to other convergent domains, and each domain's own archetype does better. At 12k: A2
   +0.23 on GPQA and +0.16 on planning; own archetype +0.36 (gpqa_A4) and +0.19 (planning_A2). Signs and order
   match the D-42 predictions written before the runs (GPQA: A2 +0.35 predicted, +0.23 seen; gpqa_A4 +0.58, +0.36;
   planning: A2 +0.10, +0.16; planning_A2 +0.23, +0.19). The linear model overstates GPQA's sizes.
5. A random direction of the same norm hurts everywhere (-0.07 to -0.23) and raises loops (15-39%). None of the
   gains are generic perturbation effects.

### Math (199 tasks x 2 seeds, run p1_budget)
| arm | 4k | 8k | 12k | 16k | 24k | 32k | answered 32k | correct among answered 32k | mean tokens 32k | loops |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | 0.16 | 0.31 | 0.39 | 0.49 | 0.58 | 0.63 | 0.88 | 0.71 | 13.2k | 9.3% |
| A2 | 0.34 | 0.57 | 0.63 | 0.68 | 0.70 | 0.71 | 0.98 | 0.72 | 6.7k | 0.5% |
| SEAL (layer 26, delimiters) | 0.18 | 0.33 | 0.42 | 0.51 | 0.60 | 0.64 | 0.89 | 0.72 | 12.3k | 9.0% |
| SEAL direction at A2's dose | 0.28 | 0.43 | 0.53 | 0.60 | 0.65 | 0.66 | 0.94 | 0.70 | 8.8k | 5.3% |
| "Be concise." | 0.22 | 0.35 | 0.45 | 0.51 | 0.58 | 0.61 | 0.83 | 0.73 | 12.8k | 16.1% |
| random direction | 0.15 | 0.28 | 0.32 | 0.40 | 0.50 | 0.54 | 0.80 | 0.68 | 15.2k | 15.8% |
Paired vs unsteered: A2 +0.24 (0.18, 0.30) at 12k, +0.09 (0.03, 0.15) at 32k. The vLLM unsteered arm at 12k (0.39)
matches the HF run on the same tasks (0.41), and A2 at 12k (+0.24) matches D-36 (+0.22).

### GPQA (200 tasks x 2 seeds, run p2_gpqa)
| arm | 4k | 8k | 12k | 16k | 24k | 32k | answered 32k | correct among answered 32k | mean tokens 32k | loops |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | 0.04 | 0.20 | 0.35 | 0.48 | 0.67 | 0.73 | 0.91 | 0.80 | 16.0k | 7.0% |
| A2 (Math vector) | 0.17 | 0.44 | 0.57 | 0.63 | 0.67 | 0.69 | 0.98 | 0.71 | 9.8k | 0.3% |
| gpqa_A4 (own archetype) | 0.39 | 0.64 | 0.71 | 0.72 | 0.73 | 0.74 | 1.00 | 0.74 | 5.2k | 0% |
| random direction | 0.04 | 0.12 | 0.24 | 0.34 | 0.46 | 0.49 | 0.59 | 0.84 | 21.6k | 39% |
Paired vs unsteered at 12k: A2 +0.23 (0.17, 0.28), gpqa_A4 +0.36 (0.31, 0.42), random -0.11 (-0.15, -0.06);
at 32k: A2 -0.04 (-0.08, 0.01), gpqa_A4 +0.01 (-0.03, 0.05), random -0.23.

### Planning (200 tasks x 2 seeds, run p2_planning)
| arm | 4k | 8k | 12k | 16k | 24k | 32k | answered 32k | correct among answered 32k | mean tokens 32k | loops |
|---|---|---|---|---|---|---|---|---|---|---|
| unsteered | 0.13 | 0.42 | 0.71 | 0.83 | 0.88 | 0.89 | 0.97 | 0.92 | 10.2k | 3.3% |
| A2 (Math vector) | 0.32 | 0.78 | 0.87 | 0.89 | 0.90 | 0.90 | 1.00 | 0.91 | 6.1k | 0.5% |
| planning_A2 (own archetype) | 0.59 | 0.86 | 0.89 | 0.90 | 0.90 | 0.90 | 1.00 | 0.90 | 4.0k | 0% |
| random direction | 0.12 | 0.36 | 0.62 | 0.75 | 0.79 | 0.80 | 0.86 | 0.93 | 13.0k | 14.5% |
Paired vs unsteered at 12k: A2 +0.16 (0.11, 0.21), planning_A2 +0.19 (0.14, 0.24), random -0.09; at 32k: A2 +0.02,
planning_A2 +0.02 (both within noise), random -0.09.

### What this changes in the story
The controllable part of the signature, on convergent tasks, is how efficiently the model gets to an answer: fewer
Q&A, conflict and backtracking turns, more subgoal structure, far fewer runaway traces. With a generous budget it
does not make answers more correct. It beats the published efficiency steering method (SEAL) and a concise prompt
at the same dose. On Idea tasks the same kind of direction costs originality (D-38). Pending: Track B on P1 seed 1
(do SEAL and A2 move the same behaviours?), and whether A2 moves GPQA and planning behaviours the way it moves Math.

### P1 behaviour shifts (Track B on seed 1, 199 Math tasks per arm, 8k window; rate points x 100, paired vs unsteered)
| behaviour (unsteered rate) | A2 | SEAL as published | SEAL direction at A2's dose | "Be concise." | random |
|---|---|---|---|---|---|
| subgoal (9.6) | +7.4 (6.4, 8.3) | +0.5 | +5.5 (4.3, 6.6) | +0.3 | -1.4 |
| verification (22.2) | -3.3 (-4.4, -2.2) | -1.0 | -4.8 (-5.8, -3.8) | -3.5 | +2.0 |
| Q&A (9.7) | -2.9 (-3.7, -2.1) | -0.3 | -4.6 (-5.4, -3.7) | -1.4 | +1.3 |
| reconciliation (4.5) | -1.4 | -0.5 | -1.9 | -0.3 | +0.4 |
| conflict (3.8) | -0.9 | +0.1 | -1.7 | +0.1 | +1.4 |
| perspective shift (12.7) | +0.8 | -0.5 | -0.3 | +2.2 | +0.6 |
| backtracking (1.5) | 0.0 | +0.4 | -0.4 | +0.7 | +0.6 |
Cosine with A2's shift vector (and size relative to A2): SEAL 0.64 (0.16), SEAL direction at A2's dose 0.93 (1.03),
concise 0.51 (0.51), random -0.78 (0.37). The vLLM A2 shift matches the HF A2 shift of D-36 (cosine 0.99).
Reading: published SEAL barely moves behaviour on this model, which fits its null accuracy effect. Given A2's dose, the
SEAL direction produces nearly the same behaviour change as A2 (more subgoals, less verification and Q&A) even though
the two residual directions are nearly orthogonal (cosine 0.09). A2 differs in the mix (more subgoal setting, less
suppression of verification) and in loops (0.5% vs 5.3%), and that goes with its extra +0.11 at 12k. The concise
prompt cuts verification but adds no subgoal structure and raises loops. So the behaviour profile that pays off is
"more subgoal structure", not "less checking" alone, and several residual directions can reach it.

## D-43 behaviour check, GPQA (Track B, seed 1, 200 tasks per arm)
Paired behaviour shifts vs unsteered (rate points x 100; unsteered rates in brackets):
| behaviour | A2 (Math vector) | gpqa_A4 |
|---|---|---|
| Q&A (11.0) | -5.1 | -7.9 |
| subgoal (5.2) | +4.0 | +5.0 |
| conflict (7.4) | -1.6 | -1.7 |
| verification (17.0) | -1.5 | +0.3 |
| perspective shift (14.9) | -0.6 | -0.9 |
| backtracking (2.9) | -0.5 | -0.9 |
The Math vector moves GPQA the way it moves Math (shift cosine 0.84, size 0.79). gpqa_A4 cuts Q&A harder and leaves
verification alone, which is the mix GPQA's within-task success pattern favours (verification +0.29, Q&A -0.21).
Prediction check with the observed GPQA shifts through the GPQA regression: A2 +0.30 (0.17, 0.46) predicted, +0.23
observed at 12k; gpqa_A4 +0.54 (0.36, 0.74) predicted, +0.36 observed. Order right; the linear model runs about 1.3-1.5x
high. Planning pending.

## D-44 P3: can steering produce different solution methods?
Step 1, strategy prompts (run p3_strat; 199 held-out Math tasks x 4 seeds, 16k budget; method judge as in D-30):
| arm | pass@1 | error corr. with unsteered | unique solves (arm / unsteered) | samples off the task's usual method | mixed pass@4 (unsteered alone 0.714) | arm alone pass@4 |
|---|---|---|---|---|---|---|
| unsteered | 0.495 | | | 7% | | 0.714 |
| "work backwards" | 0.574 (+0.08, 0.03-0.12) | 0.63 | 25 / 11 | 17% | 0.779 | 0.784 |
| "draw a picture" | 0.460 (-0.035) | 0.52 | 19 / 17 | 31% | 0.757 (+0.044, 0.008-0.081) | 0.724 |
| "small cases" | 0.489 (-0.01) | 0.64 | 13 / 13 | 21% | 0.736 (+0.02, ns) | 0.714 |
Method mix: "draw a picture" raises geometric/coordinate arguments from 7% to 22% of samples; "small cases" raises
pattern finding from 1% to 10%. Two unsteered plus two picture samples cover 1.50 methods per task, against 1.24 for four
unsteered. So an instruction can change the method, and "draw a picture" gives the first complementary solves in this
project (a mixed pool beats either pool alone); "work backwards" instead dominates, like A2.

Step 2, steering along the picture prompt's direction without the prompt (run p3_steer; same tasks and seeds). The
direction is the D-04 recipe: same tokens teacher-forced with and without the instruction, layer-16 difference over
the first 64 generated tokens (norm 1.5; per-trace consistency 0.71; cosine 0.68 with the small-cases direction, so part
of it is a generic "there is an instruction" component; cosine -0.08 with A2).
| arm | pass@1 | loops | mixed pass@4 (unsteered alone 0.698) |
|---|---|---|---|
| unsteered | 0.487 | 8% | |
| picture direction x2 (norm 3.1) | 0.386 | 16% | 0.693 |
| picture direction x4 (norm 6.2) | 0.291 | 36% | 0.694 |
| random, norm 6.2 | 0.521 | 8% | 0.720 |
The prompt's complementarity does not carry over to its direction: steering along it hurts accuracy, raises loops, and
adds no solves to a mixed pool, while a random vector of the same norm is harmless. Method judge on p3_steer pending
(does the method mix move at all?).
Method judge on p3_steer: steering along the picture direction barely changes the method mix (geometric/coordinate
arguments 8.2% unsteered, 9.2% at x4; samples off the task's usual method 8% unsteered, 12% at x2, 16% at x4, 11% for the
random vector). Verdict for D-44: an instruction can change the solution method and add complementary solves, but the
instruction's prompt-contrast direction does not carry that change; steering along it mostly degrades generation.

## D-45 P4a: does RL on a steering bias find A2? (run rlbias/zero8k, evaluation run p4_eval)
Training (60 steps x 16 problems x 8 samples, 8k budget, Dr. GRPO, Adam 2e-3, b from zero): reward 0.45 -> 0.53 (steps
1-10 vs 51-60), answered 0.57 -> 0.69, tokens 5.7k -> 5.0k; |b| 4.4 at step 60; heading stable (cosine 0.90 between b at
step 20 and step 60); raw cosine with A2 0.035.
Held-out evaluation (199 tasks x 2 seeds, same sampling seeds as p1_budget; paired vs unsteered):
| arm | 8k | 12k | 32k | answered 32k | mean tokens 32k | loops |
|---|---|---|---|---|---|---|
| unsteered | 0.31 | 0.39 | 0.63 | 0.88 | 13.2k | 9.3% |
| A2 | 0.57 | 0.63 | 0.71 | 0.98 | 6.7k | 0.5% |
| RL b, step 20 | 0.32 | 0.37 | 0.62 | 0.88 | 13.2k | 6.0% |
| RL b, step 40 | 0.34 | 0.38 | 0.57 | 0.80 | 14.2k | 13.6% |
| RL b, step 60 | 0.35 (+0.04, 0.00-0.08) | 0.41 (+0.02) | 0.58 (-0.05, -0.11-0.00) | 0.80 | 13.1k | 9.8% |
A2 minus RL b (step 60): +0.22 at 8k and 12k, +0.14 at 32k. Reading against D-45: the learned bias helps a little at the
budget it was trained on and not beyond, does not cut tokens or loops, and points elsewhere. Bias-only RL from zero at this
scale does not find A2. Next (D-45): the same RL started from A2, to tell a search failure (RL keeps A2 when started
there) from an objective that does not favour A2 (RL moves away).

D-45 correction (2026-09-26): the RL runs zero8k and a2init8k computed token log-probs at temperature 1 while rollouts
were sampled at temperature 0.6 (top-p 0.95, top-k 20). TRL and verl divide logits by the sampling temperature; without
it the update is not the policy gradient of the sampled policy. Symptom that exposed it: started at A2, reward fell
steadily (0.65 at steps 1-10, 0.41 at steps 51-60) while b drifted in the same direction the zero-start run took (cosine
0.83 between b60 - A2 and the zero-start b60). Both runs and the p4_eval numbers from zero8k are withdrawn as evidence.
Fixed in mc/rlbias.py (logits / 0.6; config records logprob_temperature). Rerun order: a2init8k_T first as a sanity
check (with a correct gradient, RL started at A2 should not steadily lose reward), then zero8k_T.

## D-43 behaviour check, planning (Track B, seed 1, 200 tasks per arm)
| behaviour (unsteered rate) | A2 (Math vector) | planning_A2 |
|---|---|---|
| Q&A (13.6) | -4.8 | -5.9 |
| subgoal (3.7) | +2.6 | +1.5 |
| verification (22.7) | -0.9 | 0.0 |
| reconciliation (3.7) | -0.9 | -0.5 |
| conflict (3.5) | -0.7 | -0.1 |
A2's planning shift has cosine 0.76 with its Math shift (size 0.64); planning_A2 is nearly the same shift (cosine 0.95)
with more Q&A cut and less subgoal. Prediction check with the observed shifts: A2 +0.11 (-0.05, 0.31) predicted, +0.16
observed at 12k; planning_A2 +0.08 (-0.09, 0.27) predicted, +0.19 observed. Planning's within-task model (47 tasks with
varying outcome) is too weak to predict sizes; the GPQA model got order and rough size.
Across the three domains the common core of the efficient profile is fewer self-questioning (Q&A) turns and more
subgoal setting; verification is cut on Math and kept on GPQA and planning by the own-domain archetypes.

D-45 fixed-gradient sanity check (a2init8k_T, 51 of 60 steps): started at A2, reward holds (0.66, 0.71, 0.69, 0.71, 0.70
over steps 1-50 in blocks of 10; the buggy run fell to 0.45 by steps 41-50), answered 0.88 -> 0.91, tokens 3.9k -> 3.5k,
b stays near A2 (cosine 0.97, norm 6.8 -> 7.0). A2 is a stable point of this objective. Zero-start rerun zero8k_T running.

D-45, RL-refined A2 (a2init8k_T, fixed gradient; training reward 0.66 -> 0.75, tokens 3.9k -> 3.0k over 60 steps, cosine
with A2 0.96, norm 6.75 -> 7.05). Held-out evaluation (run p4_eval2; 199 tasks x 2 seeds, same seeds as p1_budget):
| arm | 4k | 8k | 12k | 32k | answered 32k | mean tokens 32k | loops |
|---|---|---|---|---|---|---|---|
| unsteered | 0.16 | 0.31 | 0.39 | 0.63 | 0.88 | 13.2k | 9.3% |
| A2 | 0.34 | 0.57 | 0.63 | 0.71 | 0.98 | 6.7k | 0.5% |
| A2 + 60 RL steps | 0.47 | 0.63 | 0.68 | 0.72 | 1.00 | 4.7k | 0% |
Paired vs A2: +0.13 (0.09, 0.18) at 4k, +0.06 (0.02, 0.10) at 8k, +0.05 (0.01, 0.10) at 12k, +0.01 (-0.03, 0.05) at 32k,
2.0k fewer tokens at 32k. RL on a budgeted correctness reward keeps A2's heading and makes it more efficient; it does not
raise accuracy when the budget is generous. Pending: the zero-start rerun (zero8k_T).

D-45, zero-start RL with the fixed gradient (zero8k_T): training reward 0.50 -> 0.63 (steps 1-10 vs 51-60), answered 0.65
-> 0.84, tokens 5.4k -> 4.1k; |b| 2.36 at step 60; raw cosine with A2 -0.015. Held-out evaluation (run p4_eval3; 199
tasks x 2 seeds, same seeds as p1_budget):
| arm | 4k | 8k | 12k | 32k | answered 32k | mean tokens 32k | loops | norm | cosine with A2 |
|---|---|---|---|---|---|---|---|---|---|
| unsteered | 0.16 | 0.31 | 0.39 | 0.63 | 0.88 | 13.2k | 9.3% | | |
| A2 | 0.34 | 0.57 | 0.63 | 0.71 | 0.98 | 6.7k | 0.5% | 6.75 | 1 |
| RL from zero, step 30 | 0.28 | 0.44 | 0.54 | 0.66 | 0.98 | 7.9k | 0% | 1.75 | |
| RL from zero, step 60 | 0.32 | 0.50 | 0.59 | 0.72 | 0.98 | 7.1k | 0% | 2.36 | -0.015 |
| A2 + 60 RL steps | 0.47 | 0.63 | 0.68 | 0.72 | 1.00 | 4.7k | 0% | 7.05 | 0.96 |
RL from zero vs unsteered: +0.20 (0.14, 0.25) at 12k, +0.10 (0.04, 0.15) at 32k. Vs A2: -0.02 at 4k, -0.07 (-0.13, -0.01)
at 8k, -0.04 at 12k (ns), +0.01 at 32k. So bias-only RL on a budgeted correctness reward finds a vector that performs
close to A2, at a third of A2's norm and orthogonal to it in residual space. Whether it produces A2's behaviour profile
is the pre-registered test; Track B on both RL vectors (seed 1) is running.
Behaviour of the RL-refined A2 (Track B seed 1, vs the p1_budget unsteered traces on the same tasks and seed; points x 100):
subgoal +10.3 (A2 +7.4), verification -3.5 (-3.3), Q&A -2.6 (-2.9), reconciliation -1.8 (-1.4), conflict -1.4 (-0.9),
perspective shift +1.7 (+0.8). Shift cosine with A2 0.99, size 1.32x: RL keeps A2's profile and turns up its subgoal part.

Behaviour of the zero-start RL bias (Track B seed 1, vs the p1_budget unsteered traces; points x 100, 95% CI):
| behaviour | A2 | RL from zero (step 60) |
|---|---|---|
| subgoal | +7.4 (6.4, 8.3) | +2.9 (2.0, 3.8) |
| Q&A | -2.9 (-3.7, -2.1) | -1.7 (-2.5, -1.0) |
| verification | -3.3 (-4.4, -2.2) | -0.4 (-1.3, 0.4) |
| reconciliation | -1.4 | -0.8 |
| conflict | -0.9 | -0.7 |
| perspective shift | +0.8 | +0.8 |
Behaviour-shift cosine with A2: RL from zero 0.94 (size 0.41), RL-refined A2 0.99 (1.32), SEAL direction at A2's dose
0.93 (1.03), published SEAL 0.64 (0.16), random -0.78. The RL-from-zero shift also matches the SEAL direction's (0.86).
Residual-space cosines between these vectors are all near zero (RL-zero vs A2 -0.015, vs SEAL direction 0.015).

D-45 outcome (2026-09-27): the pre-registered test was the behaviour level, and it is met. Bias-only RL on a budgeted
correctness reward, started from zero, learns a vector that moves behaviour in A2's direction (cosine 0.94: more subgoal
setting, fewer Q&A turns; verification mostly kept), removes loops, halves tokens at 32k and nearly matches A2's
accuracy, while being orthogonal to A2 in the residual stream and a third of its norm. Started from A2, RL keeps A2 and
strengthens its subgoal component. Reading: the efficient reasoning profile the signature readout found is also what
RL finds when the budget binds, and several unrelated residual directions reach it (A2, SEAL's direction, the RL bias).
What RL finds is A2's behaviour, not A2's vector.

## Re-graded math results (D-47, 2026-09-28). These supersede every math accuracy number above for the runs listed.
Grades: free-text answer extraction (gemma-4-31B, repo prompt) with a grounding check, hand-checked on 100 traces
(extraction right in 90/90 answered and 10/10 unfinished), reference-format normalisation. The earlier direct grader
missed 2,130 of 8,196 correct math answers (26%) and credited 25 answers from unfinished traces. GPQA and planning move by
at most 2 points (their multiple-choice grader was fine). Runs covered: p1_budget, p1_sealval, p2_gpqa, p2_planning,
p3_strat, p3_steer, p4_eval2, p4_eval3, sigpilot3_math. Older runs (power1, sched1, sched1_d1, pilot3, sigx_math) are being
re-graded the same way. The RL training rewards of D-45 used the old grader.

### Math, P1 (199 tasks x 2 seeds)
| arm | 4k | 8k | 12k | 16k | 24k | 32k | answered 32k | correct among answered 32k | tokens 32k |
|---|---|---|---|---|---|---|---|---|---|
| unsteered | 22% | 43% | 54% | 66% | 80% | 85% | 88% | 97% | 13.2k |
| A2 | 45% | 72% | 81% | 87% | 90% | 92% | 99% | 93% | 6.7k |
| SEAL, published | 26% | 45% | 59% | 70% | 82% | 87% | 89% | 98% | 12.3k |
| SEAL direction at A2's dose | 38% | 57% | 71% | 82% | 88% | 89% | 94% | 95% | 8.8k |
| "Be concise." | 29% | 46% | 58% | 66% | 76% | 80% | 83% | 96% | 12.8k |
| random direction | 21% | 39% | 47% | 59% | 73% | 77% | 80% | 97% | 15.2k |
| RL from zero (step 60) | 43% | 66% | 79% | 89% | 94% | 95% | 98% | 97% | 7.1k |
| A2 + RL (step 60) | 60% | 80% | 87% | 90% | 92% | 92% | 100% | 92% | 4.7k |
Paired vs unsteered: A2 +27 (21, 32) at 12k, +7 (3, 11) at 32k; SEAL +4 (0, 9) at 12k, +2 at 32k (ns); SEAL direction
+17 (11, 22) at 12k; concise +4 (ns) at 12k, -5 (-10, 0) at 32k; random -7 at 12k and 32k. A2 minus SEAL +22 (17, 28) at
12k. RL from zero vs A2: -2 (ns) at 12k, +3.5 (1, 6) at 32k. A2 + RL vs A2: +15 (10, 19) at 4k, +6 (3, 9) at 12k, +1 (ns) at 32k.
What changes in the story: A2 still gives the same accuracy for far fewer tokens (A2 at a 16k cap, 87%, matches unsteered at
32k, 85%, with under half the tokens), but it is not "equally accurate when both answer": among answered problems A2 is
right 93% of the time against 97% unsteered. It finishes far more often and slips a little on the ones it finishes, which
fits it cutting verification. RL from zero finishes as often without that slip (it keeps verification) and is the best arm
at 32k (95%).
D-36 replication (HF, 12k, 199 tasks, one sample): unsteered 57%, A2 81% (closed and open loop), rank-one check 63%, random
subspaces 46-47%.

### GPQA and planning (P2), corrected
GPQA at 12k: unsteered 35%, A2 60% (+25), gpqa_A4 73% (+38), random 24%; at 32k: 74%, 73%, 76% (+2, ns), 49%.
Planning at 12k: unsteered 71%, A2 87% (+15), planning_A2 91% (+20), random 62%; at 32k: 89%, 90%, 92% (+2, ns), 79%.

### P3, corrected
Strategy prompts (pass@1, 16k): unprompted 67%, "work backwards" 76% (+9), "try small cases" 64%, "draw a picture" 64%.
Pools at K = 4: unprompted 0.844; half unprompted + half "draw a picture" 0.861 (+1.7, -0.6 to +4.1, not significant);
"draw a picture" alone 0.834; "work backwards" alone 0.879, mixed 0.876. The earlier "first complementary result" does not
survive the correct grading: the picture prompt changes the method (31% of samples off the usual method) but the extra
solves are within noise. Steering along its direction: 66% unsteered, 55% at x2, 43% at x4, 68% random.

Correction (2026-09-28): the "off the usual method" shares above used an arbitrary mode (polars group_by().first() after a sort). Tie-aware values: unprompted 5%, "work backwards" 14%, "try small cases" 18%, "draw a picture" 29%; steering: unsteered 5%, direction x2 9%, x4 13%, random 8%. See D-47 outcome.

## Re-graded older runs (D-47 follow-up, 2026-10-07). These supersede the math accuracies above for power1, pilot3, sched1, sched1_d1, sigx_math.
Free-text extraction (jobs 378510/378511) over 5,624 math traces, graded as in D-47 (`mc/regrade_report.py`,
data/extract_free2/report.csv). The direct grader had missed 743 correct answers and credited 26 from unfinished traces.
Paired change vs each run's unsteered arm (per-task mean over seeds, 95% task bootstrap); old direct-grader value in brackets.
Unsteered levels: power1 65% (12k budget), sched1 59% (12k), sigx_math 57% (12k), pilot3 45% (4k).

| run | arm | accuracy change |
|---|---|---|
| power1 | Idea-ward band +0.5 gap (sustained, layers 12-24) | **-20 (-29, -11)** [-13] |
| power1 | Math-ward band -1 gap | -23 (-33, -13) [-13], 46% loops |
| power1 | reflect gamma -1 / -3 at L19 | -3 (-11, +5) / -1 (-9, +8) [-1 / 0] |
| power1 | random band | +3 (-4, +9) |
| power1 | "explore" prompt / "check" prompt | **+12 (+5, +19)** [+9] / +5 (-1, +12) |
| pilot3 | Idea-ward +1 / +2 gap at L19, band +0.25 / +0.5 gap (4k budget) | -13 / -20 / -15 / -23 |
| sched1 | sustained Idea-ward, +2 gap at L19 | **-15 (-23, -8)** [-13] |
| sched1 | sustained Math-ward, -2 gap at L19 | **+8 (+1, +14)** [+5, ns before] |
| sched1 | explore -> check / check -> explore (switch at half length) | +7 (0, +13) / -7 (-13, +1) |
| sched1 | random -> opposite | +5 (-2, +11) |
| sched1_d1 (half dose) | Math-ward / Idea-ward / random schedule | +6 (0, +12) / -3 (-9, +4) / +9 (+3, +15) |
| sigx_math | Idea A4 (low shift) same vector | **+19 (+13, +25)** [+11] |
| sigx_math | Idea A3 same vector / absolute A3 profile | -16 (-24, -9) / -55 (-61, -48) |
| sigx_math | random open / closed | -12 / -14 |

What changes: (1) the cost of exploring on Math has a size now, about 15-20 points at a 12k budget, larger than the old
grader showed; (2) sustained Math-ward steering along the rank-one direction now helps a little at 12k (+8 on sched1)
instead of "unchanged", while the gamma reflection at L19 (power1) stays at zero; (3) the order effect stands: check ->
explore sits about 12 points below explore -> check; (4) at half dose the random schedule also helps (+9), so small
generic perturbations are not always harmful at 12k, and half-dose schedule contrasts should be read against it;
(5) Idea A4 on Math is +19, close to A2's +24 on the same tasks.

## Atlas re-graded (D-49, 2026-10-07): accuracy per model, free-text extraction, unfinished counts as wrong
| model | math | gpqa | planning | security (WMDP) | math, canonical file |
|---|---|---|---|---|---|
| Llama-3.1-8B (anchor) | 45.8 | 27.2 | 49.2 | 40.4 | 39.6 |
| R1-Distill-Llama-8B | 83.4 | 45.2 | 57.2 | 28.6 | 79.7 |
| Gemma-4-E4B | 92.0 | 53.2 | 83.8 | 61.8 | 55.2 |
| Gemma-4-12B | 97.0 | 61.6 | 89.4 | 56.2 | 45.8 |
| Gemma-4-31B | 98.2 | 78.0 | 93.8 | 84.4 | 47.4 |
| Qwen3.5-4B | 80.6 | 53.8 | 76.0 | 28.4 | 53.2 |
| Qwen3.5-9B | 86.2 | 67.2 | 88.6 | 46.4 | 54.3 |
| Qwen3.5-27B | 87.6 | 77.8 | 92.8 | 75.6 | 57.2 |
500 tasks per cell. On finished traces the Qwen models are at 98-100% on math; their lower numbers here are truncation at
65k tokens. The canonical file excludes unfinished traces, which is why its MCQ numbers are higher for the Qwen models
(e.g. Qwen3.5-4B security 76% on the 36% of traces that finished). Any atlas outcome analysis (prefix monitor,
timing-vs-level, paper RQ outcome models) built on the canonical math grades should be re-run on the new file.

## D-51 Society of Thought crossover: Countdown side (2026-10-07; cd_countdown, 200 four-number tasks x 2 seeds, 32k budget)
| arm (layer 16, norm 6.75) | 4k | 8k | 12k | 16k | 32k | loops | median tokens |
|---|---|---|---|---|---|---|---|
| unsteered | 0% | 5% | 22% | 44% | 62% | 36% | 18.5k |
| A2 (structure) | 23% | 66% | **85%** | 90% | **93%** | 1% | 6.4k |
| conv_down | 21% | 60% | 69% | 70% | 71% | 29% | 6.8k |
| conv_up | 0% | 0% | **1%** | 3% | **7%** | 87% | 32.8k (cap) |
| random | 0% | 2% | 6% | 15% | 24% | 73% | 32.8k (cap) |
Paired vs unsteered at 32k: A2 +31 (+26, +36), conv_down +8.5 (+2.5, +15), conv_up -55 (-61, -49), random -38 (-44, -32).
Accuracy among answered traces is 90-100% in every arm: on Countdown, as on Math, the score is set by whether the model
commits to an answer before the budget. A typical conv_up trace finds a valid equation early and then keeps asking itself
whether a simpler one exists until the budget ends. The pre-registered prediction (conversational-up helps Countdown) fails
on Qwen3.5-9B at this dose; the dose follow-up (D-51b) and the Track B manipulation check are running.

### D-51 Math side and the pre-registered contrast (cd_math + p1_budget arms, 199 tasks x 2 seeds, free-text grades)
| arm | 12k | 32k | change at 12k | change at 32k | loops |
|---|---|---|---|---|---|
| unsteered | 56% | 86% | | | 9% |
| A2 | 83% | 93% | +27 (+22, +32) | +7 (+3, +11) | 0.5% |
| conv_down | 80% | 92% | +24 (+19, +29) | +6 (+2, +9) | 6% |
| conv_up | 10% | 19% | -46 (-52, -40) | -68 (-73, -63) | 75% |
| random | 49% | 79% | -7 (-13, -3) | -8 (-13, -2) | 16% |
Pre-registered contrast at 12k, (conv_up - A2) on Countdown minus (conv_up - A2) on Math: -11 (-18, -5), the opposite
sign to the prediction. No crossover. On both tasks more conversational behaviour costs accuracy and less of it helps
(on Math about as much as A2). On Math the cost is specific to the direction: a random vector of the same norm costs 7
points with 16% loops, conv_up 46 points with 75% loops. Reading so far: on Qwen3.5-9B the Society of Thought behaviours
behave like exploration without a stopping rule, on a search task too. Still to come: the 0.25x / 0.5x doses (D-51b) and
the Track B check that conv_up raises the four conversational rates.

### D-51b dose follow-up on Countdown (post hoc; same 200 tasks x 2 seeds)
| arm | norm | 12k | 32k | loops |
|---|---|---|---|---|
| unsteered | 0 | 22% | 62% | 37% |
| conv_up 0.25x / random 0.25x | 1.7 | 11% / 19% | 49% / 57% | 51% / 44% |
| conv_up 0.5x / random 0.5x | 3.4 | 4% / 15% | 32% / 48% | 68% / 52% |
| conv_up 1x / random 1x | 6.75 | 1% / 6% | 7% / 24% | 92% / 76% |
Paired conv_up minus random at the same norm, 32k: -7.5 (-14, -1.5) at 0.25x, -16 (-21.5, -11) at 0.5x, -17 (-21.5, -12.5)
at 1x. Versus unsteered: -13 (-20, -6), -30 (-36, -24), -55. The harm grows with dose and is larger than a random push of
the same size at every dose, so it is not a dose artefact. On Qwen3.5-9B, raising conversational behaviour does not help
Countdown at any dose tried. (Math dose arms are being graded, job 379474.)

### D-51 manipulation check on Math (Track B, seed 1, 199 tasks, 8k window; rate points x 100 vs unsteered)
conv_up: Q&A +14.0 (+13.0, +15.1), perspective shift +6.6 (+5.9, +7.4), conflict +4.1 (+3.7, +4.5), reconciliation +1.5
(+1.1, +1.9), verification -4.5, backtracking +1.4, subgoal -4.1. Unsteered rates: Q&A 9.7, shift 12.7, conflict 3.8,
reconciliation 4.5. The vector raises all four conversational behaviours, as Society of Thought's positive steering did,
and accuracy still falls by 46 points at 12k. A2's shift on the same tasks reproduces P1 (subgoal +7.4, Q&A -2.9,
verification -3.3). Countdown and the conv_down / dose arms are being labelled.

### D-51b dose follow-up on Math (cd_math, 199 tasks x 2 seeds, free-text grades)
conv_up at 0.25x / 0.5x / 1x of A2's norm: -13 (-18, -8) / -23 (-29, -18) / -46 (-52, -40) at 12k; -8 (-13, -3) / -22 (-28, -16)
/ -68 (-73, -63) at 32k. A full-norm random vector costs -7 at 12k, so even the quarter dose of conv_up is worse than a random
push four times its size. The harm is monotone in dose on both tasks; there is no dose at which more conversational
behaviour helps.

## F1 minimal pairs (D-50), provisional: noise floor (second samples) still generating
80 MATH-500 + 80 GPQA items x {solve_free, solve_mcq, generate, evaluate}; one sample each; Track B over the first 8k
reasoning tokens (truncated traces included, D-50 note). Distances in the atlas log-ratio metric over the 8 behaviour rates.

Within-item distance, mean over 160 items (95% item bootstrap), and topic distance between items at the same prompt type:
| model | format (free vs MCQ) | solve vs generate | solve vs evaluate | generate vs evaluate | between items, other topic / same topic |
|---|---|---|---|---|---|
| Qwen3.5-9B | 0.69 (0.63, 0.76) | 0.92 (0.86, 0.99) | 0.75 (0.68, 0.82) | 1.00 | 1.00 / 0.87 |
| Gemma-4-12B | 1.01 (0.92, 1.09) | 1.65 (1.55, 1.74) | 1.24 (1.12, 1.35) | 1.88 | 1.60 / 1.30 |
| R1-Distill-Llama-8B | 0.87 (0.75, 1.00) | 0.81 (0.71, 0.93) | 0.91 (0.80, 1.02) | 0.97 | 2.09 / 1.38 |

Each model's atlas domain classifier (8 domains, same 8k window, F1 items held out; CV accuracy Qwen 0.69, Gemma 0.66,
R1 0.61) applied to the F1 traces, share predicted as each domain:
- Qwen: math solve_free 56% math; math generate 0% math (code 30%, security 28%, moral 18%, gpqa 12%, idea 1%); math
  evaluate 45% math; math MCQ 40% math, 20% gpqa, 18% planning. GPQA generate 9% gpqa (security 30%, moral 21%).
- Gemma: math solve_free 62% math; math generate 26% math, 40% idea, 14% moral; GPQA generate 0% gpqa, 35% math, 28% idea.
  GPQA MCQ 50% gpqa vs GPQA free 29%.
- R1: math solve_free 65% math, math generate 54% math; GPQA traces are 50-54% gpqa under every prompt type.
Prompt-time states (last prompt token, Qwen and R1; see the projection table): Qwen moves 0.55-0.6 of the Math->Idea gap for
generate prompts and 0.03-0.07 for the MCQ format; R1's state moves with format as much as with demand.

Reading against the D-50 predictions: (1) demand moves the profile more than format in Qwen and Gemma, not in R1; (2) generate
prompts on math content leave the math class in Qwen (0% math) and become idea-like in Gemma (idea + moral 54% vs math 26%),
but stay math in R1; (3) demand is readable at the prompt in Qwen, not in R1. So "deliberation follows task demand" holds
for two of three models; R1-Distill's deliberation follows topic. In Qwen, "list approaches" does not look like idea
generation either; it falls into no atlas domain. The noise floor decides whether the format distances (0.69-1.01) are
above sampling noise.

### F1 with the noise floor (D-50 amendment; second samples, seed 43, of every solve_free and generate prompt)
Distance between two samples of the same prompt: Qwen 0.52 (solve) / 0.49 (generate), Gemma 0.78 / 1.00, R1 0.78 / 0.85. Effects
are estimated from cross-sample pairs so both sides carry the same noise (format = d(solve_free s2, solve_mcq s1) minus the
solve noise; demand = d(solve_free s2, generate s1) minus the mean of the two noises), 160 items, item bootstrap:
| model | format excess | demand excess (generate) | demand minus format |
|---|---|---|---|
| Qwen3.5-9B | +0.15 (+0.08, +0.22) | +0.43 (+0.37, +0.49) | +0.26 (+0.18, +0.34) |
| Gemma-4-12B | +0.23 (+0.13, +0.32) | +0.77 (+0.66, +0.88) | +0.65 (+0.51, +0.79) |
| R1-Distill-Llama-8B | +0.10 (-0.03, +0.23) | +0.01 (-0.08, +0.09) | -0.06 (-0.19, +0.08) |
F1 verdict: in Qwen3.5-9B and Gemma-4-12B the task's demand moves deliberation about three times as much as the answer format,
and both are above sampling noise; the prompt-time state (Qwen) already carries the demand. In R1-Distill neither format nor
demand moves the behaviour profile beyond noise; its traces are organised by topic (between-item distance 2.09 across topics
vs 1.38 within). The format effect the CoT Encyclopedia reports is real but small next to demand in the two newer reasoning
models.

### D-51 manipulation check on Countdown (Track B seed 1, 200 tasks; points x 100 vs unsteered)
Unsteered Countdown rates: Q&A 18.2, perspective shift 22.5, conflict 6.3, reconciliation 2.7, verification 22.5, subgoal 6.5,
backward chaining 8.0. conv_up: Q&A +9.8 (+8.2, +11.4), shift +1.8 (+0.8, +2.8), conflict +0.9 (+0.5, +1.3), reconciliation
-0.7 (-0.9, -0.4), verification -5.2. Dose-ordered: Q&A +2.4 at 0.25x, +5.4 at 0.5x. conv_down: Q&A -12.8, shift -6.7. Random
vectors move Q&A by -3.5 at 1x and ~0 at 0.25x. The vector raises three of the four conversational behaviours on Countdown
(all four on Math), mostly self-questioning, and accuracy falls at every dose.
