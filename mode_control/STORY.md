# The story (draft, 2026-10-14; revised 2026-10-14 evening after D-52, D-53, D-62, D-65b)

What this is: the paper we can write from the atlas, the mode-control runs and the neuro-style readouts; what in the
literature it rests on; where it is new; what is still missing and how we fill it. Pending decisions are D-62 to D-67
in DECISIONS.md. Numbers come from RESULTS.md, RESULTS_NEURO.md and the atlas re-grade (D-49). Nothing here is
ratified yet.

## One paragraph

Reasoning models deliberate differently depending on what the task asks for: solve it, generate options, or judge an
answer. That difference is set before the first thinking token and lives in a low-dimensional internal state we can
read and push. The main direction of the state runs from exploring (perspective shifts, backtracking, conflict) to
committing (verification, subgoals). Pushing it needs sustained input; a pulse does nothing. What transfers between
domains is the commit component, not domain identity: the Math signature is the best transplant on every convergent
target and makes every target finish, amplifying a domain's own signature helps none of them, and divergent signatures
loop everywhere. On tasks with one right answer the commit end buys finishing, not solving: the answer is usually in the
trace already, and a forced answer at the budget recovers as much accuracy as any steering arm or more, so what steering
adds is that the model stops on its own, without a cutoff and without loops, and the RL bias also reaches answers a little
sooner. On open tasks every push trades originality for feasibility. The unit of control is the behaviour profile, not
the vector: the measured behaviour shift predicts the outcome change (R^2 0.81 over 29 cells; the residual cosine adds
nothing), four unrelated residual directions give one profile, and RL on a steering bias lands on it.

## Thesis in one line

Task demand sets a deliberation mode. The mode is a controllable internal state whose main axis runs from exploring to
committing. Turning it toward commit makes the model finish on tasks with one answer (which a forced stop also buys) and
costs originality on open ones. What a push does is predicted by the behaviour profile it produces, not by the vector.

## Why this and not the descriptive atlas

The July audit (Documents/thought-atlas-paper-directions/REPORT.md) already said it: "reasoning has temporal structure"
is occupied (ThinkARM, CoT Encyclopedia, Society of Thought, the trajectory papers), and the best follow-up is
randomised interventions. We have run them. The atlas is the map in section 2; the interventions carry the paper.

## The finding ladder

F0. The map. 8 models x 8 domains x 500 tasks; 8 behaviours per sentence (Gandhi's 4 cognitive, Kim's 4 conversational).
Domain is readable from a trace's behaviour profile at 0.56-0.74 (8-way, chance 0.125). Qwen3.5 4B, 9B and 27B have
the same profile (pairwise distance 0.09-0.19, against 0.5-1.1 across lineages). Gemma, R1-Distill and the Llama
anchor differ. Done; accuracy re-graded (D-49). Caveat: one judge, no human gold.

F1. Demand, not topic or format. Minimal pairs on 80 MATH and 80 GPQA items. Demand (solve vs list approaches vs
evaluate) moves the profile about three times as much as format (free vs MCQ): Qwen3.5-9B +0.43 vs +0.15, Gemma-4-12B
+0.77 vs +0.23, both above sampling noise. R1-Distill: neither; its traces follow topic. In Qwen the state at the last
prompt token already carries the demand. E2 (instruction delay): the demand is readable from the instruction tokens
alone (AUROC 0.93), decays on the single axis while 256 neutral tokens are read (0.55-0.60) but holds in the 8-dim
readout (0.72-0.80), and returns in full at generation. Done.

F2. The mode is a controllable state, and control needs sustained input. One direction from paired strategy prompts
separates spontaneous Math from Idea traces (AUROC about 1). A 64-token pulse does nothing at any layer, rank or
strength. Sustained steering moves Idea traces half way to the Math profile (-9.2 of a 15-logit gap at gamma -3, layer
19: verification up, backtracking, shifts and conflict down, text coherent) and Math traces a tenth to a seventh of the
way toward Idea before they loop. A matched random direction does nothing. Schedules: explore-then-check equals a
random schedule; check-then-explore costs Math about 10 points; a 3-4k-token exploratory phase is sticky, a 2k one is
not. Done; most rank-one arms are one seed; math re-graded.

F3. What turning it buys depends on the demand, and nothing buys both.
- Convergent (math, GPQA, planning, Countdown). The commit-plus-structure profile (A2: subgoal +7, verification -3,
  Q&A -3 points per 100 sentences) gives +27 at 12k on held-out MATH (54 to 81), +25 on GPQA, +15 on planning, +63 on
  Countdown; about 0 at 32k on GPQA and planning; +7 on Math from traces that never finish unsteered. Same accuracy at
  a third to a half of the tokens; loops 9% to 0.5%; accuracy among answered 93% against 97%. A random vector of the
  same norm costs 7 to 23 points everywhere and raises loops. Not SEAL (as published +4, not significant; cosine 0.09),
  not "be concise" (+4, not significant, loops 16%). Each domain's own archetype beats the Math vector (GPQA +38,
  planning +20 at 12k) and the within-task slopes predicted the order. Exploring costs Math 15-20 points at 12k. The
  Society of Thought direction (conversational up) hurts Math and Countdown at every dose, below a random vector of the
  same norm; conversational down helps both.
- Divergent (ideas). Every push sits on one line: toward exploring +0.30 originality and -0.28 feasibility, toward
  checking the reverse; the Math vector costs 0.25 to 0.79 originality with no feasibility gain; no dose, schedule or
  transplant lands in the upper right. Done, but the rubric sits near its ceiling (8.6 and 7.3 of 10), so the negative
  needs a better instrument (D-63).
- Transplant matrix (D-52, five domains, pre-registered sign and rank predictions): all arms generated, Track B
  finishing, outcomes not graded by design until prereg_observed is written.

F4. The behaviour profile, not the vector, is the unit of control. A2, SEAL's direction at A2's dose, the RL-from-zero
bias and the Idea-A4 vector are nearly orthogonal in the residual stream (|cosine| at most 0.09; RL -0.015) and produce
the same Math behaviour shift (cosine 0.86-0.99 with A2). RL on a bias-only vector from zero reaches A2's accuracy (79
against 81 at 12k; 95 against 92 at 32k, the best arm) with profile cosine 0.94 at a third of A2's norm, and it keeps
verification, so it has no 93-against-97 slip. Started from A2, RL keeps A2 and strengthens the subgoal part. The
D-53 regression (behaviour shift predicts the accuracy change, residual cosine adds nothing) waits on the transplant
Track B.

F5. Where it lives (support, not spine). Attention heads in layers 11-19 write the domain contrast (the top 10% of
components carry 70% of the gap); the MLPs write the readout in both domains. The exploratory instruction doubles the
hub x specific anti-coupling and shifts magnitudes by about one SD; the checking instruction barely registers. The A2
direction rises 16-32 tokens before verification, subgoal and question sentences in Math; the domain axis moves at
sentence onset. The straight line between the Math and Idea centroids is not a void at layer 16 (10-NN ratio
0.96-1.12); the line-against-curve steering test is running. The Qwen-Scope Base SAE does not reconstruct instruct
reasoning states (FVU 0.37-0.42), so the SAE view is descriptive only.

## What is known, and where we are new

Known (LLM side):
- Taxonomies and their value for training. Gandhi et al. 2025 (verification, backtracking, subgoal setting, backward
  chaining; priming them lets RL work). Kim et al. 2026, Society of Thought (the conversational four; RL raises them;
  steering a "surprise" SAE feature doubles Countdown on R1-Distill-Llama-8B).
- Behaviours have linear directions and can be steered one at a time. Venhoff et al. 2025 (backtracking, uncertainty,
  example testing in R1-Distill). Ward et al. 2025 (a base-model direction repurposed). CREST, Zhang et al.
  (arXiv 2512.24574, Dec 2025 / Jan 2026): probes find "cognitive heads", head-specific vectors suppress keyword-labelled
  non-linear steps; +17.5 on AMC23 and -37.6% tokens on R1-1.5B, calibrated on MATH, one budget, no random control, no
  conditional accuracy, no open tasks. This is the closest paper and must be compared head on.
- Efficiency and overthinking. SEAL (cut reflection and transition thoughts), manifold steering (NeurIPS 2025: one
  overthinking direction projected to a low-dim manifold, -71% tokens), budget forcing (s1), L1 and ThinkPrune, DEER
  (75% of traces have an early point where a forced answer is already right), answer convergence, "When more thinking
  hurts" (arXiv 2604.10739: flip events, efficiency frontiers), rational metareasoning (De Sabbata, Sumers, Griffiths
  2024: value of computation).
- Format against domain. CoT Encyclopedia (format matters more than domain); Sahoo et al. 2606.02907 (probes detect
  format); ReasonOps fingerprints.
- Steering theory. Non-identifiability (Venkatesh and Kurapath, ICLR 2026: large equivalence classes of behaviourally
  indistinguishable vectors). Representation Without Control (arXiv 2605.25151: readout, sensitivity and control are
  separate properties). arXiv 2609.01048 (representations form before causal readout).
- Pre-reasoning states. "Separable representations of task complexity and deliberation" (ICML 2026: difficulty is
  encoded before the first thinking token); arXiv 2506.08390 (reasoning length predictable before generation); task
  vectors for in-context learning (Hendel; Todd; arXiv 2509.04466).
- Creativity. CreativityNeuro (arXiv 2607.01433: weight steering raises divergent thinking). "Assessing the creativity
  of LLMs" (arXiv 2605.13450: newer models are worse at divergent thinking). Ismayilzada, Beaty et al. (arXiv
  2604.03480: a reasoning-trained variant loses alignment with creative brain activity). LiveIdeaBench.

Where we are new:
1. Interventions are reported as vectors and scored on one benchmark at one budget. We report them as behaviour
   profiles, score them on budget curves with accuracy among answered, and in each task's own currency.
2. Convergent and divergent tasks are studied by different communities with different tools. We put the same internal
   intervention on both and show the trade-off.
3. "Readout is not control" is known. What input achieves control (sustained, not pulsed), how far it reaches in each
   direction, and how long a mode persists (schedules) is not.
4. Demand, topic and format have not been separated for deliberation style. The lineage difference (RL-trained Qwen and
   Gemma follow demand; distilled R1 follows topic) is new, but rests on one distilled model (D-66).
5. Non-identifiability is a theorem about outputs. We show it operationally: four independent methods, RL included, land
   on one profile.
6. The Society of Thought crossover: more conversation does not help search on this model at any dose. It needs the
   direct replication (their feature, their model) before it is stated as a counter-claim (D-67).

## Cognitive science and neuroscience, as predictions we tested

| construct | prediction | our test | outcome |
|---|---|---|---|
| task set (Sakai and Passingham 2003; Sakai 2008; Dosenbach et al. 2006) | the instruction alone creates a sustained preparatory state that says which task is coming | E2 delay design | holds for the demand code; the operation (check vs explore) is not coded in the delay |
| coupling codes the operation (Sakai and Passingham 2003) | explore raises hub x specific coupling, with no magnitude change | E3 | coupling and magnitude both change, as a push-pull; the sign differs from the brain |
| task-set inertia, asymmetric switch cost (Allport et al. 1994; Wylie and Allport 2000) | an earlier set persists and resists reversal | sched1 order swap | explore-first is sticky after 3-4k tokens; check-first is not |
| tonic vs phasic gain (Aston-Jones and Cohen 2005) | a mode is set by sustained gain, not a pulse | pulse vs sustained steering | pulse null, sustained works |
| stability-flexibility, metacontrol (Goschke 2000; Hommel 2015; Musslick and Cohen 2021) | one dial; each end costs the other kind of task | both directions on both tasks | holds, with asymmetric reach |
| dual pathway to creativity (Nijstad et al. 2010) | persistence is also a route to original ideas | structure directions on ideas | fails here: structure always costs originality, unless the rubric cannot see it (D-63) |
| speed-accuracy trade-off, collapsing bounds (Bogacz et al. 2006; Drugowitsch et al. 2012) | a lower bound raises throughput under a deadline and lowers per-decision accuracy | budget curves, accuracy among answered | holds: +27 at 12k, about 0 at 32k, 93 against 97; RL-zero avoids the slip |
| value of computation, metareasoning (Russell and Wefald 1991; Lieder and Griffiths 2020; De Sabbata et al. 2024) | stop when the expected gain is below the cost; loops are failed stopping | forced-answer readiness probe (D-62) | to run |
| proactive vs reactive control (Braver 2012) | proactive control is anticipatory; reactive control is triggered by conflict | event-locked leads (E4) | the A2 direction leads by 16-32 tokens; the domain axis reacts at onset |
| need for closure, seizing and freezing (Kruglanski and Webster 1996) | closure seeking means early commitment and fewer alternatives | the Math vector on ideas | fits: short, structured, 0.25-0.79 less original |
| degeneracy (Edelman and Gally 2001; Prinz, Bucher and Marder 2004) | many parameter sets, one function | orthogonal vectors, one profile | holds |
| creativity networks (Beaty et al. 2016; Ismayilzada, Beaty et al. 2026) | the analytic mode opposes originality | every convergent-helpful direction on ideas | consistent |

Two predictions failed (the coupling code, the dual pathway) and one crossover failed (Society of Thought). They belong
in the paper. They say where this model is not like the brain or the literature.

## Holes, ranked, with the fix

1. One model carries every causal claim. Fix: Gemma-4-12B core cells (D-64): readout from its atlas traces, no new
   corpus; its own efficient archetype on Math and GPQA with budget curves and a random control; the Math-Idea
   transplant both ways; conversational up and down on Countdown; signs pre-registered. Optional: a second distilled
   model on F1 (D-66), so "distilled reasoners follow topic" is a lineage claim or is dropped.
2. No human validation of the labels, and one judge for every effect. Fix: a human gold set (D-65a; sampler written,
   draft sample in data/gold) and a second-family judge on the key steered and unsteered pairs (D-65b).
3. The transplant outcomes (the main test of F2/F3) are not read yet. Fix: queued. shifts and prereg_observed run on
   their own when the last Track B job ends; grading and h2a follow once we have read them.
4. The efficiency claim says "stopping", but we have not shown that the answer was already there. Fix: the forced-answer
   readiness probe (D-62; script written, not yet run).
5. The idea-frontier negative rests on a rubric at its ceiling. Fix: pairwise comparative judging (D-63).
6. F4's regression (behaviour shift against residual cosine) waits on the transplant Track B. The Countdown counter-claim
   needs their feature on their model (D-67, optional).
7. Rank-one arms are one seed. The transplant sig_math and sig_idea arms (two seeds) replace most of that evidence on
   the main axis.

## Paper skeleton

1. Introduction: the stopping and exploring problem; the thesis.
2. The map (F0): signatures by demand across 8 models; within-family invariance; lineage.
3. Demand sets the state before thinking (F1, E2).
4. The state can be turned, with sustained input, both ways, with asymmetric reach (F2, schedules).
5. What turning it buys (F3): efficiency and the speed-accuracy slip on convergent tasks; Countdown; the originality
   against feasibility line on ideas; transplant predictions.
6. The profile is the unit (F4): orthogonal vectors, SEAL, RL.
7. Where it lives (F5): heads against MLPs, coupling, leads, geometry.
8. Discussion: the bridges table; what failed; limits.

Title options: "Reasoning models have a dial between exploring and committing". "Deliberation modes in reasoning
models: set by demand, turned from inside". "What steering reasoning buys".

## Status of runs (2026-10-14 02:30 UTC)

Running: nm_math (E5, Idea-ward line against curve on Math), nm_idea_a-d (E5, Math-ward on ideas), neuro_corp5 (D-61,
five-domain component map), tb_tx_idea_main (last transplant Track B). Pending: sigacts_moral (D-61 geometry). After
tb_tx_idea_main: tx_analyze shifts and prereg_observed run on their own (outputs/tx_after_tb.log), then stop.

## Updates, 2026-10-14 evening

D-52 transplant matrix (peer session; RESULTS "D-52 ..."): the F2 sentence "each domain's signature is tuned to its own
demand" is not supported. No domain is helped by amplifying its own signature (gpqa -31 at 12k, idea never answers, moral
-0.04; planning +7.5 is the only gain). Math's signature is the best transplant on every convergent target (+36 math, +34
gpqa, +21 planning at 12k) and makes all five targets finish (0% loops); idea's and gpqa's signatures loop on every
target (66-100%). On open targets the commit profile costs a little depth (idea originality -0.27, moral rubric -0.05
among answered). H2a: the within-task slopes times the measured Track B shift predict the outcome sign in 15 of 20 cells
(p = 0.021), mean within-target Spearman 0.72; the readout-implied shift does not (12 of 20). H2b: looping follows the
vector's commit component, not the source domain; idea + half A2 halves loops and the math cost. F2 restated: signatures
differ mainly in how they trade commitment against exploration, and observational slopes predict the effect of a measured
shift.

D-53 (peer session): behaviour-shift prediction of the accuracy change, R^2 0.81 over 29 cells; the residual cosine with
A2 (0.22 alone) adds nothing. F4 holds.

D-62 readiness probe (RESULTS "D-62"): the answer is usually there; unsteered Math forced at 4k scores 87% and at 12k
95% against 22% and 54% by its own stopping; A2 commits sooner but never reaches answers sooner; RL from zero reaches
them sooner by 4-7 points at 1-4k. F3 restated: the accuracy-at-budget gains of steering are gains in commitment that a
forced stop also buys; steering's own contribution is self-stopping without a cutoff, no loops, and (for the RL bias) a
small solving gain. The paper compares against budget forcing.

D-65b second judge, interim (RESULTS "D-65b"): kappa 0.60-0.79 for the six frequent behaviours; A2's shift keeps its shape
(cosine 0.91) with subgoal up and self-questioning down under both judges, but the verification cut is gemma-only. The
efficient profile is stated as "more subgoal structure, less self-questioning" until the human gold set decides on
verification.

Open: D-63 (pairwise idea judging) and D-64 (Gemma-4-12B) are running (D-63 job queued; D-64 corpus staged, smoke queued).
D-65a needs a second coder.

D-63 pairwise idea judging, interim (gemma judge; RESULTS "D-63"): no frontier expansion with the ceiling removed. The rank-one
axis trades (Math-ward band -0.54 originality / +0.37 feasibility; exploratory prompt +0.35 / -0.04); the commit transplant is a
pure originality loss (-0.27 to -0.69 with dose, feasibility flat); schedules sit at zero. Surprise to check under the second
judge: the "think like a math problem" instruction raises originality (+0.46) where the Math vector lowers it.

D-65b final: the second judge family agrees with gemma on every headline shift except "A2 cuts verification on Math" (the
commit transplant on ideas, cosine 0.92; conversational-up on Countdown, cosine 0.94; random, 0.96). Segment kappa 0.60-0.79
on the six frequent behaviours.

D-63 final: both judge families agree (net-win correlation 0.96, signs 90%); no frontier expansion anywhere; the "think like a
math problem" instruction's originality gain (+0.46 gemma) holds under Qwen too, so the prompt-vs-vector split on ideas is real.
