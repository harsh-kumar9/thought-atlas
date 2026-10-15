# Neuro-style readouts of the mode subspace (D-55 to D-59)

Started 2026-10-13. Harsh asked for mechanistic experiments on the low-rank signature steering work that borrow
from two outside pieces: Sakai and Passingham (2003, Nature Neuroscience 6:75, "Prefrontal interactions reflect
future task operations") and Goodfire's "The world inside neural networks" (May 2026). This file is the design,
the mapping of their measures onto ours, the predictions written before the runs, and the status log. Decisions
are D-55 to D-59 in DECISIONS.md. Code: `mc/neuro_extract.py` (GPU), `mc/neuro_geometry.py` and
`mc/neuro_analyze.py` (CPU). Outputs under `data/neuro/` and `data/runs/nm_{math,idea}/`.

## What we borrow

Sakai and Passingham showed four things with fMRI on a 2 x 2 working-memory task (spatial vs verbal items, forward
vs backward recall), with the items identical across conditions and only the instruction differing:

1. A task set exists before the task: the instruction alone, followed by a 4-12 s empty delay, produced sustained
   activity whose pattern told which task was coming.
2. The item domain (spatial vs verbal) was a magnitude code in domain-specific regions (superior frontal sulcus and
   parietal for spatial, Broca's and Wernicke's areas for verbal), with one anterior prefrontal region active for
   both domains and driven only by the instruction.
3. The operation (forward vs backward) was not a magnitude code anywhere. It was a coupling code: the regression
   slope and correlation between the anterior prefrontal region and the domain-specific regions rose before
   backward tasks, time-locked to the instruction.
4. Methods we want: event-locked time courses with a varied delay, the three-colour contrast map (A > B, B > A, both),
   a conjunction across subjects as the significance rule, and a pre-instruction period as the null for coupling.

Goodfire's argument: representations sit on curved manifolds; a straight steering path between two valid states
can cross a "void" where the model's behaviour breaks or jumps, while a path that follows the manifold moves the
behaviour smoothly. Our own asymmetry looks like that: Math-ward steering of Idea tasks works, Idea-ward steering of
Math tasks loops at the same dose (D-23, D-24, D-38).

## Big picture (2026-10-14, after E1-E6 and before the steering results)

What we are going for. One claim in three parts. Reasoning models carry a mode state: a low-dimensional variable,
set by the task demand before any task content, that organises how the model deliberates (which behaviours it shows,
how long it takes, whether it commits).
1. It is real and readable. The demand sets it at the instruction (E2); it lives in a few readout directions; a small
   set of late attention heads write it (E1), and the MLPs that execute deliberation are domain-general (E1 hub).
2. It is controllable, and what control buys is efficiency and style, not capability. Moving it changes the behaviour
   profile and the tokens to finish; accuracy moves only through finishing within budget (P1, D-36, D-41, F3).
3. It fails in a specific way. Pushing a task toward a mode the model never uses for that task (Idea-mode on Math)
   produces loops; the centroid-line geometry does not explain this (E5), so the explanation to test is that the
   steered states leave the region where the model has natural states (D-68).
F1 to F3 in the paper are the behaviour-level versions of these three; the neuro-style experiments are the
mechanism-level versions.

What the brain comparison is for. Not analogy for its own sake. Sakai and Passingham give a standard measurement set
for a task set (delay-period activity, magnitude vs coupling codes, event-locked time courses, conjunction across
subjects, contrast maps). Running the same measurements on the model gives a table of matches and mismatches, and the
table is the result:

| measurement | brain (Sakai and Passingham 2003) | model (Qwen3.5-9B, this work) |
|---|---|---|
| instruction sets a state before the content | yes, sustained in the delay | yes: AUROC 0.93 at the instruction, held in the readout subspace through a 256-token delay, decays on the single axis |
| domain is a magnitude code in specific regions | yes (SFS/SPL vs IFG/STG) | yes: late attention heads; the top 10% of components carry 70% of the gap |
| a domain-general region driven by the instruction | anterior PFC | the layer 11-19 MLPs: most active in the readout subspace, equal across domains |
| operation is a coupling code in the delay, not a magnitude code | yes (backward raises APF-SFS and APF-IFG coupling) | no: nothing in the delay; once reasoning starts the divergent instruction changes both magnitude (d about 1) and coupling (z doubles) |
| convergent vs divergent operation | both coded (forward vs backward symmetric) | asymmetric: checking barely registers, exploring restructures the whole system (push-pull between MLP hub and Idea heads) |
| anticipation before an event | readiness-like activity before action | the A2 direction rises during the sentence before a structured sentence in Math; the domain axis reacts at onset |

What this changes in the plan.
1. The Goodfire question is re-posed on real trajectories (D-68). Centroid lines were the wrong object: no void, and
   the line is mostly content (cosine 0.16 with the axis). The claim to test: steering fails when it creates states
   with no natural neighbours. Measure the 10-NN density of the actual steered states (line, data path, low-rank
   signature and random arms) against the natural corpus, per trace, and relate it to loops.
2. The component map gets a causal test (D-69). Ablate the top Idea-carrying heads (gate closed) during generation,
   against Math-carrying heads and layer-matched random heads, and see whether the gap and the behaviour profile move.
   If they do, heads are a second unit of control next to the residual vector and the behaviour profile.
3. The SAE line stops at the gate result. The extra steering pair of D-61 is kept only if the ten-pair geometry shows
   a line void above 1.2.
4. The E2 delay is not where the action is for an LLM: the state is set at the instruction and re-expressed at
   generation. Keep the result, do not elaborate it.

Paper figures from this line: the component atlas (E1 + D-61); the task-set time course (E2); coupling restructuring
under the exploratory instruction with the brain-model table (E3); the event-locked A2 lead (E4); steered-state density
vs loops with the three vector families (E5 + D-68); the head-ablation test (D-69).

## Mapping

| Brain term (Sakai) | Our term |
|---|---|
| Region | Component: one attention head or one MLP block of one layer. Qwen3.5-9B has 32 heads x 128 in each of the 24 linear-attention layers, 16 heads x 256 in the 8 full-attention layers, one MLP per layer. We use layers 0-19 (580 components plus the token embedding). |
| Activation magnitude | The component's write onto a direction. The residual stream is a sum (pre-norm), and each attention output is a sum over heads, so the projection of the residual at layer L onto a direction decomposes exactly into per-component terms. Directions: the rank-one domain axis at layer 16 and at layer 19 (`prompt_r1`, + = Idea-like), the 8-dim column space of the rank-8 readout map at layer 16, and the A2 vector. |
| Spatial / verbal | Math / Idea. For the task-set study: the F1 minimal-pair stems with the solve demand vs the generate demand (same stem, D-50). |
| Forward / backward | Checking / exploratory instruction (strategy_prompts.json, paraphrase 0), the same sentences used in power1. |
| Instruction delay | A neutral filler passage of 0, 64 or 256 tokens between the system instruction and the task text. |
| Task epoch | The first 192 generated tokens (task-set study); the first 1536-2048 reasoning tokens (corpus studies). |
| Subjects; conjunction | Sampling seeds (four per task in the corpus) and items. A component counts only if its contrast has the same sign and |t| > 2 in every seed. |
| Inter-regional interaction (slope, r) | Correlation across token positions between two components' write series; slope of a specific component on the hub component by condition; sliding 64-token windows for the time course. |
| Pre-instruction null | Correlation after pairing the hub series of item i with the specific series of item j (shared position effects only), and a permutation of condition labels. |
| Fig. 2 map | Layer x head grid in three classes. |
| Figs. 3, 4, 6 time courses | Projection per token aligned to instruction onset, task onset and generation onset; coupling per sliding window. |

## Experiments

### E1. Component map of the domain gap (D-55)

Data: corpus1_{math,idea}_{a,b} baseline traces (800 per domain, seeds 1-4), prompt plus the first 2048 reasoning
tokens. Per component and trace: mean write onto dom16, dom19, the 8 readout directions and A2 over the window, and
the RMS norm of the 8-dim write (activity). Contrast per component: mean write on Idea traces minus Math traces
(signed, on dom19) and activity Idea minus Math (unsigned). Conjunction over the four seeds. The sum of the signed
contrasts over components equals the Idea-minus-Math gap on the axis, so the signed map is a decomposition of the gap
we steer.

Figure: (a) activity map, three classes (Math > Idea, Idea > Math, active in both, conjunction-masked), layers on x,
heads on y, MLP as an extra row; (b) gap decomposition, diverging colour; (c) cumulative share of the gap by rank.

Prediction P1: a small set carries the gap (top 10% of components carry more than half); the carriers are in layers
8-16 and are mostly MLP blocks and linear-attention heads; the conjunction holds for the top carriers; some
components write strongly along the axis in both domains with no contrast (domain-general axis writers).

### E2. Task set: the instruction delay (D-56)

Prompts: 80 math and 80 GPQA items (F1 stems) x demand {solve, generate} x operation {none, checking, exploratory}
x delay {0, 64, 256}. System message = demand sentence (+ operation sentence). User message = filler (delay tokens)
+ the F1 prompt of that demand. Generate 192 tokens, then teacher-force the whole sequence and read every position:
residual projections at layers 16 and 19, and every component's dom19 write.

Measures: (i) time course of the dom19 projection and of the readout-predicted behaviour profile aligned to
instruction onset, task onset and generation onset, per condition, bootstrap over items; (ii) demand decodability
per position bin (AUROC of solve vs generate on dom19; cross-validated logistic on the 8-dim readout); (iii) per
component in the pre-task span (instruction + delay): magnitude code for demand and for operation (paired t over
items); (iv) coupling: hub set = the components most active in the pre-task span with no demand contrast
(anterior-prefrontal analogue); specific sets = the top demand-contrast components each way; correlation and slope
of specific on hub across the pre-task positions, per prompt, Fisher-z, compared across operation and demand with
items paired; shuffled-item and permutation nulls; sliding-window time course of r.

Prediction P2: demand is readable in the delay (AUROC > 0.9 at delay 256) and sustained to the task; operation is
a weak magnitude code on the demand-specific components (|t| < 2 in the pre-task span) but raises hub-specific
coupling (exploratory > checking, Fisher-z, paired over items). If operation is a magnitude code as well, the paper
says the model differs from the brain on this point.

### E3. Coupling during reasoning (D-57)

Data: power1 baseline, prompt_checking and prompt_exploratory (150 tasks per domain, seed 1, same tasks across
arms), first 1536 reasoning tokens, every component's dom19 write per position. Same hub and specific sets as E2
(defined from the corpus map, not from these traces). Coupling per trace, paired over tasks, by domain x operation;
nulls as in E2; sliding windows of 64 tokens.

Prediction P3: the exploratory instruction raises coupling between the hub and the Idea-specific components on Math
tasks relative to checking; the item-shuffled null is near zero.

### E4. Event-locked time courses (D-58)

Data: the corpus residual series (E1) and the Track B sentence labels. Onset of a labelled sentence = its first
token (segment offsets as in sigctrl). Event-locked average of the dom19, dom16, A2 and 8-dim projections from
-64 to +64 tokens, baseline-corrected on -64..-33, averaged within trace then across traces; null = the same number
of random onsets per trace.

Prediction P4: the axis moves Idea-ward in the 16-32 tokens before a backtracking or perspective-shift sentence and
Math-ward before a verification sentence; the A2 direction rises before subgoal sentences; the shuffled null is flat.
We report the earliest bin whose CI excludes the null as the lead time.

### E5. Geometry of the Math-Idea path and curve steering (D-59)

Data: the sigctrl segment states at layer 16 (and 19) for math, idea, gpqa, planning (corpus1 a+b). PCA to 50 dims
on a 40k-state sample. Straight line = s (mu_idea - mu_math), s in [0, 1.5]. Curve = the shortest path through the
15-nearest-neighbour graph from the state nearest mu_math to the state nearest mu_idea, smoothed by replacing each
node with the mean of its 50 neighbours, parameterised by arc length. Void measure at each point: distance to its
10th nearest data state divided by the median 10-NN distance of the data. Also the domain mix of the neighbours along
the curve (which domains the path passes through).

Steering: fixed vectors at layer 16 in vLLM (vsteer), on the 199 held-out Math tasks of p1_budget (Idea-ward) and
the 200 tx_idea keywords (Math-ward), 2 seeds, 32k budget. Arms per direction: line at s = 0.5 and 1.0, curve at
the same s, curve at the norm of line s = 1.0, random at both norms (p1_budget / tx_idea baselines and random arms
are shared where the norm matches). Outcomes: loops, answered, tokens, Track B on seed 1 (the frozen Idea-likeness
score and the eight rates), Math accuracy by free-text extraction, Idea originality and feasibility.

Prediction P5: the straight line crosses a void (10-NN ratio at the midpoint > 1.5) while the curve stays in
density; Idea-ward curve steering on Math at matched norm produces fewer loops and a larger Idea-ward behaviour
shift than the line; Math-ward on Idea, line and curve behave alike.

### E6. The axis and the path in the SAE basis (D-60)

Qwen-Scope has a TopK SAE (k = 50, 65,536 features) for every layer of Qwen3.5-9B-Base, hooked on the decoder-layer
output. Gate first (trained on Base, applied to instruct reasoning states): FVU < 0.35 per domain at layer 16. Then the
Idea-minus-Math feature difference and its concentration (how many features carry half of the gap on the axis), the SAE
codes along the line and the curve of E5 (active-set turnover, reconstruction error off the data), and two feature-built
steering arms per direction (top 8 and top 64 features, norm-matched to line s = 0.5) in the E5 runs.

Prediction P6: the SAE reconstructs our instruct states adequately at layer 16 (FVU 0.15-0.35); the gap is spread over
many features (more than 100 carry half of it); the active set turns over along both paths (consecutive Jaccard < 0.5)
while the readout moves smoothly; the top-8 feature vector moves behaviour less than the line at the same norm and the
top-64 vector approaches it. If the gate fails, E6 is reported as the gate result only, and the 27B instruct SAE (layer
16) becomes the fallback once the readout pipeline exists for the 27B.

## Figures planned (paper style, no headline text)

1. Component map: activity classes and gap decomposition (E1).
2. Time courses in the instruction delay, small multiples by delay length, lines by demand, line style by operation
   (E2), with the demand AUROC per bin below.
3. Magnitude vs coupling: per-component t for demand and for operation (scatter), and hub-specific coupling by
   condition with nulls (E2, E3), plus the sliding-window r (Fig. 6 analogue).
4. Event-locked averages, 8 behaviours x 3 directions (E4).
5. Atlas map: 2-D embedding of states by domain with the line and the curve drawn, the void profile along both, and
   the steering outcomes (E5).
6. SAE view: feature concentration of the gap and active-set turnover along the line and the curve (E6).

## Status

- 2026-10-13 23:15 UTC: design and code written. Smoke (job 395135, mira): the per-component sum reproduces the
  residual projection at layers 16 and 19 to 0.4-0.5% (bf16 noise), so the decomposition is exact; the delay prompts
  render with the intended spans (256 filler tokens). First look on four traces: the largest single writers on the
  layer-19 axis are the layer-18 and layer-19 MLPs (Math-ward on both domains) and layer-19 full-attention heads 15,
  11, 2, 0; the Idea traces sit at +3.1 on the axis, Math at -1.3 to -2.0.
- 2026-10-13 23:20 UTC: extraction jobs submitted (corpus 395142, prompts 395143, taskset 395144). SAE checkpoints for
  layers 16 and 19 fetched. Geometry (E5) and the SAE gate (E6) running on ada.
- 2026-10-13 23:25 UTC, E5 geometry (layer 16, 10k states per domain, PCA-50 keeps the sample; data/neuro/geometry_L16.json):
  the straight line from the Math to the Idea centroid does NOT cross a void by the 10-NN rule: the ratio runs 0.96
  (Math end) to 1.12 (Idea end) and 1.25 at s = 1.5, against the 1.5 threshold in P5. The data path is three times
  longer than the line (23.0 vs 7.8) and detours through GPQA and planning states (neighbour mix at s = 0.5: 72% GPQA,
  24% planning, 4% Math, 0% Idea). Its smoothed points sit at ratio 0.63-0.84 in the middle, biased low as noted in
  D-59; the raw path nodes average 0.87. The centroid difference is almost orthogonal to the prompt axis at this layer
  (cosine 0.16): the gap on the axis is 1.26 of the 7.8 total. So P5's void claim is not supported at layer 16 by this
  density measure; the steering comparison (line vs curve at matched norm) still runs as designed, and the SAE
  reconstruction error along both paths is reported as a second, model-based off-manifold measure.
- 2026-10-13 23:25 UTC, E6 gate (layer 16, 12k instruct reasoning states per domain): FVU 0.37 (math), 0.42 (idea),
  0.37 (gpqa), 0.42 (planning); reconstruction cosine 0.93-0.94; 22-31k of 65k features used per domain. The gate
  (FVU < 0.35) fails, so no SAE steering arms are added (D-60 step 4). The feature contrast and the path codes are
  still computed and reported as descriptive, with this caveat; layer 19 gate also run.
- 2026-10-13 23:50 UTC, E1 (data/neuro/analysis/map_summary.json, fig_neuro_map): the 581 component writes reproduce
  the layer-19 gap exactly (+3.33 = +3.33). The gap is written late and by attention: layers 18-19 carry 49% and layers
  14-17 another 22%; linear-attention heads 62% and full-attention heads 41% in aggregate, MLPs -3% net. The largest
  single carriers are full-attention heads (L19 h3 6.1%, L15 h2 4.4%, L19 h11 4.3%, L11 h9 3.6%) plus the L18 MLP (4.9%)
  and L18 linear head 9 (4.6%). The top 10% of components carry 70% of the signed gap, so P1's "small set" holds, but
  the guess "MLPs and linear heads in layers 8-16" was wrong: the carriers are attention heads of both kinds in layers
  11-19. Conjunction at |t| > 2 passes 509 of 581 (n = 200 + 200 per seed), so the rule is |d| > 0.5 in every seed: 457
  pass. Domain-general writers (large same-sign write in both domains): the L18 and L19 MLPs (Math-ward in both) and
  L19 heads 13 and 15 (Idea-ward in both); the most active components in the readout subspace are the layer 11-19 MLPs
  with near-equal activity in both domains. Reading: MLPs write the deliberation readout regardless of domain, attention
  heads (which read the context) carry the domain contrast.
- 2026-10-14 00:00 UTC, E3 (coupling_summary.csv, coupling_paired.csv, coupling_magnitude.csv; sets disjoint: hub = L18
  MLP, L19 h8, L16/15/14/13/12/11 MLPs; Idea-specific = L19 h3, L18 lin9, L15 h2, L19 h11, L11 h9, L14 lin15, L19 h1,
  L18 lin4 or similar; Math-specific = L16/15/13/17 MLPs and L11/L15 heads): at baseline the hub is anti-coupled with the
  Idea-specific heads (z -0.14 Math, -0.14 Idea) and weakly coupled with the Math-specific set (+0.10, +0.09); shuffled
  nulls within 0.01 of zero. The exploratory instruction doubles both: z -0.32 / -0.29 and +0.21 / +0.19 (paired shifts
  -0.16 and +0.11, t 13-41, 150 tasks); the checking instruction moves them by 0.02 or less. The same instruction also
  shifts the magnitude of the Idea-specific heads by about one SD (paired d 1.05 on Math, 0.74 on Idea) and of the hub
  by 0.5-0.9. So the operation is coded by magnitude AND coupling here, not coupling alone as in Sakai and Passingham;
  what matches is the asymmetry: the convergent instruction (checking) barely registers, the divergent one restructures
  both. P3's sign was wrong: exploring makes hub and Idea-specific heads push against each other on the axis (the MLPs
  pull Math-ward while the heads push Idea-ward), a push-pull rather than a joint drive.
- 2026-10-14 00:00 UTC, E4 (events_lead.csv, fig_neuro_events): on the domain axis the movement starts at sentence
  onset, not before: in Math traces the axis rises Idea-ward after a backtracking (+0.29 over tokens 0-32), conflict
  (+0.22) or question sentence begins and falls after a subgoal sentence (-0.14); in Idea traces it dips Math-ward in
  the 24 tokens before a backtracking sentence and rises after a subgoal sentence (+0.31). The A2 (efficiency) direction
  leads: in Math it rises during the 16-32 tokens before verification, subgoal, perspective-shift and question sentences
  and stays up. Caveat written now: the pre-onset window is the previous sentence, so a lead means the state during the
  preceding sentence predicts the coming one, not a silent readiness signal; sentence offsets carry a few tokens of
  tokenisation slop. P4 holds for A2 in Math, not for the domain axis.
- 2026-10-14 00:05 UTC, E2 first run (fixed filler): the delay is deterministic (D-56 amendment), so the pre-task
  statistics are void; the instruction span and the generation epoch are fine. In the generation epoch, generate
  prompts on math stems sit at +3 to +5 on the layer-19 axis (the Idea pole) and solve prompts at 0 to -2, so the
  demand, not the topic, sets the axis (the F1 result seen directly in the state). Varied-filler rerun submitted.
- 2026-10-14 00:05 UTC, E6 descriptive (sae_contrast_L16.json, sae_path_L16.json; gate failed at both layers, FVU
  0.33-0.42): the SAE recovers 0.69 of the 1.26 gap on the layer-16 axis; 818 features carry half of the feature-space
  gap (spread, as P6 said); the active set along the straight line barely turns over (consecutive Jaccard 0.92, 109
  features over 31 points) and the curve turns over more (0.81, 134 features); reconstruction error along the line
  (0.20-0.30 relative) is no worse than along the curve (0.20-0.26). Neither the density measure nor the SAE marks the
  straight line as off-manifold at layer 16. No SAE steering arms.
- 2026-10-14 01:10 UTC, E2 varied-filler run (job 395209; data/neuro/taskset_varied; analysis_taskset_varied): demand
  readable from the instruction tokens (AUROC 0.93), decays on the axis during the filler (0.55-0.64) but holds in the
  8-dim readout (0.72-0.89), returns in generation (0.78-0.91). Operation: tiny magnitude code in the delay and no
  coupling code over the filler tokens (explore minus check 0.001-0.007 z; an apparent -0.14 came from the shared
  instruction tokens and vanished when the span was restricted to the filler). P2 half-holds: task set = magnitude
  code for the demand; the operation's coupling signature belongs to the task epoch (E3). RESULTS_NEURO.md updated.
- 2026-10-14 01:30 UTC, E5 runs: nm_math (job 395159) runs its seven arms in sequence at about 40 min per arm.
  nm_idea as one job was on pace for 18 h (Idea traces are long), so it was cancelled after its first arm was
  checkpointed and relaunched as four single-GPU jobs with disjoint arms (nm_idea_a: line_s0p5, line_pm; _b: line_s1p0,
  curve_s0p5; _c: curve_n0p5, rand_n0p5; _d: rand_n1p0); vsteer resumes partial traces. Outcome script:
  mc/neuro_steer_eval.py (loops, answered, tokens, math accuracy, paired vs the shared baselines; fig_neuro_steer).
- 2026-10-14 02:30 UTC, D-61 ratified. Jobs: neuro_corp5 (component extraction on gpqa, planning, moral corpora) and
  sigacts_moral (moral segment states for the ten-pair geometry). Analysis code for the five-domain atlas follows; the
  extra steering pair waits for the geometry read.
- 2026-10-14 03:00 UTC: big-picture section added above; D-68 (steered-state density) and D-69 (head ablation)
  logged and built; both queue behind the running steering and D-61 jobs.
- 2026-10-14 03:40 UTC, D-61 done: axis positions math -0.07 < gpqa 1.49 < planning 1.91 < idea 3.26 < moral 3.89;
  territories disjoint (Jaccard <= 0.13; planning = layer-19 heads, idea = early MLPs); push-pull coupling identical in
  every domain; A2 lead in the convergent domains only; no centroid line void above 1.19 -> no extra steering pair.
  D-68 (states) and D-69 (abl1) jobs queued behind the steering runs and another session's jobs.
- 2026-10-14 03:30 UTC: D-69 zero-ablation breaks generation for the carrier heads (not for random heads); judge not run;
  D-69b directional ablation built (run abl2) and queued. D-68 interim: neighbour density does not track loops; the
  vector family x task domain does (low-rank Math signature fixes Idea loops; raw lines and the Idea signature break).
- 2026-10-14 04:40 UTC: density relation inverted (loops = denser states); logged under D-68 interim. Waiting on the
  steering runs (post-run stages, final density, E5 table) and abl2 (D-69b).
- 2026-10-14 09:40 UTC, D-69b: directional ablation of 8 or 16 carrier heads leaves the gap at 13.6-13.7 (baseline
  13.3); random 13.4-14.4; projecting the axis out of the whole residual at L19 gives 11.7 (12%). Heads write the axis
  but do not decide the behaviour; the axis is re-written downstream. E5 arms done (RESULTS_NEURO "E5"). Pending:
  Track B and Math grades on the E5 arms, Idea quality, final density.

