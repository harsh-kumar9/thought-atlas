# Results: neuro-style readouts of the mode subspace (D-55 to D-60)

Design, mapping and predictions: NEURO.md. Numbers here are from data/neuro/analysis and data/neuro/*.json on ada,
2026-10-13/14. Figures in data/neuro/figures (paper style). Steering outcomes (E5 arms) are added when nm_math and
nm_idea finish.

## E1. Where the Math-Idea axis is written (D-55)

Setup: 800 Math and 800 Idea corpus traces (seeds 1-4), first 2048 reasoning tokens, every component's write onto the
layer-19 domain axis (581 components: embedding, every head and MLP of layers 0-19). The sum of the component writes
equals the residual projection (gap +3.33 on both sides).

Verdict on P1: a small set carries the gap, as predicted (top 10% of components = 70% of the signed gap, 62% of the
unsigned contrast). Where it sits was mispredicted. The gap is written late and by attention, not by MLPs:

| layers | share of the gap |
|---|---|
| 0-9 | 0.18 |
| 10-13 | 0.11 |
| 14-17 | 0.22 |
| 18-19 | 0.49 |

| kind | n | share |
|---|---|---|
| linear-attention heads | 480 | 0.62 |
| full-attention heads | 80 | 0.41 |
| MLP blocks | 20 | -0.03 |
| embedding | 1 | 0.00 |

Largest single carriers: L19 full-attention head 3 (6.1%), L18 MLP (4.9%), L18 linear head 9 (4.6%), L15 head 2
(4.4%), L19 head 11 (4.3%), L11 head 9 (3.6%), L14 linear head 15 (3.3%), L19 head 1 (3.3%). Conjunction across the
four seeds at |d| > 0.5: 457 of 581 components; all top carriers pass.

Domain-general writers (the anterior-prefrontal analogue): the L18 and L19 MLPs write Math-ward in both domains (L18:
-0.46 Math, -0.29 Idea), L19 heads 13 and 15 write Idea-ward in both; the most active components in the readout
subspace are the layer 11-19 MLPs, with near-equal activity in both domains (53 components in the "both" class).
Reading: MLP blocks write the deliberation readout regardless of domain; attention heads, which read the context,
carry the domain contrast. Figure: fig_neuro_map.

## E3. Coupling during reasoning (D-57)

Sets from the map, disjoint: hub = the eight most active no-contrast components (L18 MLP, L19 head 8, L16-L11 MLPs);
Idea-specific = the eight strongest positive carriers; Math-specific = the eight strongest negative carriers. Coupling
= Fisher z of the correlation across the first 1536 reasoning tokens between the hub signal (mean write of the hub set
on the axis) and each specific component's write, averaged over the set. Data: power1 (150 tasks per domain, seed 1,
same tasks in every arm).

| arm | Math: hub x Idea-spec | hub x Math-spec | Idea: hub x Idea-spec | hub x Math-spec |
|---|---|---|---|---|
| unsteered | -0.16 | +0.10 | -0.13 | +0.08 |
| checking prompt | -0.18 | +0.11 | -0.14 | +0.08 |
| exploratory prompt | -0.32 | +0.21 | -0.29 | +0.19 |
| shuffled-pair null | 0.00 to -0.03 | 0.00 to +0.03 | -0.01 to -0.04 | +0.01 to +0.03 |

Paired shifts vs unsteered (150 tasks): exploratory -0.16 / +0.11 on both domains (t 13 to 41); checking -0.02 / +0.02
on Math (t 4.5) and under 0.01 on Idea. The exploratory instruction also moves the magnitude of the same sets: paired
d of the Idea-specific heads 1.05 (Math) and 0.74 (Idea), of the hub 0.47 and 0.89; checking: 0.2-0.4.

Verdict on P3: the sign was wrong and the brain analogy holds only in part. The operation is coded by coupling AND by
magnitude in this model (Sakai and Passingham: coupling only). What matches is the asymmetry: the convergent
instruction barely registers, the divergent one restructures the whole system. The direction is a push-pull: when
exploring, the hub MLPs pull Math-ward while the Idea-specific heads push Idea-ward, and the two become more
anti-correlated. The change appears within the first 100 tokens and persists (fig_neuro_coupling b).

## E4. Event-locked averages (D-58)

Residual projections aligned to the first token of every labelled sentence (Track B), minus matched random onsets,
baseline -64..-33; 800 traces per domain, 800 to 29,000 events per cell.

Domain axis (layer 19): movement starts at sentence onset, not before. Math traces: Idea-ward after backtracking
(+0.29 over tokens 0-32), conflict (+0.22), question (+0.06) and reconciliation (+0.10) sentences begin; Math-ward after
subgoal (-0.14). Idea traces: a Math-ward dip in the 24 tokens before a backtracking sentence, Idea-ward after subgoal
(+0.31). A2 direction (layer 16): in Math it rises during the 16-32 tokens before verification, subgoal, perspective-shift
and question sentences and stays up (+0.11 to +0.23 after onset); in Idea it dips before conflict and verification.

Verdict on P4: holds for the A2 direction in Math (the state during the preceding sentence predicts a structured
sentence), not for the domain axis, which reacts rather than anticipates. Caveats: the pre-onset window is the previous
sentence, so a lead is a predictive state, not a silent readiness signal; sentence offsets carry a few tokens of
tokenisation slop. Figure: fig_neuro_events.

## E5. Geometry of the Math-Idea path (D-59): geometry, then the steering arms

Geometry (layer 16, 10k segment states per domain, PCA-50). Straight line between the Math and Idea centroids: length
7.8; data path through the 15-NN graph: 23.0, detouring through gpqa and planning states (at s = 0.5 the 50 nearest
states are 72% gpqa, 24% planning, 4% math, 0% idea). Void ratio (10-NN distance over the data median) along the
line: 0.96 at the Math end, 1.07 at the midpoint, 1.12 at the Idea end; along the smoothed path 0.63-0.96 (biased low;
the raw nodes average 0.87). The centroid difference is nearly orthogonal to the prompt axis at this layer (cosine
0.16; 1.26 of the 7.8 lies on the axis). With moral added, no pair of the ten has a line void above 1.19, and the
high values sit at the Idea end of every pair, which is Idea's own lower density (D-61). First half of P5: not
supported; the straight line between centroids is inside the cloud.

Steering arms (fixed vectors at layer 16 in vLLM, 199 held-out Math tasks and 200 Idea keywords, 2 seeds, 32k budget;
paired against the shared unsteered baselines; analysis/steer_outcomes.csv, steer_paired.csv, fig_neuro_steer).
Track B on seed 1 of the new arms and the free-text Math grades are still queued; the accuracy column below is the
direct grader applied to every arm alike, and the Track B column exists only for the borrowed arms.

Math tasks, toward Idea (hard-loop rate; change in accuracy vs baseline, 95% CI; Idea-likeness shift where labelled):

| arm | norm | on the axis (gap units) | hard loops | d accuracy | d s_all |
|---|---|---|---|---|---|
| unsteered | 0 | 0 | 0.09 | 0 | 0 |
| line s = 0.5 | 3.9 | +0.50 | 0.10 | -0.01 (-0.07, +0.06) | pending |
| line, projection-matched | 2.7 | +0.35 | 0.11 | -0.04 (-0.10, +0.02) | pending |
| line s = 1.0 (the Idea centroid) | 7.8 | +1.00 | 0.59 | -0.43 (-0.50, -0.36) | pending |
| data path, norm-matched | 3.9 | +0.35 | 0.26 | -0.18 (-0.25, -0.12) | pending |
| data path s = 0.5 | 7.6 | +0.70 | 0.90 | -0.57 (-0.63, -0.52) | pending |
| random | 3.9 | 0 | 0.14 | -0.04 (-0.10, +0.02) | pending |
| random | 6.75 | 0 | 0.16 | -0.08 (-0.14, -0.02) | +0.6 |
| random | 7.8 | 0 | 0.30 | -0.15 (-0.21, -0.10) | pending |
| Idea signature, low-rank | 6.75 | (readout) | 0.90 | -0.56 (-0.62, -0.50) | +4.0 |
| Idea signature + half A2 | 6.75 | (readout) | 0.46 | -0.23 (-0.29, -0.16) | +1.3 |
| A2, low-rank | 6.75 | (readout) | 0.005 | +0.09 (+0.03, +0.15) | -0.1 |

Idea tasks, toward Math (hard-loop rate; change in loops and in tokens vs baseline; Idea-likeness shift where labelled):

| arm | norm | on the axis (gap units) | hard loops | d loops | d tokens | d s_all |
|---|---|---|---|---|---|---|
| unsteered | 0 | 0 | 0.32 | 0 | 0 | 0 |
| line, projection-matched | 0.8 | -0.11 | 0.33 | +0.01 (-0.06, +0.07) | +0.4k | pending |
| line s = 0.5 | 3.9 | -0.50 | 0.75 | +0.43 (+0.36, +0.50) | +13.1k | pending |
| line s = 1.0 (the Math centroid) | 7.8 | -1.00 | 0.96 | +0.64 (+0.59, +0.69) | +18.1k | pending |
| data path, norm-matched | 3.9 | -0.11 | 0.29 | -0.03 (-0.10, +0.03) | +1.3k (ns) | pending |
| data path s = 0.5 | 7.9 | -0.30 | 0.98 | +0.66 (+0.61, +0.71) | +18.1k | pending |
| random | 3.9 | 0 | 0.66 | +0.34 (+0.28, +0.39) | +8.8k | pending |
| random | 6.75 | 0 | 0.92 | +0.60 (+0.54, +0.66) | +17.3k | +11.1 |
| random | 7.8 | 0 | 0.87 | +0.55 (+0.50, +0.61) | +14.7k | pending |
| Math signature, low-rank | 6.75 | (readout) | 0.00 | -0.32 (-0.37, -0.27) | -12.9k | -5.7 |

Verdict on the second half of P5: the data path does not help in either direction. On Math it is worse than the
straight line at the same norm (loops 0.26 vs 0.10, accuracy -0.18 vs -0.01) and worse than a random vector at the
full norm (0.90 vs 0.30); on Idea the norm-matched path leaves everything at baseline, and so does the line at the
same axis projection, so the path's extra norm is off-axis content that is harmless on Idea and harmful on Math. What
the arms show instead: every raw-space vector (line, path, random) breaks Idea tasks from norm 3.9 up and Math tasks
from norm 7 up, regardless of direction; the one vector that steers coherently is the low-rank Math signature on Idea
tasks (loops 0.32 to 0.00, 13k fewer tokens, Idea-likeness down 5.7 logits of a 16-logit gap, backtracking -0.08 and
verification and subgoal +0.07 to +0.09); the low-rank Idea signature on Math breaks like a raw vector (0.90 loops).
Coherent control is low-rank and convergent-ward. The density test (D-68) says why the raw vectors fail is not
"leaving the manifold": looping traces sit in denser regions than coherent ones.

## E6. The axis in the SAE basis (D-60)

Gate (Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50 on instruct reasoning states): FVU 0.37 / 0.42 / 0.37 / 0.42 at layer 16
and 0.33 / 0.40 / 0.34 / 0.39 at layer 19 (math / idea / gpqa / planning), reconstruction cosine 0.89-0.94. Fails the
pre-set FVU < 0.35, so no SAE steering arms. Descriptive, with that caveat: the SAE recovers 0.69 of the 1.26 gap on
the layer-16 axis; 818 features carry half of the feature-space gap (spread, as P6 said; top 8 features 0.32, top 64
0.46 of the signed share). Along the straight line the active set barely turns over (consecutive Jaccard 0.92, 109
features over 31 points); along the data path it turns over more (0.81, 134 features); reconstruction error along the
line (0.20-0.30 relative) is no worse than along the path (0.20-0.26). The SAE does not mark the line as off-manifold
either.

## E2. Task set in an instruction delay (D-56)

Two runs. Fixed filler (data/neuro/taskset): the delay is deterministic within a condition, so its statistics are
void (D-56 amendment). Varied filler (data/neuro/taskset_varied, one 28-sentence filler per item): 80 math + 80 GPQA
stems x demand {solve, generate} x operation {none, checking, exploratory} x delay {0, 64, 256}, 192 generated tokens.

Demand (AUROC over items, per 8-token bin; math stems; data/neuro/analysis_taskset_varied/taskset_auroc.csv):

| where | axis (layer 19) | 8-dim readout (layer 16) |
|---|---|---|
| instruction tokens (delay 0) | 0.93-0.94 | 0.91-0.93 |
| delay, 64 filler tokens (bins 8+) | 0.54-0.64 | 0.80-0.89 |
| delay, 256 filler tokens (bins 8+) | 0.55-0.60 | 0.72-0.80 |
| generation epoch, after a 256 delay | 0.78-0.89 | 0.85-0.91 |

So the demand is set by the instruction, decays on the single axis while neutral text is read (the filler itself sits
at +2.0 to +2.4 on the axis, solve and generate 0.1-0.2 apart), stays partly readable in the readout subspace, and
returns in full once generation starts. In the generation epoch generate prompts on math stems sit at +3 to +5 (the Idea
pole) and solve prompts at 0 to -2, under every operation and delay: the demand, not the topic, sets the axis.

Operation (checking vs exploratory) in the delay: a tiny magnitude code (mean |shift| per component 0.0013 axis units,
versus 0.0019 for demand) and no coupling code. Coupling over the filler tokens between the hub and the specific sets
(sets from the pre-task data by effect size: Idea-specific = L18/L16/L14 MLPs and L19 heads 2, 11, 3; Math-specific = L19
head 8, L15 heads 0 and 5, L11 heads 3 and 9) is -0.05 (hub x Idea-specific) and -0.03 to -0.05 (hub x Math-specific) in
every condition, shuffled null within 0.01 of zero; explore minus check is 0.001 to 0.007 (taskset_coupling_op.csv).
An earlier pass that included the system-message tokens in the span showed an apparent operation effect of -0.14; it
came from the shared instruction text (identical across items) and vanished when the span was restricted to the filler.

Verdict on P2: the task set exists as a magnitude code for the demand, readable before any task content, with the
readout subspace holding it better than the single axis; the operation is not coded by coupling in the delay. Its
coupling and magnitude effects (E3) appear only once reasoning starts. The brain's delay-period coupling code for the
operation does not transfer to this model. Figures: figures_taskset_varied/fig_neuro_taskset_timecourse, _codes.
Note on statistics: the model is deterministic, so paired t over items is inflated for any instruction effect; effect
sizes and the shuffled null carry the result.

## D-61. All five domains (atlas, coupling, event-locked, ten-pair geometry)

Data: the corpus traces of math, gpqa, planning, idea and moral (800 each, seeds 1-4), same extraction as E1.
Outputs: analysis/atlas_components.csv, atlas_summary.json, fig_neuro_atlas5; coupling_summary.csv; events_lead.csv;
geometry_pairs_L16.json.

Where each domain sits on the Math-Idea axis (layer 19, mean over the first 2048 reasoning tokens): math -0.07,
gpqa 1.49, planning 1.91, idea 3.26, moral 3.89. The axis orders the domains by how divergent their deliberation is,
and moral sits beyond idea.

Territories. For each domain, the 30 components that write most into the readout subspace for that domain relative
to the other four (every-seed conjunction). They barely overlap: Jaccard 0.13 for math-gpqa, 0.11 for gpqa-planning
and idea-moral, 0.05 or less for every other pair. Their make-up differs: math = late linear heads and the layer
17-19 MLPs (median layer 17); gpqa = layer-19 full-attention heads with mid MLPs (median 13); planning = almost only
layer-19 full-attention heads (28 of 30); idea = layer 6-9 MLPs, layer-11 head 0 and layer-15 heads (median 11.5);
moral = layer 13-17 linear heads and layer-19 heads. Overlap with the Math-Idea gap carriers of E1 is 0.13-0.33, so
"what distinguishes a domain" and "what carries the Math-Idea axis" are different sets. Verdict on the D-61
prediction: wrong. gpqa and planning do not share Math's territory; each domain has its own, and the convergent trio
is not one territory in component space even though it is close in centroid space (3.4-4.6 apart vs 6.3-7.8 for the
pairs involving idea or moral). Caveat: one-vs-rest activity also reflects prompt format and length (the MCQ domains
have option lists). The topic-controlled version settles it: on the same 160 stems (varied-filler task set, generation
epoch), the generate-minus-solve territory is layer-19 full-attention heads 12, 11, 2, 3, 9, 10 and 4 plus layer-15
head 0, and its axis carriers (layer-19 heads 11, 10, 2, 9, 3, the layer-19 MLP, layer-18 linear head 9, layer-15
head 2) overlap the Math-Idea gap carriers at 0.57; the summed axis shift from solve to generate is +2.29, two thirds
of the corpus gap. So the demand has one set of carriers, the late full-attention heads, while the domain territories
above are format-laden (Jaccard of the demand territory with them: 0.02-0.22).

Coupling (unsteered, hub and specific sets fixed from the Math-Idea map): hub x Idea-specific is -0.14 to -0.17 and
hub x Math-specific +0.08 to +0.10 in every domain (moral +0.15), shuffled nulls within 0.01 of zero. The push-pull
between the MLP hub and the Idea-carrying heads is a property of the model, not of the domain.

Event-locked A2 lead (tokens, then the mean over 0-32 after onset): math 8 / +0.23 before subgoal, 8 / +0.20 before
perspective shift, 8 / +0.15 before a question; planning 8 / +0.15 and 16 / +0.20; gpqa 16 / +0.16 before perspective
shift, otherwise 0 with rises of +0.08 to +0.11; moral 24 / +0.03 before subgoal and nothing else; idea negative
(dips of -0.12 before verification and conflict). Verdict: the A2 lead belongs to the convergent domains, as predicted,
with gpqa weaker than math and planning; moral shows a lead without a rise; idea shows the opposite sign.

Ten-pair geometry (layer 16, five domains): no straight line between centroids reaches a void ratio of 1.2. The
maximum is 1.19 (idea to moral) and 1.18 (math to idea, idea to gpqa, idea to planning), and in every case the high
values sit at the Idea end, which is Idea's own lower density (its centroid scores 1.18 against 0.94-1.00 for the
others). Paths detour through gpqa and planning for every pair that involves idea or moral (path over straight 1.7 to
8.1). Per the rule in the big-picture section, no extra steering pair is run; the budget goes to D-68 and D-69.

