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
Math accuracy is the free-text extraction grade (D-47) on every arm alike, all traces, unfinished counted wrong; Idea
originality and feasibility are the quality judge (1-10), paired on the traces that produced an idea in both arms (n
given). Idea-likeness (s_all, the frozen 8-behaviour score; + = Idea-like) is the Track B shift on seed 1, paired over tasks; the
Math-Idea gap on it is about 16 logits. Shifts in arms that loop heavily are labels on degenerate text and are read
with the loop column next to them.

Math tasks, toward Idea (hard-loop rate; accuracy and its change vs baseline, 95% CI; Idea-likeness shift where labelled):

| arm | norm | on the axis (gap units) | hard loops | accuracy | d accuracy | d s_all |
|---|---|---|---|---|---|---|
| unsteered | 0 | 0 | 0.09 | 0.86 | 0 | 0 |
| line s = 0.5 | 3.9 | +0.50 | 0.10 | 0.85 | -0.01 (-0.06, +0.04) | +0.83 (+0.53, +1.12) |
| line, projection-matched | 2.7 | +0.35 | 0.11 | 0.85 | -0.01 (-0.06, +0.03) | +0.54 (+0.27, +0.82) |
| line s = 1.0 (the Idea centroid) | 7.8 | +1.00 | 0.59 | 0.39 | -0.47 (-0.55, -0.40) | +7.26 (+6.16, +8.49) |
| data path, norm-matched | 3.9 | +0.35 | 0.26 | 0.63 | -0.23 (-0.29, -0.17) | +0.87 (+0.61, +1.15) |
| data path s = 0.5 | 7.6 | +0.70 | 0.90 | 0.08 | -0.78 (-0.83, -0.73) | +6.26 (+5.59, +6.93) |
| random | 3.9 | 0 | 0.14 | 0.81 | -0.06 (-0.11, -0.01) | +0.14 (-0.14, +0.42) |
| random | 6.75 | 0 | 0.16 | 0.79 | -0.08 (-0.13, -0.02) | +0.6 |
| random | 7.8 | 0 | 0.30 | 0.63 | -0.23 (-0.28, -0.18) | +0.36 (+0.06, +0.71) |
| Idea signature, low-rank | 6.75 | (readout) | 0.90 | 0.09 | -0.77 (-0.82, -0.73) | +4.0 |
| Idea signature + half A2 | 6.75 | (readout) | 0.46 | 0.50 | -0.36 (-0.43, -0.30) | +1.3 |
| A2, low-rank | 6.75 | (readout) | 0.005 | 0.94 | +0.07 (+0.03, +0.11) | -0.1 |

Idea tasks, toward Math (hard-loop rate; change in loops and in tokens vs baseline; originality and feasibility vs
baseline, paired, with n; Idea-likeness shift where labelled):

| arm | norm | on the axis | hard loops | d loops | d tokens | d originality | d feasibility | d s_all |
|---|---|---|---|---|---|---|---|---|
| unsteered (8.5 / 7.3) | 0 | 0 | 0.32 | 0 | 0 | 0 | 0 | 0 |
| line, projection-matched | 0.8 | -0.11 | 0.33 | +0.01 (-0.06, +0.07) | +0.4k | -0.15 (-0.26, -0.06), n 200 | +0.17 (-0.01, +0.33) | -1.59 (-2.83, -0.40) |
| line s = 0.5 | 3.9 | -0.50 | 0.75 | +0.43 (+0.36, +0.50) | +13.1k | -0.83 (-1.03, -0.62), n 63 | +0.73 (+0.40, +1.06) | -1.98 (-3.35, -0.72) |
| line s = 1.0 (the Math centroid) | 7.8 | -1.00 | 0.96 | +0.64 (+0.59, +0.69) | +18.1k | too few ideas | | -2.49 (-4.67, -0.32) |
| data path, norm-matched | 3.9 | -0.11 | 0.29 | -0.03 (-0.10, +0.03) | +1.3k (ns) | -0.15 (-0.27, -0.04), n 204 | +0.13 (-0.03, +0.28) | -4.80 (-5.87, -3.64) |
| data path s = 0.5 | 7.9 | -0.30 | 0.98 | +0.66 (+0.61, +0.71) | +18.1k | too few ideas | | +0.29 (-1.41, +1.94) |
| random | 3.9 | 0 | 0.66 | +0.34 (+0.28, +0.39) | +8.8k | +0.13 (+0.01, +0.25), n 115 | -0.23 (-0.40, -0.05) | +2.39 (+0.82, +3.99) |
| random | 6.75 | 0 | 0.92 | +0.60 (+0.54, +0.66) | +17.3k | n 10 | | +11.1 |
| random | 7.8 | 0 | 0.87 | +0.55 (+0.50, +0.61) | +14.7k | -0.19 (-0.37, 0.00), n 43 | -0.02 | +6.41 (+4.59, +8.00) |
| Math signature, low-rank | 6.75 | (readout) | 0.00 | -0.32 (-0.37, -0.27) | -12.9k | -0.26 (-0.35, -0.17), n 272 | -0.14 (-0.30, +0.01) | -5.7 |

Verdict on the second half of P5, with the labels in: it depends on the direction. Toward Idea (on Math) nothing
coherent moves the profile far. The straight line at half the gap gives +0.8 logits of a 16-logit gap with no loops and
no accuracy cost; the data path at the same norm gives the same +0.9 but loops 0.26 and loses 23 accuracy points; the
arms that move the score by 4 to 7 logits (line at the full gap, path at s = 0.5, the low-rank Idea signature) loop
59-90%, so those shifts are labels on degenerate text. Toward Math (on ideas) two vectors work and the straight line
does not: the low-rank Math signature (-5.7, 0% loops, 13k fewer tokens) and the data path at matched norm (-4.8 with
loops at the baseline rate, backtracking -0.05 and perspective shift -0.06, originality -0.15, feasibility +0.13),
while the line at the same norm gives -2.0 and loops 0.75, and the projection-matched line -1.6. So Goodfire's claim
holds in one direction: following the data path from Idea toward Math, which runs through the gpqa and planning
states, moves ideas toward the convergent profile without breaking them, where the straight line breaks; from Math
toward Idea the path is worse than the line. Every raw-space vector at norm 6.75 or more breaks both domains, and
random vectors make ideas look more Idea-like by the score only because their loops are labelled as exploration
(+2.4 to +11.1 with 66-92% loops). The rule across all 24 arms: coherent control is convergent-ward, by the low-rank
signature or by the data path; divergent-ward control is small when coherent and destructive when large, whatever
the vector family. On ideas every coherent Math-ward push sits on the originality-feasibility line of F3 (path and
projection-matched line -0.15 originality, +0.13 to +0.17 feasibility; the Math signature -0.26 / -0.14). The density
test (D-68) says why the raw vectors fail is not "leaving the manifold": looping traces sit in denser regions than
coherent ones.

## D-68. Steered-state density, final (all 24 arms; analysis/density_arms.csv, density_summary.json, fig_neuro_density)

Layer-16 residual every 32 tokens of every trace, scored by 10-NN distance against a five-domain reference (10k states
per domain, PCA-50), per-trace mean, related to hard loops. Arm means run 1.03-1.15 with no separation by family;
within every arm the Spearman correlation between a trace's mean ratio and its looping is between -0.17 and +0.16;
pooled by task domain the loop rate falls as the ratio rises (Math 0.37, 0.31, 0.26, 0.13 across the bins 0.9-1.0,
1.0-1.1, 1.1-1.25, 1.25-1.5; Idea 0.80, 0.62, 0.58, 0.18). Neighbour domain mix: Idea-ward raw lines on Math stay among
Math and gpqa neighbours (Idea share 0.5-2%); Math-ward raw lines on Idea reach Math neighbours only at the full norm
(15%) and loop; the low-rank signatures leave the mix almost unchanged (Math signature on Idea: Idea share 0.80 vs
0.89) while changing behaviour the most. Verdict: prediction (i) fails and is reversed (looping states are denser,
not sparser); (ii) holds on Idea (signature and path below the line at matched norm) and is reversed on Math; (iii)
is wrong in both directions (coherent steers do not move the states toward the other domain's neighbours; the raw
line does, and loops). Steering failure here is collapse into a repetitive attractor inside the data cloud. The
Goodfire manifold picture does not describe it.

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

## D-69 and D-69b. Do the gap-carrying heads cause the behaviour?

Zero-ablation (D-69, run abl1; 40 held-out tasks per domain, 4096 budget): silencing the eight Idea-carrier heads
(layer-19 heads 3, 11, 1, 2; layer-18 linear head 9; layer-15 head 2; layer-11 head 9; layer-14 linear head 15) or the
eight Math-carrier heads breaks generation on both domains (hard loops 0.83-0.93, nothing answered, every trace to the
budget), while eight layer-matched random heads cost nothing (loops 0). The carriers are load-bearing for coherence.

Directional ablation (D-69b, run abl2; same tasks and budget): remove only each carrier head's write along the
layer-19 axis, at every position, keeping the rest of the head. Loops 0-2% in every arm; answered and lengths at
baseline. Idea-minus-Math gap on the frozen score (mc.analyze; 40 + 40 tasks, bootstrap CI):

| arm | gap | 95% CI | attenuation |
|---|---|---|---|
| unsteered | 13.25 | 12.2 to 14.3 | 0 |
| 8 Idea carriers, axis write removed | 13.74 | 12.7 to 14.9 | -0.04 |
| 16 Idea carriers | 13.57 | 12.4 to 14.8 | -0.02 |
| 8 Math carriers | 14.62 | 13.4 to 15.8 | -0.10 |
| 16 Math carriers | 15.48 | 13.9 to 17.5 | -0.17 |
| 8 random heads, layer-matched | 14.42 | 12.9 to 16.1 | -0.09 |
| 16 random heads | 13.43 | 12.2 to 14.8 | -0.01 |
| whole residual, axis projected out at layer 19 (ceiling) | 11.68 | 10.7 to 12.8 | 0.12 |

The paired shifts per domain are within noise for every head arm (s_all shifts of -0.4 to +0.2 on Math, +0.2 to
+1.8 on Idea, CIs crossing zero except the Math-carrier arms on Idea, which move Idea traces further Idea-ward); the
ceiling arm moves Idea traces -2.0 and Math traces -0.4.

Verdict: the prediction fails. Removing the axis write of the heads that carry 30-45% of the gap at the write leaves
the behaviour gap untouched, and removing the whole axis at layer 19 at every position attenuates it by only 12%.
The gap the heads write is re-established downstream (the layer 17-19 MLPs are the largest axis writers in both
domains), and the behaviour is not set by the axis reading at layer 19. Control needed an overshoot (sustained
gamma = -3 gave 54%, D-24), not a removal. Heads are not a unit of control here; the component map says where the
axis is written, not where the behaviour is decided. This is the operational form of "readout is not control" in
the component domain, next to the vector-level form in F4.

