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

## E5. Geometry of the Math-Idea path (D-59), before the steering arms

Layer 16, 10k segment states per domain (math, idea, gpqa, planning), PCA-50. Straight line between the Math and Idea
centroids: length 7.8; data path through the 15-NN graph: 23.0, detouring through GPQA and planning states (at s = 0.5
the 50 nearest states are 72% GPQA, 24% planning, 4% Math, 0% Idea). Void ratio (10-NN distance over the data median)
along the line: 0.96 at the Math end, 1.07 at the midpoint, 1.12 at the Idea end, 1.25 at s = 1.5; along the smoothed
path 0.63-0.96 (biased low, the raw nodes average 0.87). The centroid difference is nearly orthogonal to the prompt
axis at this layer (cosine 0.16; 1.26 of the 7.8 lies on the axis).

Verdict on the first half of P5: not supported. By this density measure the straight line between the centroids is
inside the cloud, not a void. The steering comparison (line vs path at matched norm, both directions) is the real test
and runs as designed; results pending. Figure: fig_neuro_atlas_L16.

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
