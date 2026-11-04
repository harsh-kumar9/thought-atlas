# Thought Atlas manuscript

Editable source for the [Overleaf project](https://www.overleaf.com/project/6ae66af2dc9a0493439722d8). The current title is **Thought Atlas: Task-Dependent Reasoning Profiles and the Effects of Steering**. The anonymous ACL review draft leads with reproducible task signatures, intervention fidelity, and dose-dependent answer production. It remains an evidence-repair draft, not a submission-ready manuscript.

## Evidence-repair revision

Completed analyses document attempted/completed/annotated populations; fitting, pilot, and evaluation overlaps; readout validation with prompt-disjoint training-only preprocessing; intervention fidelity and dose response; shorter-window and shared-baseline sensitivities; all matched-demand conditions; and the 112 candidate/94 eligible outcome tests. A new timing control preserves each trace's length, counts, and joint labels while shuffling their positions. All 14 observed timing contrasts exceed that control.

The readout's mean prompt-balanced MSE skill is 7.61% over training-domain means. Unseen-domain generalization is weaker. Domain vectors amplify contrasts with a reference mean; they do not generally reproduce a source profile. Accuracy and answer availability now appear together, and matched SEAL/random outcome controls are included where historical outputs exist.

The historical RL source contains temperature-variable shadowing; exact executed per-run source versions and rollout rewards cannot be recovered. Saved vectors remain fixed interventions in a secondary appendix analysis. Cosine similarity is not treated as equivalence of achieved profiles or proof of representational insufficiency. Corrected-reward replications use explicit temperature, three training seeds per initialization, and retained rollout/reward/optimizer records. Training is underway; final replication results remain pending.

The old checking/exploration pairwise creativity summaries included math rows from a mixed-task run. The revised summaries use verified idea-task trace IDs only (133 and 54 paired keywords). Math-profile and math-prompt originality point estimates are unchanged. All-attempt coverage and separate experimental budgets are reported.

## Fresh-item confirmation

A2 and the moral-associated vector are fixed at layer 16 and norm 6.75. A new 256-problem sample is frozen from the remaining MATH test set after excluding MATH-500 and screening against 740 saved math prompt texts. Each arm and baseline receives two completions at 8,192 tokens; primary paired accuracy tests use Holm correction. Matched-total-budget elicitation and a fixed 64-problem behavioral subset are secondary checks. All 1,536 attempts and the outcome analysis are complete. A2 gains 32.8 pp [27.9, 37.9] in natural-completion accuracy; the moral contrast gains 8.6 pp [3.3, 13.9], both passing Holm correction. At the same total budget with answer elicitation, the differences are +0.6 pp [-1.6, 2.7] and -3.3 pp [-6.1, -0.6] (secondary, pointwise intervals). The baseline is 95.1% under elicitation. Pairing, token bounds, hashes and forced grading were verified. The 64-problem behavioral audit remains pending. This tests fixed interventions on new project items, not model-pretraining decontamination or a universal selection rule.

## Annotation protocol

Established Kim/Gandhi definitions and their prior human validation are retained and cited. At the user's request, the new audit uses Gemma4-31B-it, Qwen3.5-27B, and Llama-3.1-8B-Instruct on identical blinded prefixes. Its frozen sample includes 1,381 atlas, intervention, and matched-demand traces, including unfinished attempts. This tests cross-family robustness; majority agreement is not treated as truth or as new human validation. Exact templates are in `supplement/`.

Matched-total-budget elicitation is complete (27,920 rows with exact task/seed/budget pairing). At 12,288 tokens, math-to-GPQA changes from +33.8 pp under natural completion to -7.0 pp under elicitation; math-to-planning shrinks from +20.8 to +1.25 pp, with an interval crossing zero. The matched-budget figure and companion tables report the completed control. The three-judge audit, corrected RL replications, and confirmation behavioral audit continue on Ada with pilot checks and resumable jobs. Explicit draft notes identify results not yet available. `revision/STATUS.json` in the repository records job IDs, completed evidence, and remaining work. No pending result is incorporated as an observed finding.

## Source layout

- `main.tex` and `sections/`: editable prose and appendix audits.
- `references.bib`, `acl.sty`, `acl_natbib.bst`: bibliography and official ACL style.
- `tables/`: generated LaTeX summaries.
- `figures/`: vector PDFs and PNG previews; full checkpoint display names use `model_names.json`.
- `supplement/`: aggregate audit CSVs and exact annotation/intervention/evaluation templates.
- `revision/scripts/` and `revision/tests/`: analysis and compute implementations in the repository, excluded from the Overleaf ZIP.
- `archive/before_arr_repair/`: untouched prior draft, figures, scripts, and source hashes.

## Build

From the repository root, using the populated revision outputs:

    MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/confirmation_figure.py
    MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/elicitation_figure.py
    MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/revision_figures.py
    MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/build_figures.py
    .venv/bin/python paper/scripts/build_tables.py
    .venv/bin/python paper/scripts/revision_tables.py
    .venv/bin/python paper/scripts/package_overleaf.py

Compile `main.tex` with pdfLaTeX in Overleaf or locally with Tectonic/latexmk. The ZIP is self-contained and includes aggregate companion tables and prompt templates, but excludes raw traces, activation arrays, private cluster paths, and analysis scripts. Final PDF pagination and visual checks are recorded in `validation.json`.

## Figure design and synchronization

Figures retain the sober pastel, colorblind-conscious design: charcoal text, minimal rules, numeric heatmap cells, and shapes or line patterns alongside hue. Figure 1 separates annotation families and uses one cumulative prediction ladder. The appendix shows real timing and both random controls on an incremental scale. Transfer uses a compact four-panel accuracy/availability matrix. The fidelity/dose figure is now Figure 5 in the main paper and includes the 11% unsteered literal-repetition reference. Historical RL performance is retained in the appendix.

The repository and Overleaf are separate copies. Read and reconcile remote text before replacement; `overleaf_project.json` records synchronization. Do not overwrite collaborator edits from an older snapshot.

## Remaining submission work

Inspect and integrate the queued measurement and control results, then reassess the claims. The frozen fresh-item outcome confirmation is complete; its behavioral-fidelity check remains pending. It tests fixed interventions, not a universal intervention-selection rule. Historical reward error counts remain unrecoverable without rollout logs. Authors, venue cycle, anonymized artifact release, licenses, and applicable assistance disclosures remain to be finalized. Current evidence does not justify claiming validated control, an RL-general mechanism, or independent semantic validity of the labels.

## Companion evidence package

`thought-atlas-evidence.zip` packages aggregate outputs, exact prompts, analysis code, the inspected historical source snapshots, frozen protocols, and a hashed file inventory. Large activation arrays, full trace releases, model weights, and cluster-specific environments are not bundled. It is a review evidence companion, not an anonymized or fully self-contained rerun environment.
