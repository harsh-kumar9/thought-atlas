# Thought Atlas manuscript

Editable source for the [Overleaf project](https://www.overleaf.com/project/6ae66af2dc9a0493439722d8).

This is now a full prose draft: abstract, introduction, related work, methods, results, discussion, conclusion, limitations, ethical considerations, and appendices. It uses the official ACL review style with anonymous authors, line numbers, and A4 two-column layout. Five main figures establish signatures and trajectories, test matched demand, and connect cross-domain transfer, atlas-guided steering, and RL. Four appendix figures cover domain similarity, qualitative examples, open-task tradeoffs, and a matched random-feature control for timing. There are 28 literature entries, including cognitive-science and neuroscience foundations.

The current revision keeps the first paper focused on existing findings. The introduction and discussion connect shared operations and task-dependent organization to Minsky's *The Society of Mind* and flexible cognitive control; budget effects to resource-rationality; behavioral convergence to degeneracy as a qualified analogy; and open-task effects to creative cognition. These connections interpret the evidence without claiming human neural homology, causal mediation, or optimal control. The discussion now includes explicit hypotheses on demand-based transfer, stage-dependent control, and functional differences hidden by average signatures. Limitations pair measurement and inference gaps with concrete follow-ups, including recurrent dynamics, human comparisons, and corrected-reward RL replication. These studies remain future work, outside the current empirical claims. Previous drafts are preserved in `archive/before_cognitive_connections/` and `archive/before_future_directions/`.

## Editing and model names

- `main.tex`: document structure and ACL review option.
- `sections/`: editable prose. `results.tex` is active; `results_scaffold.tex` is a legacy alias.
- `references.bib`: literature and benchmark references.
- `figures/`: vector PDFs and PNG previews.
- `tables/`: generated tables from frozen source data.
- `model_names.json`: display names and exact checkpoint identifiers; figures read this registry.
- `archive/scaffold_2026-10-31/`: preserved prior scaffold.
- `archive/overleaf_before_arr/`: text read from Overleaf before this revision; matched prior source apart from trailing whitespace.
- `acl_template_provenance.json`: official style source URLs and hashes.

Use full names such as Qwen3.5-9B, Gemma4-31B-it, and DeepSeek-R1-Distill-Llama-8B. Exact checkpoint IDs retain their spelling (e.g. `google/gemma-4-31B-it`). Preserve historical internal data keys.

## Build and package

From the repository root:

    .venv/bin/python paper/scripts/build_tables.py
    MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/build_figures.py
    .venv/bin/python paper/scripts/package_overleaf.py

For selective figure rebuilding, append `--only atlas demand ideas`. Compile `main.tex` in Overleaf with pdfLaTeX or locally with `latexmk -pdf main.tex`. The Codex LaTeX editor also compiles the project.

Figure 1 uses a cumulative domain-prediction ladder exported by `PYTHONPATH=. MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/export_signature_ladder.py`. This reproduces the frozen five-fold fits, verifies their stored log losses and classification decisions, and jointly bootstraps cumulative gains over uniform eight-domain prediction. The existing paired amount/timing tests remain the inferential basis. Provenance is in `data/signature_ladder_manifest.json`.

The ZIP includes only manuscript sources, ACL style files, table fragments, figure PDFs, and this README. Raw data, private cluster paths, analysis scripts, and output PDFs are excluded.

## Synchronization

The repository and Overleaf are separate copies, without automatic two-way sync. Inspect/download remote source and reconcile collaborator changes before replacing files. `overleaf_project.json` records the delivered version and source hashes. Do not overwrite a shared project from an older snapshot.

## Evidence and remaining work

The atlas uses corrected October 31 grades including Gemma4-12B-it. RL natural-completion results use reconciled current extraction grades; training rewards and archetype selection used the earlier implementation. No new LM generation or post-training is part of this manuscript revision.

The paper distinguishes full-trace atlas features from first-8,192-token intervention behavior; domain-average vectors from archetypes; and paired quality from scores assigning zero to unanswered attempts. It retains the primary native-outcome null.

This is a first full draft, not submission-ready. Remaining work includes readout fitting/selection provenance, exact prompt/run-manifest audit, population/coverage tables, independent label validation, and RL replication with corrected rewards and additional training seeds. Some are limitations to report, not prerequisites for every narrower claim. Authors, target cycle, anonymized artifacts, licensing review, and applicable AI-assistance disclosure remain to be settled.

Formatting references: [ARR call](https://aclrollingreview.org/cfp), [ACL formatting](https://acl-org.github.io/ACLPUB/formatting.html), [official style files](https://github.com/acl-org/acl-style-files). The current long-paper limit is eight content pages, with limitations and ethical considerations after the conclusion outside that limit. Check pagination after substantive edits.

## Figure design

Figures use a shared sober pastel style: charcoal type, serif panel headings, pale blue sequential maps, and blue–ochre diverging maps centered at zero. Shape and line-pattern encodings supplement color; heatmaps retain numeric values, and trace lanes are explicitly labeled. Gridlines and frames are kept minimal. See `figure_style.json` for the palette and accessibility checks. Earlier figures are preserved in `archive/figures_before_restyle/`.

Compact layouts reserve two-column width for the signature, trajectory, and steering/RL comparisons. Demand, transfer, and open-task plots are designed for a single column; trace examples use a 2×2 grid. Shared labels and legends reduce repetition. The compact layout and random-feature control yielded 16 pages; cognitive-science connections and references brought it to 17. Explicit future-work hypotheses and expanded limitations bring the current local draft to 18 pages. The conclusion remains on page 8 and standard ACL text formatting is unchanged. Earlier layouts are preserved in `archive/before_compact_layout/` and `archive/before_random_timing/`.

Figure 1A separates the conversational and cognitive-operation annotation families with labeled headers and a divider; its caption identifies their respective sources. Figure 1B combines the former amount and timing panels into one connected ladder: uniform guessing, length only, length plus amount, and length plus amount plus timing. Family colors and circle/square shapes match A; open and filled endpoints distinguish the last two stages. The preceding figure is in `archive/before_signature_ladder/`.

The random-feature control is in appendix Figure 9 on an incremental scale, where its small negative effects remain legible. The control jointly shuffles the eight timing features within each training and evaluation partition, retaining the original amount/length predictors and frozen folds. There are 100 randomizations per model/family (1,400 total), with 10,000 paired prompt/randomization bootstrap samples and Bonferroni intervals for 14 contrasts. This is a dimension-matched negative control, not an exact conditional permutation test. `data/random_timing_control/` contains full summaries, randomization-level scores, saved per-trace arrays, the run record, and provenance. `scripts/random_timing_control.py` reproduces it; `scripts/random_timing_control.sbatch` specifies the CPU-only Ada/Vega run. The original job completed all numerical work but needed a JSON-integer serialization repair, documented in its manifest; no numerical outputs changed.
