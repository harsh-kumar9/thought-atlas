# Paper-analysis progress

For the durable methods, assumptions, challenge ledger, and clean-checkout
reproduction order, see [README.md](README.md).

**Updated:** August 18, 2026

## Current status

The project owner approved the four non-repairable release exceptions as appendix limitations. The exact exception strings are version-controlled; any new audit issue still blocks. The final full registry contains 24,416 traces. RQ1–RQ9 are now implemented, including within-model domain and quality signature tests, Tufte-style modal-path figures, exact figure sidecars, the finding registry, Markdown/HTML reports, and the final hash manifest.

The central result is moderate rather than maximal: domain/model organization and context dependence are strong, but the prespecified 4+4 family split is not cleanly separated by profile similarity, and coupling/motifs add little held-out outcome value on average. Within a fixed model, both conversational and cognitive amounts robustly distinguish domains and timing adds a smaller increment. Quality-related differences are much narrower: only two cognitive-amount cells survive FDR correction, timing survives in none, and incomplete attempts lack enough usable sentence labels for a defensible three-class analysis.

## What PR1 now provides

- A single hashed config: `configs/paper_analysis.yaml`.
- A stage-selectable CLI with `--stage`, `--dev`, `--force`, and `--jobs`.
- A lazy Polars data layer that avoids materializing the 9.27-million-row Track-B table in pandas.
- A one-row-per-trace index with analysis channel, model family, configured budget, and generation provenance.
- Exact full/isolated pairing on `trace_id + seg_idx`, with unmatched and duplicate keys reported rather than discarded.
- A domain-specific outcome registry. Moral, idea, and safety remain continuous; safety keeps raw harm and a reversed higher-is-better value; security is explicitly hazardous capability, not safety.
- Deterministic prompt-disjoint, leave-one-domain-out, nested leave-one-model-family-out, and optional model-disjoint assignments.
- All seven required `00_audit` CSVs plus a SHA256 manifest.
- Synthetic unit and smoke tests for IO, outcomes, splits, exact exception adjudication, sensitive-text protection by construction, and the audit artifact contract.

## Real dev-mode audit

Command:

```bash
.venv/bin/python scripts/run_paper_analysis.py \
  --config configs/paper_analysis.yaml \
  --stage audit --dev --force
```

Observed:

- 11,025 deterministic dev traces, capped at 200 per model×domain cell;
- seven observed models and eight domains;
- configured Gemma-4 12B condition absent;
- 2,160,826 full rows and 2,160,826 isolated rows;
- 2,160,826 exact pairs, zero full-only rows, zero isolated-only rows;
- zero duplicate `trace_id + seg_idx` keys;
- 431 parse-failure rows inside the dev sample, retained as null rather than zero;
- Qwen3.5-4B dev completion rate: 0.675556;
- one sample seed;
- anchor channel/budget: `answer_text`, 4,096 tokens;
- reasoning-model channel/budget: `think_text`, 65,536 tokens.

The combined full-release strict audit reports 673 Track-B parse-failure rows and these scoring issues:

- 13 invalid answer-extraction rows;
- 1,622 truncated extraction prompts;
- 1 row marked accepted that fails downstream consumer validation.

The dev manifest remains a record of the pre-decision stop. The reviewed full manifest under `data/v2/analysis/paper/` records the exact exception adjudication and is analysis-ready.

## Verification

```text
60 passed in 18.80s
```

This is 37 pre-existing repository tests plus 23 paper-analysis tests.

## Final artifacts

- `docs/paper_analysis/generated/ANALYSIS_REPORT.md`
- `docs/paper_analysis/generated/ANALYSIS_REPORT.html`
- `docs/paper_analysis/generated/SIGNATURE_ANALYSIS_REPORT.md`
- `docs/paper_analysis/generated/SIGNATURE_ANALYSIS_REPORT.html`
- `docs/paper_analysis/generated/figures/`
- `docs/paper_analysis/generated/figure_data/`
- `data/v2/analysis/paper/finding_registry.csv`
- `data/v2/analysis/paper/tables/`
- `data/v2/analysis/paper/final_analysis_manifest.json`
- `data/v2/analysis/paper/09_signatures/`

Outstanding work is external validation rather than another static-corpus RQ: hydrate and label the context-validation sample, and use exact label-level motif permutations if the motif claim is promoted beyond a secondary/null result.
