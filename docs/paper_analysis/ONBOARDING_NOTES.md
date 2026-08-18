# Paper-analysis onboarding notes

**Recorded:** August 9, 2026  
**Commit:** `90bad30c8a3e0a1e0860969bc84a133e99bd170a`

## Repository and data state

- Current release root: `data/v2/`.
- Canonical traces present: anchor, DeepSeek-R1 distill, Gemma-4 E4B/31B, Qwen3.5 4B/9B/27B.
- Configured but absent canonical condition: Gemma-4 12B.
- Domains present: math, code, GPQA, planning, moral, idea, safety, security.
- Canonical trace count: 24,416 over 3,488 prompt instances.
- Full and isolated Track-B tables are present with 4,635,520 rows each.
- All checked Parquet files have `PAR1` magic bytes; no checked file is an unresolved LFS pointer.
- Git LFS reports no pending LFS object changes.

## Analysis-source and budget caveats

- `anchor` uses `answer_text` and a 4,096-token generation budget.
- The reasoning conditions use `think_text` and a 65,536-token generation budget.
- The anchor must not enter inferential reasoning-trace comparisons without matched channel/budget controls.
- Only one generation seed (`42`) is present per prompt.
- Safety and security are post-hoc extension domains.

## Test and audit status

- Pre-change existing test suite: **37 passed** on August 9, 2026.
- Final combined test suite: **58 passed** (37 existing + 21 paper-analysis tests).
- Trace manifests: valid.
- Duplicate trace natural keys: zero.
- Literal combined strict-v2 audit: **not passing** (`ok: false`); reviewed effective audit: **passing** because all and only the four owner-approved appendix exceptions match.
- Reported audit issues: 13 invalid answer extractions, 1,622 truncated extraction prompts, 1 accepted row failing consumer validation, and 673 Track-B parse-failure rows across both contexts.
- The owner adjudicated the discrepancy on August 9, 2026. Exact exception strings are frozen in `configs/paper_analysis.yaml`; invalid outcomes remain null and invalid Track-B rows remain excluded from denominators.

## Current paper-analysis artifacts

`paper_results/` remains an interim, monolithic analysis package. The prescribed PR1 staged package now exists under `src/analysis/paper/`, with its config at `configs/paper_analysis.yaml`, runner at `scripts/run_paper_analysis.py`, synthetic tests under `tests/paper_analysis/`, and real dev audit under `data/v2/analysis/paper_dev/`.

The dev run deterministically retained at most 200 traces per model×domain cell: 11,025 traces total. Its 2,160,826 full-context segment rows pair exactly with 2,160,826 isolated rows on `trace_id + seg_idx`, with no unmatched rows or duplicate keys. That pre-decision dev manifest remains frozen. The reviewed full audit is analysis-ready because all and only the four owner-approved exceptions match; RQ1–RQ7 outputs are complete under `data/v2/analysis/paper/`.

## Historical working-tree note

At the time this onboarding snapshot was recorded, `docs/app.js`,
`docs/index.html`, and `docs/styles.css` were separate pre-existing dashboard
changes. They are included in the same repository release but are not scientific
analysis inputs. See [README.md](README.md) for the current source and artifact
map.
