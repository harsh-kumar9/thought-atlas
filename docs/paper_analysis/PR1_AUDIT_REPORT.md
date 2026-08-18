# PR1 audit report: is the dataset ready for the paper analyses?

> **Historical gate (August 9, 2026).** This report records the pre-adjudication
> stop decision. The owner later approved the four exact non-repairable issues as
> appendix limitations. The full manifest now records
> `paper_analysis_ready=true`; see [README.md](README.md#data-quality-decisions)
> and [PROGRESS.md](PROGRESS.md) for the current status. The underlying invalid
> rows remain null/excluded and were not repaired or imputed.

## Short answer

The analysis machinery is ready, but the release is not yet cleared for new scientific modeling under the supplied specification.

The data join itself is healthy: in the deterministic dev sample, every full-context segment has exactly one isolated-context partner. The blocker is governance and upstream quality control: the required strict audit returns false.

## Why this audit comes before the research questions

The later RQs compare behavior families, timing, motifs, outcomes, and context sensitivity. A wrong join, a missing label treated as zero, or a reversed safety outcome could create an impressive but false result. PR1 therefore tests the data contracts before testing any theory.

This protects all later RQs:

- RQ1/RQ2 need correct behavior labels and model/domain metadata.
- RQ3/RQ4 need prompt-disjoint splits and correctly directed outcomes.
- RQ5 needs exact full/isolated segment pairing.
- RQ6 needs completion, channel, and token-budget caveats recorded before scale comparisons.

## What the data allows us to conclude now

- The seven available model files and all eight domains are readable Parquet data, not LFS pointers.
- Gemma-4 12B is configured but absent, so no result may imply an observed 12B cell.
- The dev sample contains 11,025 traces, using at most 200 traces per model×domain cell.
- All 2,160,826 dev full-context rows pair exactly with isolated-context rows.
- There are no duplicate segment keys and no context join loss in the dev sample.
- Missing judge labels remain null. The dev sample contains 431 such parse-failure rows; the full Track-B audit reports 673.
- Outcome directions are explicit and tested, including lower-is-better safety harm and higher-hazardous-capability security accuracy.

## Why modeling remains stopped

The combined strict-v2 audit reports four issues: 13 invalid answer extractions, 1,622 truncated extraction prompts, one downstream consumer-contract rejection, and 673 Track-B parse-failure rows. `DATASET.md` describes some of these as accepted release exceptions, while the supplied execution specification says a failed strict audit is a stop condition.

Until that policy conflict is resolved in writing—or the rows are repaired—the generated manifest correctly records:

```text
paper_analysis_ready = false
```

## Where to inspect the evidence

The machine-readable outputs are in `data/v2/analysis/paper_dev/00_audit/`. The most useful starting points are:

- `data_quality_flags.csv` for blockers and required caveats;
- `context_pair_coverage.csv` for exact pairing;
- `outcome_coverage.csv` for outcome availability and direction;
- `analysis_source_by_model.csv` for channel and token-budget confounds;
- `analysis_manifest.json` for hashes and the readiness decision.
