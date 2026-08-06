# Dataset

The repo keeps the current dataset in `data/`. Parquet files are tracked with Git LFS
because several trace files exceed GitHub's normal 100 MB file limit.

## Artifact status

The files under the unversioned `data/` paths listed below are **legacy v1
artifacts**. They remain useful for provenance, but fail
`scripts/audit_pipeline.py --strict-v2` and are not valid inputs for final
performance claims. The August 2026 dashboard and research snapshot use the
corrected `data/v2/` run.

The v2 release preserves its audit reports under `data/v2/analysis/`. Task, trace,
manifest, coverage, Track A, grading, and quality contracts pass. The accepted
release exceptions are 13 invalid answer extractions out of 24,416 traces, 1,622
extractor prompts clipped to the configured 65,536-token OOM-safe window, and 673
unparsed Track B rows out of 9,271,040 labels. Accepted extraction rows still pass
the evidence-validation contract; dashboard aggregation excludes unparsed judge
rows and retains null outcomes rather than imputing them. See
`audit_scoring.json`, `audit_track_a.json`, and `audit_track_b.json` for exact
machine-readable results.

## Raw Task Inputs

```text
data/tasks/math.parquet
data/tasks/code.parquet
data/tasks/gpqa.parquet
data/tasks/planning.parquet
data/tasks/moral.parquet
data/tasks/idea.parquet
```

Those checked-in paths are the six-domain legacy task set. A rebuilt v2 battery
also contains:

```text
data/v2/tasks/security.parquet
data/v2/tasks/safety.parquet
```

`security.parquet` is a deterministic sample of
[WMDP-Cyber](https://huggingface.co/datasets/cais/wmdp). Its metadata records the
pinned source revision, source answer index, stable option permutation, and prompt
variant. Higher accuracy is evidence of hazardous cyber capability, not a safety
or refusal score.

`safety.parquet` contains all 313 StrongREJECT harmful requests. Metadata records
the six-category stratum, upstream source, pinned Hugging Face mirror revision,
fixed harm threshold, and `prompt_variant=direct_request`. This condition measures
direct-request refusal/harmful compliance; no jailbreak transformation is applied.

Common columns:

- `instance_id`
- `task_type`
- `prompt`
- `reference_answer`
- `difficulty_raw`
- `metadata`

## Generated Traces

```text
data/traces/traces_anchor.parquet
data/traces/traces_reasoner.parquet
data/traces/traces_qwen35_4b.parquet
data/traces/traces_qwen35_9b.parquet
data/traces/traces_qwen35_27b.parquet
```

Historical keys map to model names as follows: `anchor` is
`Llama-3.1-8B-Instruct`, and `reasoner` is
`DeepSeek-R1-Distill-Llama-8B`.

Those are the checked-in legacy files. The corrected run writes the same schema
under `data/v2/traces/` for every key in `configs/exp.yaml`, including
`gemma4_e4b`, `gemma4_12b`, and `gemma4_31b`.

Important columns:

- `trace_id`
- `gen_model`, `gen_model_id`
- `task_type`, `instance_id`, `seed`
- `prompt`
- `full_text`
- `think_text`
- `answer_text`
- `reasoning_text_for_analysis`
- `n_new_tokens`
- `completed`, `finish_reason`, `failure_mode`

The corrected generation contract additionally requires:

- `generation_kind`, `generation_version`, `generation_fingerprint`
- `task_fingerprint`, `task_set_fingerprint`
- `sampling_seed`, `sampling_params`
- `decode_temperature`, `decode_top_p`, `decode_top_k`
- `thinking_style`, `parse_status`, `answer_source`, `close_tag_count`

Generation v3 records `skip_special_tokens=false` in `sampling_params` so
tokenizer-registered reasoning delimiters survive capture. A stopped reasoning
response with no close tag remains available for answer extraction but is excluded
from `reasoning_text_for_analysis`.

`trace_id` is a deterministic UUID over model, instance, seed, prompt, and output.
Canonical trace parquets have adjacent manifests containing a content hash, row
counts, generation fingerprint, natural key, and source-shard hashes.

## Behavior Labels

```text
data/v2/judge/trackA_counts__google_gemma-4-31B-it.parquet
data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet
data/v2/judge/trackB_isolated__google_gemma-4-31B-it.parquet
```

The whole-trace count table has one row per trace with behavior counts. It is
stored under the legacy `trackA` file names for compatibility with earlier runs.

Track B has one row per segment with:

- `trace_id`
- `seg_idx`
- `n_segments`
- `norm_pos`
- `section_type`
- `context_mode`
- one binary column per behavior

V2 behavior tables also preserve `kim_parsed`, `gandhi_parsed`, raw judge JSON
(once per Track B batch), `judge_model`, `score_version`, and truncation flags.
Invalid judge batches have null labels, not implicit zeros.

## Performance and Analysis

```text
data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet
data/v2/perf/success_grades.parquet
data/v2/perf/code_grades.parquet
data/v2/judge/quality__google_gemma-4-31B-it.parquet
data/v2/perf/features.parquet
data/v2/perf/mechanism_coefs.csv
data/v2/perf/mechanism_model_fit.csv
data/v2/analysis/cross_model/model_distance_shape.csv
data/v2/analysis/cross_model/model_distance_mag.csv
```

`success_grades.parquet` covers deterministic domains, including v2 `security`.
`code_grades.parquet` covers LiveCodeBench execution. `quality__*.parquet` covers
safety, moral, and idea answer-only rubric scoring with the same production judge.

`answer_extractions__*.parquet` is a reference-blind normalization layer covering
every trace. It stores the selected answer/span/block, verbatim evidence, validation
status, model and prompt versions, exact source/output hashes, and raw guided-JSON.
It does not contain reference answers or correctness labels.

V2 objective grades preserve the extracted prediction, reference, answer hash,
parse method/status, and grader version. Unparsed answers have `success=null`.
Incomplete length-truncated generations are explicitly ungradeable with
`success=null`; tentative values from unfinished reasoning are not scored.
V2 code grades distinguish unsupported language and worker/infrastructure failures
from model failures and use every official test by default. V2 quality rows preserve
raw axis scores or moral verdicts/weights; signed moral weights are normalized with
absolute weights so `quality_score` is always within `[0,1]`.

StrongREJECT rows additionally contain:

- `refusal`
- `convincingness`, `specificity`
- `safety_harm_score` (`0` safer/refusal, `1` maximally useful harmful response)
- `safety_score = 1 - safety_harm_score`
- `high_harmful_compliance`, using the configured 0.5 threshold
- `harm_threshold`

For compatibility, `quality_score` equals `safety_score` on safety rows, preserving
the repository-wide higher-is-better convention. Safety analyses should use
`safety_harm_score` or `high_harmful_compliance` explicitly.

## Dashboard JSON

`docs/data/` is generated from the parquets:

```text
docs/data/manifest.json
docs/data/summary.json
docs/data/trackA.json
docs/data/heartbeat.json
docs/data/prefix_monitor.json
docs/data/safety_prefix_monitor.json
docs/data/timing_level.json
docs/data/trace_samples.json
docs/data/distance.json
```

Aggregate `security` and `safety` results are exported normally, but their raw
prompt/reasoning samples are omitted from `trace_samples.json` by default. Full
traces remain in the research parquets; reviewed private exports can opt in with
`--include-sensitive-trace-samples`.

`docs/data/prefix_monitor.json` and `docs/data/timing_level.json` are exported as
downstream analysis examples. They are not shown in the main atlas dashboard by
default; the paper can cite them as examples of how to reuse the dataset.

`data/v2/analysis/prefix_monitor/` contains the corresponding predictability CSVs:
out-of-fold metrics, temporal-vs-baseline deltas, top logistic coefficients, and
analysis metadata.

`data/v2/analysis/safety_prefix_monitor/` uses the same machinery with
`--outcome-mode safety_violation`. It reports the fixed-threshold StrongREJECT
endpoint, AUPRC, AUROC, recall at 5% FPR, calibration metrics, and temporal-feature
deltas.

Regenerate it with:

```bash
python -m src.analysis.export_dashboard --out-dir docs/data --samples-per-cell 12
```

That short command uses the legacy default paths. Use the explicit `data/v2/`
command in `RUNBOOK.md` when exporting the safety/security battery.
