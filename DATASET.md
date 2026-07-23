# Dataset

The repo keeps the current dataset in `data/`. Parquet files are tracked with Git LFS
because several trace files exceed GitHub's normal 100 MB file limit.

## Artifact status

The checked-in files listed below are **legacy v1 artifacts**. They are useful for
provenance and the existing dashboard, but fail `scripts/audit_pipeline.py
--strict-v2` and are not valid inputs for final performance claims. A corrected run
is written to `data/v2/` following `RUNBOOK.md`; it should be promoted only after the
strict audit passes.

## Raw Task Inputs

```text
data/tasks/math.parquet
data/tasks/code.parquet
data/tasks/gpqa.parquet
data/tasks/planning.parquet
data/tasks/moral.parquet
data/tasks/idea.parquet
```

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
data/judge/prod/trackA_counts__google_gemma-4-31B-it.parquet
data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet
data/judge/prod/trackB_isolated__google_gemma-4-31B-it.parquet
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
data/perf/success_grades.parquet
data/perf/code_grades.parquet
data/judge/prod/quality__google_gemma-4-31B-it.parquet
data/perf/features.parquet
data/perf/mechanism_coefs.csv
data/perf/mechanism_model_fit.csv
data/analysis/cross_model/model_distance_shape.csv
data/analysis/cross_model/model_distance_mag.csv
```

`success_grades.parquet` covers deterministic domains. `code_grades.parquet` covers
LiveCodeBench execution. `quality__*.parquet` covers moral and idea rubric scoring.

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

## Dashboard JSON

`docs/data/` is generated from the parquets:

```text
docs/data/manifest.json
docs/data/summary.json
docs/data/trackA.json
docs/data/heartbeat.json
docs/data/prefix_monitor.json
docs/data/timing_level.json
docs/data/trace_samples.json
docs/data/distance.json
```

`docs/data/prefix_monitor.json` and `docs/data/timing_level.json` are exported as
downstream analysis examples. They are not shown in the main atlas dashboard by
default; the paper can cite them as examples of how to reuse the dataset.

`data/analysis/prefix_monitor/` contains the corresponding predictability CSVs:
out-of-fold metrics, temporal-vs-baseline deltas, top logistic coefficients, and
analysis metadata.

Regenerate it with:

```bash
python -m src.analysis.export_dashboard --out-dir docs/data --samples-per-cell 12
```
