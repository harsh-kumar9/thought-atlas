# Dataset

The repo keeps the current dataset in `data/`. Parquet files are tracked with Git LFS
because several trace files exceed GitHub's normal 100 MB file limit.

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

## Behavior Labels

```text
data/judge/prod/trackA_counts__google_gemma-4-31B-it.parquet
data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet
data/judge/prod/trackB_isolated__google_gemma-4-31B-it.parquet
```

Track A has one row per trace with behavior counts.

Track B has one row per segment with:

- `trace_id`
- `seg_idx`
- `n_segments`
- `norm_pos`
- `section_type`
- `context_mode`
- one binary column per behavior

## Performance and Analysis

```text
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

## Dashboard JSON

`docs/data/` is generated from the parquets:

```text
docs/data/manifest.json
docs/data/summary.json
docs/data/trackA.json
docs/data/heartbeat.json
docs/data/trace_samples.json
docs/data/distance.json
```

Regenerate it with:

```bash
python -m src.analysis.export_dashboard --out-dir docs/data --samples-per-cell 12
```
