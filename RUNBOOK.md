# Runbook

Commands assume the repo is on the CSSLab shared filesystem and jobs are submitted
from `ada` to `vega` or `mira`.

## Environment

The production runs used the `sote` conda env on Blackwell machines:

```bash
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
```

`scripts/blackwell.sbatch` sets the important cache and telemetry variables. Keep
caches off the tiny home quota:

```bash
HF_HOME=/ada1/u/harsh/.cache/huggingface
VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm
TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
XDG_CACHE_HOME=/ada1/u/harsh/.cache
XDG_CONFIG_HOME=/ada1/u/harsh/.config
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## 1. Prepare Tasks

```bash
python scripts/02_prepare_tasks.py --config configs/exp.yaml
```

Outputs:

```text
data/tasks/{math,code,gpqa,planning,moral,idea}.parquet
data/analysis/setup_notes.md
```

## 2. Generate Traces

One job per configured model key:

```bash
sbatch -w mira scripts/blackwell.sbatch generate anchor      # Llama-3.1-8B-Instruct
sbatch -w mira scripts/blackwell.sbatch generate reasoner    # DeepSeek-R1-Distill-Llama-8B
sbatch -w mira scripts/blackwell.sbatch generate qwen35_4b
sbatch -w mira scripts/blackwell.sbatch generate qwen35_9b
sbatch -w mira scripts/blackwell.sbatch generate qwen35_27b
```

Each GPU runs one single-card vLLM replica and writes a shard. The sbatch merges
shards into:

```text
data/traces/traces_<model>.parquet
```

Verify every model before judging:

```bash
python - <<'PY'
import glob, polars as pl
for path in sorted(glob.glob("data/traces/traces_*.parquet")):
    d = pl.read_parquet(path)
    print(path, "n", d.height, "completed", round(d["completed"].mean(), 3))
    print(d.group_by("task_type").agg(pl.col("completed").mean().round(3)).sort("task_type"))
PY
```

## 3. Behavior Judging

Use the same production judge for all generation models.

```bash
sbatch -w vega scripts/blackwell.sbatch judge google/gemma-4-31B-it A
sbatch -w mira scripts/blackwell.sbatch judge google/gemma-4-31B-it B
```

Track A is whole-trace behavior counts. Track B is per-sentence full-context labels
plus the isolated pass when enabled in config.

Sharded runs can be merged with Polars:

```bash
python - <<'PY'
import glob, polars as pl
for stem, keys in [
    ("trackA_counts__google_gemma-4-31B-it", ["trace_id"]),
    ("trackB_full__google_gemma-4-31B-it", ["trace_id", "seg_idx"]),
    ("trackB_isolated__google_gemma-4-31B-it", ["trace_id", "seg_idx"]),
    ("quality__google_gemma-4-31B-it", ["trace_id"]),
]:
    files = sorted(glob.glob(f"data/judge/prod/{stem}.shard*.parquet"))
    if files:
        pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").unique(keys).write_parquet(
            f"data/judge/prod/{stem}.parquet"
        )
PY
```

## 4. Performance Grading

Math, GPQA, and planning:

```bash
python -m src.perf.grade --traces-glob "data/traces/traces_*.parquet" --out data/perf/success_grades.parquet
```

Code execution, only inside a sandboxed compute job:

```bash
srun -w mira --partition=ashton --qos=ashton --cpus-per-task=16 --mem=32G --time=02:00:00 \
  python -m src.perf.grade_code_exec --traces-glob "data/traces/traces_*.parquet" \
  --out data/perf/code_grades.parquet --timeout 8 --cpu-s 10 --max-private 60
```

Moral and idea rubric quality:

```bash
sbatch -w vega scripts/blackwell.sbatch quality google/gemma-4-31B-it
```

## 5. Analysis

Aggregate and heartbeat figures:

```bash
python scripts/run_analysis.py --config configs/exp.yaml --judge-tag google_gemma-4-31B-it --judge-dir data/judge/prod
python scripts/paper_figures.py --config configs/exp.yaml --judge-tag google_gemma-4-31B-it --judge-dir data/judge/prod
```

Mechanism analysis:

```bash
python -m src.perf.features \
  --trackA data/judge/prod/trackA_counts__google_gemma-4-31B-it.parquet \
  --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet \
  --out data/perf/features.parquet

python -m src.perf.mechanism \
  --features data/perf/features.parquet \
  --grades data/perf/success_grades.parquet data/perf/code_grades.parquet data/judge/prod/quality__google_gemma-4-31B-it.parquet \
  --traces-glob "data/traces/traces_*.parquet" \
  --out-dir data/perf
```

Cross-model temporal similarity:

```bash
python -m src.analysis.model_similarity \
  --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob "data/traces/traces_*.parquet" \
  --out-dir data/analysis/cross_model \
  --kind shape

python -m src.analysis.model_similarity \
  --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob "data/traces/traces_*.parquet" \
  --out-dir data/analysis/cross_model \
  --kind mag
```

## 6. Dashboard Export

```bash
python -m src.analysis.export_dashboard \
  --traces-glob "data/traces/traces_*.parquet" \
  --trackA data/judge/prod/trackA_counts__google_gemma-4-31B-it.parquet \
  --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet \
  --grades data/perf/success_grades.parquet data/perf/code_grades.parquet \
  --quality data/judge/prod/quality__google_gemma-4-31B-it.parquet \
  --out-dir docs/data \
  --samples-per-cell 12
```

Serve locally:

```bash
python -m http.server 8000 -d docs
```

## Operational Notes

- For cross-model temporal claims, keep `model_similarity.py`'s completed-only default.
- Truncated traces can masquerade as temporal differences when token budgets differ.
- Qwen3.5 needs `max_num_batched_tokens >= 4096`; the 27B config also caps `max_num_seqs`.
- The code grader executes untrusted code. Do not run it on a login node.
- If a vLLM job crashes, check for orphaned `VLLM::EngineCore` processes before rerunning.
