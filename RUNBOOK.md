# Production runbook (v3 generation and extraction contract)

These commands assume the repository is on the CSSLab shared filesystem and jobs
are submitted from `ada` to `vega` or `mira`.  V2 writes under `data/v2/`; do not
mix it with the legacy parquets under `data/tasks`, `data/traces`, and
`data/judge/prod`.

## 1. Update and activate the server environment

```bash
cd /ada1/u/harsh/society-task-exp2
git pull --ff-only origin main
git status --short

set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u

python -m pip install -r requirements.txt -r requirements-gpu.txt
python -m pytest tests/ -q
mkdir -p outputs data/v2/{tasks,traces,perf,judge,analysis}
```

Do not continue unless the tests pass.  The sbatch wrapper defaults to the same
`data/v2/{tasks,traces,judge}` directories and retains all worker shards.

## 2. Rebuild and validate task inputs

```bash
python scripts/02_prepare_tasks.py \
  --config configs/exp.yaml \
  --out-dir data/v2/tasks

python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob '' \
  --strict-v2
```

Check `data/v2/tasks/manifest.json` and `data/v2/tasks/setup_notes.md`.  The task
builder now fails before writing if an MCQ reference is invalid or the ACP prompt
contains both the source and reshuffled option blocks.  GPQA keeps every available
Diamond item before filling from Extended.

## 3. Generate traces

Submit one job per configured model key:

```bash
sbatch -w mira scripts/blackwell.sbatch generate anchor
sbatch -w mira scripts/blackwell.sbatch generate reasoner
sbatch -w mira scripts/blackwell.sbatch generate qwen35_4b
sbatch -w mira scripts/blackwell.sbatch generate qwen35_9b
sbatch -w mira scripts/blackwell.sbatch generate qwen35_27b
```

Monitor with `squeue -u "$USER"` and inspect both the top-level Slurm log and
`outputs/<job>_<model>_shard*.log`.  A failed replica now makes the job fail and
prevents a partial merge.  A successful job creates:

```text
data/v2/traces/traces_<model>.parquet
data/v2/traces/traces_<model>.manifest.json
data/v2/traces/traces_<model>.shardNNofMM.parquet
```

After all five jobs finish:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --json-out data/v2/analysis/post_generation_audit.json \
  --strict-v2
```

The resolver reads canonical files instead of loading canonical files and their
retained shards twice.  It rejects incomplete shard-only sets.

## 4. Extract submitted answers

Run the reference-blind extractor over every trace before any performance grader:

```bash
sbatch -w vega scripts/blackwell.sbatch extract google/gemma-4-31B-it
```

The wrapper launches one replica per GPU and merges only a complete, disjoint shard
set. The extractor never receives `reference_answer`, rubric verdicts, or success
labels. Every usable output is tied to the exact `full_text` hash and either:

- returns a compact math/MCQ answer with verbatim evidence;
- selects an exact code block without rewriting it; or
- identifies a verbatim final-response span for moral/idea.

Validate coverage and evidence contracts:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions-glob \
    'data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet' \
  --strict-v2
```

Do not grade until all extraction rows validate. Invalid rows are not cached as
complete, so resubmitting the same command retries them. If an LLM response violates
the evidence contract but the reference-free deterministic parser can recover an
answer, the artifact records `extraction_method=deterministic_fallback` rather than
silently discarding the row.

## 5. Grade objective tasks

Math, GPQA, and planning are CPU-side and do not execute model code:

```bash
python -m src.perf.grade \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions \
    data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet \
  --out data/v2/perf/success_grades.parquet
```

Run code grading only in a disposable, network-isolated compute environment.  The
grader applies process time/memory/file limits, but those limits are not a security
boundary.  Do not run generated programs on a login node.

```bash
srun -w mira --partition=ashton --qos=ashton \
  --cpus-per-task=16 --mem=32G --time=04:00:00 \
  python -m src.perf.grade_code_exec \
    --traces-glob 'data/v2/traces/traces_*.parquet' \
    --extractions \
      data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet \
    --out data/v2/perf/code_grades.parquet \
    --timeout 8 --cpu-s 10 --max-tests 0
```

`--max-tests 0` means the full public+private suite.  A positive cap is only for
debugging and must not be used for reported pass@1.  Non-Python legacy submissions
are marked unsupported/ungradeable rather than counted as Python failures; fresh v2
prompts explicitly require Python 3.

Validate deterministic-grade coverage:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions-glob \
    'data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet' \
  --grades-glob 'data/v2/perf/*_grades.parquet' \
  --strict-v2
```

## 6. Score moral and idea quality

```bash
sbatch -w vega scripts/blackwell.sbatch quality google/gemma-4-31B-it
```

The wrapper requires the canonical extraction artifact and merges only after every
replica succeeds. The v2 quality table stores
the raw judge JSON, rubric weights/verdicts, judge model, prompt hash, parse status,
and score version.  Signed moral weights use a bounded `[0,1]` formula.

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --quality-glob 'data/v2/judge/quality__google_gemma-4-31B-it.parquet' \
  --strict-v2
```

## 7. Judge deliberation behavior

Run Track A over all traces:

```bash
sbatch -w vega scripts/blackwell.sbatch judge google/gemma-4-31B-it A
```

The config defaults to a Track B pilot of 100 traces per task/model.  Run and
inspect that pilot first:

```bash
sbatch -w mira scripts/blackwell.sbatch judge google/gemma-4-31B-it B
```

If parse coverage, label rates, and truncation rates are acceptable, resume the
same shard files with the limit disabled.  Already-valid pilot traces are skipped:

```bash
sbatch -w mira --export=ALL,TRACK_B_LIMIT=0 \
  scripts/blackwell.sbatch judge google/gemma-4-31B-it B
```

Then validate the canonical behavior outputs:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --judge-glob 'data/v2/judge/track*_counts__google_gemma-4-31B-it.parquet' \
  --strict-v2

python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --judge-glob 'data/v2/judge/trackB_*__google_gemma-4-31B-it.parquet' \
  --strict-v2
```

Track A and B now use the same reasoning-only text.  Missing or malformed judge
batches remain null with explicit parse flags; they are retried on resume rather
than silently becoming all-zero behavior labels.

## 8. Run analysis only after the final audit

Use the v2 paths explicitly:

```bash
python -m src.perf.features \
  --trackA data/v2/judge/trackA_counts__google_gemma-4-31B-it.parquet \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --out data/v2/perf/features.parquet

python -m src.perf.mechanism \
  --features data/v2/perf/features.parquet \
  --grades data/v2/perf/success_grades.parquet \
           data/v2/perf/code_grades.parquet \
           data/v2/judge/quality__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --out-dir data/v2/perf
```

Do not replace or publish the legacy dataset until v2 passes the strict audit and
the expected row counts are reviewed.  Keep the manifests with every promoted
artifact; they record row counts, content hashes, fingerprints, and source shards.
