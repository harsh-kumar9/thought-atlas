# Thought Atlas

Thought Atlas is a cleaned, shareable research repo for `society-task-exp2`, an
extension of the [Society of Thought](https://arxiv.org/abs/2601.10825) line of work.
It studies whether reasoning models use different deliberation behaviors across task
domains, and whether those temporal "heartbeat" patterns are scale-driven or
family/training-lineage driven.

The repo includes the current dataset, the full generation/judging/analysis pipeline,
and a static GitHub Pages dashboard in `docs/`.

## Current Dataset

The August 2026 release is under `data/v2/`; the unversioned `data/` paths retain
the legacy v1 artifacts for provenance only. The current snapshot contains:

- 7 completed generation conditions: Llama 3.1, DeepSeek R1 Distill, Gemma 4
  E4B/31B, and Qwen3.5 4B/9B/27B
- 8 domains: `math`, `code`, `gpqa`, `planning`, `moral`, `idea`, `safety`, and
  `security`
- 24,416 generated traces and 22,111 whole-trace behavior-count rows
- 9,271,040 Track B full/isolated per-sentence label rows
- Deterministic grades for objective tasks, sandboxed code grades, and answer-only
  rubric scores for moral/idea/safety

The release audits preserve rather than impute a small amount of missingness: 13
invalid answer extractions and 673 unparsed Track B rows. A further 1,622 extraction
prompts used the configured 65,536-token clipping policy; accepted rows still pass
the evidence-validation contract. Exact reports are stored under
`data/v2/analysis/audit_*.json` and summarized in [DATASET.md](DATASET.md).

The v2 battery adds:

- `security`: a pinned 500-item sample of
[WMDP-Cyber](https://huggingface.co/datasets/cais/wmdp), graded through the
existing MCQ path. Higher accuracy means more hazardous knowledge, not safer
behavior.
- `safety`: all 313
[StrongREJECT](https://strong-reject.readthedocs.io/) direct harmful requests,
loaded from a pinned, ungated
[Hugging Face mirror](https://huggingface.co/datasets/Machlovi/strongreject-dataset).
The existing answer-only production judge returns refusal, convincingness,
specificity, harmfulness, and a fixed high-harmful-compliance endpoint.

See [SAFETY_SECURITY.md](SAFETY_SECURITY.md) for the interpretation and scoring
contract.

The configuration defines an eight-model battery. This release contains the seven
completed canonical model artifacts listed above; the absent Gemma 4 12B condition
is not represented in the dashboard or release counts.

Large parquet files are intentionally tracked with Git LFS. Before pushing or cloning:

```bash
git lfs install
git lfs track "*.parquet" "*.pdf"
```

The persisted `gen_model` keys still use historical identifiers such as `anchor`
for `Llama-3.1-8B-Instruct` and `reasoner` for
`DeepSeek-R1-Distill-Llama-8B`; the dashboard and docs use the model names.

## Repo Map

```text
configs/                 Experiment config and judge candidates
data/                    Versioned dataset and derived analysis artifacts
docs/                    GitHub Pages dashboard and compact JSON exports
env/                     Cluster environment bootstrap
prereg/                  Pre-registration notes
scripts/                 Cluster dispatch, task prep, analysis, paper figures
src/generate/            vLLM trace generation and thinking-delimiter normalization
src/segment/             ThinkARM-compatible sentence segmentation
src/judge/               Behavior and quality judging with guided JSON
src/perf/                Task grading, behavior features, mechanism analysis
src/analysis/            Aggregate, heartbeat, model-similarity, dashboard export
tests/                   CPU smoke tests
```

See `DATASET.md` for a data dictionary, `RUNBOOK.md` for end-to-end commands, and
`SAFETY_SECURITY.md` for the safety/security benchmark decision.

## Local Setup

For CPU-side inspection, tests, analysis, and dashboard export:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

GPU generation and judging are expected to run in the Blackwell cluster environment
described in `RUNBOOK.md`:

```bash
.venv/bin/pip install -r requirements-gpu.txt
```

On the cluster, prefer the existing `sote` conda env instead of rebuilding locally.

## Pipeline

1. Prepare task parquets, including WMDP-Cyber `security` and StrongREJECT
   `safety`:

```bash
python scripts/02_prepare_tasks.py --config configs/exp.yaml --out-dir data/v2/tasks
```

2. Generate traces, one job per model:

```bash
sbatch -w mira scripts/blackwell.sbatch generate qwen35_9b
```

3. Extract final answers without exposing references:

```bash
sbatch -w vega scripts/blackwell.sbatch extract google/gemma-4-31B-it
```

4. Grade performance:

```bash
python -m src.perf.grade \
  --traces-glob "data/v2/traces/traces_*.parquet" \
  --extractions data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet \
  --out data/v2/perf/success_grades.parquet
srun -w mira --partition=ashton --qos=ashton \
  --cpus-per-task=16 --mem=32G --time=04:00:00 \
  python -m src.perf.grade_code_exec \
    --traces-glob "data/v2/traces/traces_*.parquet" \
    --extractions data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet \
    --out data/v2/perf/code_grades.parquet --max-tests 0
sbatch -w vega scripts/blackwell.sbatch quality google/gemma-4-31B-it
```

The extraction model sees the original task and response but never the reference
answer or rubric. Its evidence must match the raw response verbatim. Objective
correctness remains symbolic/exact/execution-based, and code is selected as an
unchanged source block rather than rewritten by the extractor.

### Re-run scoring without re-running generation

You do not need to generate the model responses again when the canonical files in
`data/v2/traces/` already pass the v3 audit. You may also score the models whose
traces are ready while other generation jobs are still running. For the final
combined dataset, run the commands again after every intended model trace is
present so the broad trace glob covers all models.

First check that the stored traces use the current
`generation-v3-special-tokens` contract:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --strict-v2
```

If the audit passes and the canonical answer-extraction parquet already exists,
validate it and skip directly to the graders:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions-glob \
    'data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet' \
  --strict-v2
```

If that extraction file is missing, stale, or incomplete, re-run only extraction
and wait for the Slurm job to finish. The stage is resumable and skips rows whose
trace hash, extractor, and extraction version still match:

```bash
sbatch -w vega scripts/blackwell.sbatch extract google/gemma-4-31B-it
```

Then re-run objective scoring. Keep the broad trace glob when writing the canonical
`success_grades.parquet`: this grader atomically replaces that file, so using a
single-model glob there would remove the other models from the output.

```bash
python -m src.perf.grade \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions \
    data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet \
  --out data/v2/perf/success_grades.parquet
```

Code scoring is also resumable, but generated code must not be executed on the
login node. It does not require a GPU; use `srun` for an isolated compute
allocation:

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

Safety, moral, and idea outcomes use the same answer-only LLM judge and can
likewise be resumed:

```bash
sbatch -w vega scripts/blackwell.sbatch quality google/gemma-4-31B-it
```

Behavior judging (`judge A`/`judge B`) is independent of performance scoring and
does not need to be re-run merely because answer extraction or performance grades
were refreshed. After all scoring jobs finish, verify exact coverage:

```bash
python scripts/audit_pipeline.py \
  --tasks-dir data/v2/tasks \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --extractions-glob \
    'data/v2/judge/answer_extractions__google_gemma-4-31B-it.parquet' \
  --grades-glob 'data/v2/perf/*_grades.parquet' \
  --quality-glob 'data/v2/judge/quality__google_gemma-4-31B-it.parquet' \
  --strict-v2
```

The checked-in legacy traces cannot be converted into production-valid v3 traces by
re-scoring. Their missing reasoning delimiters were discarded during generation,
so direct re-scoring is useful only for diagnostics; regenerate those traces before
making final accuracy or quality claims.

5. Judge behaviors:

```bash
sbatch -w vega scripts/blackwell.sbatch judge google/gemma-4-31B-it A
sbatch -w mira scripts/blackwell.sbatch judge google/gemma-4-31B-it B
```

6. Analyze and export dashboard data:

```bash
python scripts/run_analysis.py \
  --config configs/exp.yaml \
  --judge-tag google_gemma-4-31B-it \
  --judge-dir data/v2/judge \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --out data/v2/analysis
python -m src.analysis.model_similarity \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --out-dir data/v2/analysis/cross_model --kind shape
python -m src.analysis.model_similarity \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --out-dir data/v2/analysis/cross_model --kind mag
python -m src.analysis.prefix_monitor \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --grades data/v2/perf/success_grades.parquet \
           data/v2/perf/code_grades.parquet \
  --quality data/v2/judge/quality__google_gemma-4-31B-it.parquet \
  --out-dir data/v2/analysis/prefix_monitor --boot 1000
python -m src.analysis.prefix_monitor \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --grades data/v2/perf/success_grades.parquet \
           data/v2/perf/code_grades.parquet \
  --quality data/v2/judge/quality__google_gemma-4-31B-it.parquet \
  --outcome-mode safety_violation \
  --out-dir data/v2/analysis/safety_prefix_monitor \
  --boot 1000
python -m src.analysis.timing_level \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --grades data/v2/perf/success_grades.parquet \
           data/v2/perf/code_grades.parquet \
  --quality data/v2/judge/quality__google_gemma-4-31B-it.parquet \
  --out data/v2/analysis/timing_level.parquet
python -m src.analysis.export_dashboard \
  --traces-glob 'data/v2/traces/traces_*.parquet' \
  --trackA data/v2/judge/trackA_counts__google_gemma-4-31B-it.parquet \
  --trackB data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet \
  --grades data/v2/perf/success_grades.parquet \
           data/v2/perf/code_grades.parquet \
  --quality data/v2/judge/quality__google_gemma-4-31B-it.parquet \
  --distance-dir data/v2/analysis/cross_model \
  --timing-level data/v2/analysis/timing_level.parquet \
  --prefix-monitor-dir data/v2/analysis/prefix_monitor \
  --safety-prefix-monitor-dir data/v2/analysis/safety_prefix_monitor \
  --out-dir docs/data
```

## Adding Another Model

Add a new entry under `gen_models:` in `configs/exp.yaml` with:

- `hf_id`
- `kind`: `reasoning` or `non_reasoning`
- `analysis_source`: usually `think_text` for reasoning models and `answer_text` for non-reasoning controls
- `thinking_style`: one of the styles in `src/generate/thinking_spec.py`
- optional vLLM overrides such as `max_num_seqs` or `gpu_memory_utilization`

Then run:

```bash
sbatch -w mira scripts/blackwell.sbatch generate <model_key>
```

New model families should get a small smoke run first. Do not infer delimiter or vLLM
requirements from config keys alone; check the model card/serving notes and add a
`ThinkingSpec` if the model uses different thinking delimiters.

## Dashboard

The dashboard is a static site in `docs/`, suitable for GitHub Pages. Regenerate its
compact JSON from the checked-in legacy paths with:

```bash
python -m src.analysis.export_dashboard --out-dir docs/data --samples-per-cell 12
```

For the corrected safety/security battery, use the explicit `data/v2/` export
command in Pipeline step 6 above; it also exports
`safety_prefix_monitor.json`.

Run it locally with any static server:

```bash
python -m http.server 8000 -d docs
```

Open `http://127.0.0.1:8000/`.

The dashboard is intentionally an atlas, not an analysis appendix. The Compare tab
explores temporal behavior trajectories across model/domain/outcome lanes. The
Behavior Counts tab uses the same lanes for whole-trace count and presence
comparisons without a temporal cursor. Raw trace and overview tabs expose sampled
task text, thinking text, answer text, model/domain summaries, and model-distance
matrices.

Derived analyses such as prefix predictability and timing-vs-level decompositions
are exported under `data/analysis/` and `docs/data/` so they can be used in the
paper or downstream notebooks without crowding the public explorer.

The dashboard samples raw trace text for browser speed. The full raw prompt,
thinking, and answer fields remain in `data/traces/*.parquet`.
Raw `security` and `safety` trace text is excluded from the public dashboard
export by default; aggregate curves, counts, and safety outcomes remain available.
