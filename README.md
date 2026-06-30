# Thought Atlas

Thought Atlas is a cleaned, shareable research repo for `society-task-exp2`, an
extension of the [Society of Thought](https://arxiv.org/abs/2601.10825) line of work.
It studies whether reasoning models use different deliberation behaviors across task
domains, and whether those temporal "heartbeat" patterns are scale-driven or
family/training-lineage driven.

The repo includes the current dataset, the full generation/judging/analysis pipeline,
and a static GitHub Pages dashboard in `docs/`.

## Current Dataset

- 5 generation conditions: `anchor`, `reasoner`, `qwen35_4b`, `qwen35_9b`, `qwen35_27b`
- 6 domains: `math`, `code`, `gpqa`, `planning`, `moral`, `idea`
- 13,375 generated traces
- Track A behavior counts for 13,374 traces
- Track B per-sentence labels for 6,004,702 segments
- Deterministic grades for math/gpqa/planning, sandboxed grades for code, rubric quality for moral/idea

Large parquet files are intentionally tracked with Git LFS. Before pushing or cloning:

```bash
git lfs install
git lfs track "*.parquet" "*.pdf"
```

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

See `DATASET.md` for a data dictionary and `RUNBOOK.md` for end-to-end commands.

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

1. Prepare task parquets:

```bash
python scripts/02_prepare_tasks.py --config configs/exp.yaml
```

2. Generate traces, one job per model:

```bash
sbatch -w mira scripts/blackwell.sbatch generate qwen35_9b
```

3. Judge behaviors:

```bash
sbatch -w vega scripts/blackwell.sbatch judge google/gemma-4-31B-it A
sbatch -w mira scripts/blackwell.sbatch judge google/gemma-4-31B-it B
```

4. Grade performance:

```bash
python -m src.perf.grade --traces-glob "data/traces/traces_*.parquet" --out data/perf/success_grades.parquet
python -m src.perf.grade_code_exec --traces-glob "data/traces/traces_*.parquet" --out data/perf/code_grades.parquet
sbatch -w vega scripts/blackwell.sbatch quality google/gemma-4-31B-it
```

5. Analyze and export dashboard data:

```bash
python -m src.analysis.model_similarity --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet --kind shape
python -m src.analysis.model_similarity --trackB data/judge/prod/trackB_full__google_gemma-4-31B-it.parquet --kind mag
python -m src.analysis.export_dashboard --out-dir docs/data
```

## Adding Another Model

Add a new entry under `gen_models:` in `configs/exp.yaml` with:

- `hf_id`
- `kind`: `reasoning` or `non_reasoning`
- `analysis_source`: usually `think_text` for reasoning models and `answer_text` for anchors
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
compact JSON from the canonical parquets with:

```bash
python -m src.analysis.export_dashboard --out-dir docs/data --samples-per-cell 12
```

Run it locally with any static server:

```bash
python -m http.server 8000 -d docs
```

Open `http://127.0.0.1:8000/`.

The dashboard samples raw trace text for browser speed. The full raw prompt,
thinking, and answer fields remain in `data/traces/*.parquet`.
