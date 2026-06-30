# Design Notes

This file records the current public-facing design decisions for the cleaned repo.
Older exploratory probes and stale five-domain notes were removed from the shareable
surface; the working six-domain experiment is the source of truth.

## Scope

Domains:

- `math`: MATH-500
- `code`: LiveCodeBench code generation lite, 2025 window
- `gpqa`: GPQA Diamond topped up with Extended
- `planning`: ACPBench MCQ competencies
- `moral`: MoReBench
- `idea`: LiveIdeaBench keyword-conditioned idea generation

Generation conditions:

- `anchor`: non-reasoning Llama-3.1-8B-Instruct control
- `reasoner`: DeepSeek-R1-Distill-Llama-8B
- `qwen35_4b`, `qwen35_9b`, `qwen35_27b`: Qwen3.5 scale ladder

The main extension question is whether deliberation heartbeat shape is stable across
scale within a family and different across training lineages.

## Behavior Taxonomy

Conversational behaviors from Kim:

- `Question_and_Answering`
- `Perspective_Shift`
- `Conflict_of_Perspectives`
- `Reconciliation`

Cognitive behaviors from Gandhi:

- `verification`
- `backtracking`
- `subgoal`
- `backward_chaining`

## Labeling Tracks

Track A is whole-trace behavior counts. It is non-temporal and remains the aggregate
composition view.

Track B is per-sentence multi-label presence on the deterministic ThinkARM split. It
is the primary temporal heartbeat dataset. The full-context pass is the default
temporal view; the isolated pass is retained for context-dependence checks.

## Thinking Normalization

Downstream code expects canonical `<think>...</think>` delimiters. Model-specific
formats are normalized at capture time in `src/generate/thinking_spec.py`, so parsing,
segmentation, judging, and dashboard export stay model-agnostic.

When adding a model, update `configs/exp.yaml` and add a `ThinkingSpec` only if its
delimiter or thinking-enable behavior differs from the existing styles.

## Performance Measures

- math: symbolic/string answer match
- gpqa/planning: MCQ letter extraction
- code: sandboxed LiveCodeBench execution
- moral: rubric checklist quality score
- idea: originality and feasibility score

Mechanism analysis is per-domain. Do not pool domains for behavior-to-success claims;
the sign and meaning of behavior features differs by task family.

## Dashboard

The dashboard is named Thought Atlas and lives in `docs/` for GitHub Pages. It consumes
compact JSON produced by `src.analysis.export_dashboard`; the full raw dataset remains
in parquet under `data/`.

Dashboard trace text is sampled and clipped for browser performance. Full prompt,
thinking, and answer text remains available in the raw trace parquets.
