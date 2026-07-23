# Generation and scoring audit

Audit date: 2026-07-22. The checked-in dataset is retained as legacy v1; all
corrected artifacts must be rebuilt under `data/v2/`.

## Why visible answers became blank or incorrect

There were three independent failure layers:

1. The generation parser required `</think>`. A reasoning model that stopped with
   a direct answer but no close tag produced an empty `answer_text`; length-truncated
   and normally stopped outputs were not distinguished. Outputs with repeated
   think/answer cycles were split at the first close, so interim reasoning and later
   cycles contaminated the answer.
2. The math grader relied too heavily on one output format. Qwen frequently gave a
   correct terminal answer without the exact expected box/marker, so a nonblank
   `answer_text` still produced a blank prediction. The MCQ extractor could select an
   earlier bold option discussed in the reasoning instead of the later explicit
   final choice.
3. Parse failure was stored as `success=0`. This made extraction coverage look like
   model accuracy and prevented the failures from being found through null/coverage
   checks.

V2 distinguishes stopped direct-answer fallback from truncation, uses the suffix
after the last close tag, records delimiter diagnostics, extracts several ordered
final-answer candidates, uses `math-verify` plus normalized comparison, selects the
latest explicit MCQ conclusion, and records unparsed results as null. Every grade
stores prediction, reference, method/status, version, and answer hash.

## Follow-up hardening

A second adversarial review found two remaining objective-grading errors and one
capture error:

- math grading searched every extracted number and could mark a response correct
  when an earlier echoed problem value matched the reference despite a different
  boxed final answer;
- a weak trailing MCQ letter could override an earlier explicit final answer; and
- vLLM's default `skip_special_tokens=true` could remove reasoning delimiters before
  parsing, making the answer blank or mixing answer prose into reasoning analysis.

Generation v3 preserves special tokens and excludes unseparated fallback responses
from reasoning-only analysis. Objective v3 grades only the strongest math candidate,
prevents terminal MCQ recaps from overriding explicit conclusions, and recognizes
hedged final-answer language. Length-truncated generations are ungradeable/null
across objective, code, and subjective scoring rather than being promoted from an
unfinished reasoning fragment.

The production pipeline also has a reference-blind answer-extraction stage over
every task. It receives the task and raw response but never the reference, rubric,
or success label. Guided JSON outputs are accepted only with verbatim evidence,
exact source/input hashes, and task-specific validation. Code is selected as an
unchanged source block; moral/idea answers are exact response spans. Deterministic
symbolic, exact-letter, execution, and rubric graders still determine correctness
or quality. Extraction shards are resumable, fingerprinted, coverage-validated,
and merged only when complete and disjoint.

## Other defects found

- All 500 legacy ACP planning prompts included the source option mapping and a
  separately reshuffled option mapping. In 349 prompts the mappings differed. The
  model therefore saw contradictory letter semantics. V2 removes the embedded block
  before a stable reshuffle and validates exactly one final block.
- Sampling used Python's process-randomized `hash`, and request seeds were not
  actually passed into vLLM. V2 uses SHA-256 seeds, per-request `SamplingParams`,
  deterministic trace IDs, actual decode metadata, task/generation fingerprints,
  and incremental atomic checkpoints.
- The legacy GPQA sample retained 182 of 198 Diamond records. V2 retains the entire
  filtered Diamond set first, then fills from de-duplicated Extended records.
- LiveCodeBench prompts did not constrain language although the runner executed only
  Python. Functional method names were missing, string outputs were compared against
  quoted literals incorrectly, and the reported suite was capped below the full
  private suite. V2 requires Python 3, derives the target method, handles common
  annotation/import names and typed outputs, grades all tests by default, and marks
  unsupported/infrastructure cases ungradeable instead of wrong.
- Moral rubrics contain negative weights in 359/500 legacy tasks. Dividing signed
  contributions by the signed weight sum yielded 56 published scores outside
  `[0,1]`. V2 treats negative criteria as penalties, divides by absolute-weight mass,
  requires complete verdict arrays, and preserves raw verdicts/weights. Idea scoring
  similarly requires every configured axis in range.
- Track A saw reasoning text while Track B preferred full text, so the two measures
  labeled different content. Judge parse omissions became zero counts/labels. V2
  uses the same reasoning-only scope, requires complete unique sentence indices,
  stores parse/truncation flags and raw outputs, and retries invalid rows.
- Broad trace globs loaded both canonical files and their shards. Generation merged
  even after worker failure, overwrote outputs without fingerprint validation, and
  deleted shards by default; judge/quality shards had no production merger. V2 uses
  validated discovery, exact shard coverage, overlap/fingerprint checks, atomic
  canonical writes and manifests, retains shards, and refuses to merge after any
  worker failure.

## Legacy dataset audit counts

- 13,375 traces, all without v2 generation provenance
- 176 blank stored answers; 4 stopped outputs are directly recoverable
- 126 outputs with multiple close tags
- 2,009 unparsed rows with a numeric score in the broad legacy grade set
- 182/198 GPQA Diamond records
- 56 quality scores outside `[0,1]`
- 3,000 missing moral/idea quality rows (the three Qwen generators)
- behavior and quality tables lack score versions and parse-validity flags

Run `scripts/audit_pipeline.py` after each production stage. Analysis and dashboard
export should begin only after `--strict-v2` succeeds on the relevant artifacts.
