"""utils/characterize.py — Post-hoc trace characterization.

Pure analysis utilities; no model loading, no GPU. Used to label each trace
with categorical fields that downstream analysis can bifurcate on:

    - loop_collapse: strict literal repetition (alphabet-style)
    - soft_loop:     n-gram-overlap repetition (what literal misses)
    - failure_mode:  one of {completed, truncated_productive, soft_loop,
                              hard_loop, empty}
    - is_correct:    answer matches reference (for math, code; None for moral)

Note: failures are NOT filtered. They are LABELED. The descriptive society-
of-thought experiment treats failure modes as data — different failure
modes plausibly have distinct activation/behavioral profiles worth measuring.

This module is designed to be re-run as detectors are refined. Each call is
idempotent given the same input traces.
"""

from __future__ import annotations

import re
from typing import Optional


# -----------------------------------------------------------------------------
# Loop / repetition detectors
# -----------------------------------------------------------------------------

def loop_collapse_detect(text: str, *, tail_chars: int = 200,
                         min_total_chars: int = 1000) -> bool:
    """Strict literal repetition: tail substring appears earlier verbatim.

    Catches the catastrophic failure mode where the model emits a templated
    phrase ("Let A = f(A). Let B = g(B)...") cycling through the same
    structure for hundreds of tokens.
    """
    if not text or len(text) < min_total_chars:
        return False
    tail = text[-tail_chars:]
    body = text[:-tail_chars]
    if len(body) < tail_chars:
        return False
    return tail in body


def _tokenize_for_ngrams(text: str) -> list[str]:
    """Lightweight tokenization for n-gram analysis: split on whitespace,
    normalize numbers and very long tokens. NOT model tokenization.
    """
    # Replace runs of digits with #NUM# so "284" loops aren't masked by digit choice
    text = re.sub(r"\d+", "#NUM#", text)
    return text.split()


def soft_loop_detect(text: str, *, n: int = 8, tail_words: int = 600,
                     overlap_threshold: float = 0.5,
                     min_words: int = 200) -> bool:
    """N-gram overlap: tail of trace re-uses many n-grams from earlier body.

    Catches softer loops where the model repeats itself with minor variations
    ("Then S(284) = 284. Then the sum of proper divisors of 284 is 284. Then
    the answer is 284. But 284 is not a perfect number. Wait, 284 is not a
    perfect number...").

    Default: 8-grams, last 600 words, >50% of tail n-grams seen earlier.
    """
    if not text or len(text) < 2000:
        return False
    words = _tokenize_for_ngrams(text)
    if len(words) < min_words:
        return False
    if len(words) <= tail_words + n:
        return False

    body_words = words[:-tail_words]
    tail = words[-tail_words:]
    if len(body_words) < n or len(tail) < n:
        return False

    body_ngrams = set(
        tuple(body_words[i:i + n]) for i in range(len(body_words) - n + 1)
    )
    tail_ngrams = [
        tuple(tail[i:i + n]) for i in range(len(tail) - n + 1)
    ]
    if not tail_ngrams:
        return False

    overlap = sum(1 for ng in tail_ngrams if ng in body_ngrams)
    return overlap / len(tail_ngrams) > overlap_threshold


# -----------------------------------------------------------------------------
# Failure-mode classifier
# -----------------------------------------------------------------------------

# Order matters: we test most-specific first. A trace with literal repetition
# also has soft-loop overlap; we want to call it a hard_loop, not a soft_loop.
def classify_failure_mode(*, text: str, completed: bool,
                          finish_reason: str) -> str:
    """Categorical descriptor of how the trace ended.

    Returns one of:
        completed             — </think> closed, answer followed, finish=stop
        empty                 — no/very-little content (e.g. immediate eos)
        hard_loop             — literal-repetition collapse
        soft_loop             — n-gram overlap repetition
        truncated_productive  — hit max_tokens with no obvious loop pattern
        unknown               — should not occur; debug if seen
    """
    if completed:
        return "completed"
    if not text or len(text.strip()) < 100:
        return "empty"
    if loop_collapse_detect(text):
        return "hard_loop"
    if soft_loop_detect(text):
        return "soft_loop"
    if finish_reason == "length":
        return "truncated_productive"
    return "unknown"


# -----------------------------------------------------------------------------
# Correctness (math + code only; moral has no reference answer)
# -----------------------------------------------------------------------------

def _normalize_math_answer(s: str) -> str:
    """Best-effort normalization for math-answer string comparison.

    Aggressive: strips whitespace, LaTeX wrappers, dollar signs, markdown bold
    markers, common punctuation, leading/trailing units words.

    Note: not bulletproof. False negatives expected where the model's
    expression form differs from the reference (e.g., '4/3' vs '\\frac{4}{3}'
    vs '1.333'). For Phase 1 this is signal, not ground truth.
    """
    s = s.strip()
    # Strip markdown bold/italic
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"(?<!\\)\*", "", s)  # single asterisk not preceded by backslash
    # Strip surrounding $...$ or \(...\) or \[...\]
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(\s*|\s*\\\)$", "", s).strip()
    s = re.sub(r"^\\\[\s*|\s*\\\]$", "", s).strip()
    # Strip \boxed{...}
    m = re.match(r"^\\boxed\{(.*)\}$", s)
    if m:
        s = m.group(1).strip()
    # Strip trailing period
    s = s.rstrip(".")
    # Collapse whitespace, lowercase
    s = re.sub(r"\s+", "", s)
    return s.lower()


def parse_math_answer(answer_text: str) -> Optional[str]:
    """Best-effort extraction of the final answer from a math response.

    Tries several patterns in priority order, returns the LAST match
    (model's final answer is at the end of the response). Patterns:

        1. \\boxed{X}                      — formal MATH convention
        2. **Answer:** X / Answer: X       — explicit label
        3. final markdown bold near end    — **N** as the closing emphasized
                                              answer (Qwen3.5's most common)
        4. "the answer is X" / "= X"       — natural-language closing

    Returns None if no clear answer found.
    """
    if not answer_text:
        return None

    text = answer_text.strip()

    # Pattern 1: \boxed{...} — most reliable when present
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        return _normalize_math_answer(boxes[-1])

    # Pattern 2: explicit "Answer:" label (with optional ** wrappers)
    # Match "**Answer:** X", "Answer: X", "**Answer**: X" near end
    answer_label = re.findall(
        r"\*{0,2}answer\*{0,2}\s*:\*{0,2}\s*([^\n]{1,200})",
        text, re.IGNORECASE,
    )
    if answer_label:
        # Take the last; sometimes "Step 1: Answer the question" appears earlier
        candidate = answer_label[-1].strip()
        # Trim trailing prose if any: take first chunk before two consecutive newlines
        # or before a period followed by capital letter (sentence break)
        candidate = re.split(r"\.\s+[A-Z]|\n\n", candidate)[0].strip()
        return _normalize_math_answer(candidate)

    # Pattern 3: final markdown bold near end of text — Qwen3.5's natural style
    # Look at last ~500 chars; find all **...** chunks; take the last one that
    # looks like an answer (numeric, fraction, or short expression).
    tail = text[-500:]
    bolds = re.findall(r"\*\*([^*]{1,80})\*\*", tail)
    # Filter out section headers like "Conclusion", "Step 1", "Answer" itself
    answer_like = []
    for b in bolds:
        b_strip = b.strip().rstrip(".")
        # Skip prose-y bolds (more than 4 words = probably a heading)
        if len(b_strip.split()) > 4:
            continue
        # Skip if it's just a label
        if b_strip.lower() in {"answer", "conclusion", "final answer", "result"}:
            continue
        answer_like.append(b_strip)
    if answer_like:
        return _normalize_math_answer(answer_like[-1])

    # Pattern 4: "answer is X" / trailing "= X"
    m = re.search(
        r"(?:answer\s+is|equals?|=)\s+([^\n.]{1,80})\s*\.?\s*$",
        text, re.IGNORECASE,
    )
    if m:
        return _normalize_math_answer(m.group(1))

    return None


def is_correct_math(answer_text: str, reference: Optional[str]) -> Optional[bool]:
    """Compare parsed model answer to reference. Returns None if unable to parse."""
    if not reference:
        return None
    parsed = parse_math_answer(answer_text)
    if parsed is None:
        return None
    return parsed == _normalize_math_answer(reference)


def assess_correctness(*, task_type: str, answer_text: str,
                       reference_answer: Optional[str]) -> Optional[bool]:
    """Dispatch correctness assessment by task type.

    Returns:
        True / False if assessable
        None if not assessable (no reference, parse failure, or moral task)
    """
    if task_type == "math":
        return is_correct_math(answer_text, reference_answer)
    if task_type == "code":
        # Code correctness needs sandboxed execution against test cases —
        # significant infra. Out of scope for Phase 1; left None.
        return None
    if task_type == "moral":
        # No canonical reference; correctness undefined.
        return None
    return None


# -----------------------------------------------------------------------------
# Bulk apply (used by scripts/04_characterize_traces.py)
# -----------------------------------------------------------------------------

def characterize_row(row: dict) -> dict:
    """Compute all post-hoc fields for one trace row. Returns the additions
    to be merged onto the row.
    """
    full_text = row.get("full_text", "")
    completed = bool(row.get("completed", False))
    finish_reason = row.get("finish_reason", "unknown")

    loop_collapse = loop_collapse_detect(full_text)
    soft_loop = soft_loop_detect(full_text)
    failure_mode = classify_failure_mode(
        text=full_text, completed=completed, finish_reason=finish_reason,
    )

    # Correctness needs the reference answer, which lives in instance_metadata
    # (a JSON blob with the source-dataset fields). Math reference is at
    # row['reference_answer'] if we copied it from data_loaders, else None.
    reference = row.get("reference_answer")
    is_correct = assess_correctness(
        task_type=row.get("task_type", "?"),
        answer_text=row.get("answer_text", ""),
        reference_answer=reference,
    )

    return {
        "loop_collapse": loop_collapse,
        "soft_loop": soft_loop,
        "failure_mode": failure_mode,
        "is_correct": is_correct,
    }
