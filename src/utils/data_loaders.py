"""src/utils/data_loaders.py — Load + normalize task domains to a unified schema.

Unified row:
    instance_id, task_type, prompt, reference_answer (str|None),
    difficulty_raw (Any), metadata (JSON str)

math/code/moral loaders ported from society-task-exp1 (verified working there).
All loaders share _filter_and_sample (length filter + dedup + stratified sample).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Optional

import polars as pl

MAX_PROMPT_CHARS = 16000  # ~4 chars/token, ~4K-token prompt cap


def _hash_norm(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _stable_seed(*parts, base_seed: int = 0) -> int:
    """Process-independent 32-bit seed derived from stable content.

    Python's built-in ``hash`` is salted per process, so it must never be used for
    persisted option permutations or dataset sampling.
    """
    payload = json.dumps([base_seed, *parts], ensure_ascii=False, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _filter_dedup(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the common prompt-length filter and stable prompt de-duplication."""
    return (df.filter(pl.col("prompt").str.len_chars() <= MAX_PROMPT_CHARS)
              .with_columns(pl.col("prompt").map_elements(
                  _hash_norm, return_dtype=pl.Utf8).alias("_hash"))
              .unique(subset=["_hash"], maintain_order=True)
              .drop("_hash"))


def _filter_and_sample(df: pl.DataFrame, *, n: int, seed: int,
                       strata_col: Optional[str] = None) -> pl.DataFrame:
    df = _filter_dedup(df)
    if df.height < n:
        import warnings
        warnings.warn(f"Only {df.height} instances after filtering; requested n={n}. "
                      f"Using all {df.height} (uneven N is fine for the between-domain contrast).")
        n = df.height

    use_strata = (strata_col is not None and strata_col in df.columns
                  and df[strata_col].null_count() < df.height)
    if not use_strata:
        return df.sample(n=n, seed=seed, shuffle=True)

    df_s = df.filter(pl.col(strata_col).is_not_null())
    if df_s.height < n:
        return df.sample(n=n, seed=seed, shuffle=True)
    counts = df_s.group_by(strata_col).len().sort(strata_col)
    total = counts["len"].sum()
    # Largest-remainder allocation guarantees exactly n rows. Independently
    # rounding each stratum can over-allocate and return more than requested.
    quotas, fractions = {}, []
    for row in counts.iter_rows(named=True):
        exact = (row["len"] / total) * n
        q = min(math.floor(exact), row["len"])
        quotas[row[strata_col]] = q
        fractions.append((exact - q, str(row[strata_col]), row[strata_col], row["len"]))
    remainder = n - sum(quotas.values())
    for _, _, st, available in sorted(fractions, reverse=True):
        if remainder <= 0:
            break
        if quotas[st] < available:
            quotas[st] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError(f"could not allocate exact stratified sample: {remainder=} {n=}")
    parts = []
    for st, k in quotas.items():
        if k > 0:
            parts.append(df_s.filter(pl.col(strata_col) == st).sample(
                n=k, seed=_stable_seed("stratum", st, base_seed=seed), shuffle=True))
    out = pl.concat(parts).sample(fraction=1.0, seed=seed, shuffle=True)
    if out.height != n:
        raise RuntimeError(f"stratified sampler returned {out.height} rows; expected {n}")
    return out


# ----------------------------------------------------------------- math
def load_math500(n=100, seed=42, *, hf_id="HuggingFaceH4/MATH-500") -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(hf_id, split="test")
    rows = [{
        "instance_id": f"math:{ex.get('unique_id', ex.get('id', ex['problem'][:40]))}",
        "task_type": "math", "prompt": ex["problem"],
        "reference_answer": str(ex.get("answer", "")) or None,
        "difficulty_raw": ex.get("level"),
        "metadata": json.dumps({"subject": ex.get("subject"), "solution": ex.get("solution"),
                                "source_dataset": hf_id}, ensure_ascii=False),
    } for ex in ds]
    return _filter_and_sample(pl.DataFrame(rows), n=n, seed=seed, strata_col="difficulty_raw")


# ----------------------------------------------------------------- code
def load_livecodebench(n=100, seed=42, *, hf_id="livecodebench/code_generation_lite",
                       version_files: Optional[list[str]] = None,
                       post_date: Optional[str] = "2025-01-01") -> pl.DataFrame:
    from huggingface_hub import hf_hub_download
    version_files = version_files or ["test6.jsonl"]
    rows_raw = []
    for fname in version_files:
        path = hf_hub_download(repo_id=hf_id, filename=fname, repo_type="dataset")
        with open(path) as f:
            rows_raw += [json.loads(l) for l in f if l.strip()]
    rows = []
    for ex in rows_raw:
        cd = ex.get("contest_date", "")
        if post_date and cd and cd[:10] < post_date:
            continue
        parts = [f"# {ex.get('question_title', 'Coding Problem')}", "", ex.get("question_content", "")]
        starter = ex.get("starter_code", "")
        if starter:
            parts += ["", "Starter code:", "```python", starter, "```"]
        parts += ["", "Return a complete Python 3 solution in one ```python fenced code block."]
        raw_meta = ex.get("metadata") or {}
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except Exception:
                raw_meta = {}
        fn_name = raw_meta.get("func_name") if isinstance(raw_meta, dict) else None
        if not fn_name and starter:
            m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", starter)
            fn_name = m.group(1) if m else None
        rows.append({
            "instance_id": f"code:{ex.get('question_id', ex.get('platform','?'))}:{ex.get('question_title','')[:40]}",
            "task_type": "code", "prompt": "\n".join(parts).strip(),
            "reference_answer": None, "difficulty_raw": ex.get("difficulty"),
            "metadata": json.dumps({
                "platform": ex.get("platform"), "contest_date": cd,
                "question_id": ex.get("question_id"), "starter_code": starter,
                # test cases carried for the execution-based perf metric:
                "public_test_cases": ex.get("public_test_cases"),
                "private_test_cases": ex.get("private_test_cases"),
                "fn_name": fn_name,
                "required_language": "python3",
                "source_dataset": hf_id, "source_files": version_files,
            }, ensure_ascii=False),
        })
    return _filter_and_sample(pl.DataFrame(rows), n=n, seed=seed, strata_col="difficulty_raw")


# ----------------------------------------------------------------- moral
def load_morebench(n=100, seed=42, *, hf_id="morebench/morebench",
                   config_name="morebench_public") -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(hf_id, config_name)
    split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
    rows = []
    for ex in split:
        dilemma = ex.get("DILEMMA") or ex.get("dilemma") or ex.get("scenario")
        if not dilemma:
            continue
        dil_type = ex.get("DILEMMA_TYPE") or ex.get("dilemma_type")
        idx = ex.get("id") or ex.get("ID") or _hash_norm(dilemma)[:16]
        rows.append({
            "instance_id": f"moral:{idx}", "task_type": "moral",
            "prompt": f"{dilemma}\n\nReason carefully through this dilemma and explain your conclusion.",
            "reference_answer": None, "difficulty_raw": None,
            "_strata": dil_type,
            "metadata": json.dumps({
                "theory": ex.get("THEORY") or ex.get("theory"),
                "dilemma_source": ex.get("DILEMMA_SOURCE") or ex.get("dilemma_source"),
                "dilemma_type": dil_type,
                "rubric": ex.get("RUBRIC") or ex.get("rubric"),   # for moral_rubric perf metric
                "role_domain": ex.get("ROLE_DOMAIN") or ex.get("role_domain"),
                "context": ex.get("CONTEXT") or ex.get("context"),
                "source_dataset": hf_id, "config": config_name,
            }, ensure_ascii=False),
        })
    out = _filter_and_sample(pl.DataFrame(rows), n=n, seed=seed, strata_col="_strata")
    return out.drop("_strata")



# ----------------------------------------------------------------- MCQ helper (gpqa, acp)
def _format_mcq(stem: str, options: list[str], correct_idx: int, seed: int,
                instruction: str) -> tuple[str, str, list[int]]:
    """Shuffle options into a stable order; return prompt, correct letter, permutation."""
    import random
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    letters = [chr(65 + i) for i in range(len(options))]  # A,B,C,D
    correct_letter = letters[order.index(correct_idx)]
    lines = [f"{letters[i]}. {options[oi]}" for i, oi in enumerate(order)]
    prompt = f"{stem.strip()}\n\n" + "\n".join(lines) + f"\n\n{instruction}"
    return prompt, correct_letter, order


def _strip_embedded_mcq_options(question: str, labels: list[str]) -> str:
    """Remove an A/B/C/D block already embedded at the end of an ACP question.

    ACPBench exposes the choices both inside ``question`` and in the structured
    ``choices`` field. Keeping both and then shuffling the structured list creates
    two contradictory letter mappings in one prompt.
    """
    if not question or len(labels) < 2:
        return (question or "").strip()
    matches = list(re.finditer(r"(?:^|\s)A\.\s", question))
    for match in reversed(matches):
        suffix = question[match.start():]
        cursor = 0
        valid = True
        for label in labels:
            m = re.search(rf"(?:^|\s){re.escape(label)}\.\s", suffix[cursor:])
            if not m:
                valid = False
                break
            cursor += m.end()
        if valid:
            return question[:match.start()].strip()
    return question.strip()


# ----------------------------------------------------------------- planning (ACPBench, pooled)
ACP_MCQ_CONFIGS = ["acp_reach_mcq", "acp_prog_mcq", "acp_app_mcq", "acp_val_mcq", "acp_land_mcq"]

def load_acpbench(n=500, seed=42, *, hf_id="ibm/acp_bench",
                  configs=tuple(ACP_MCQ_CONFIGS)) -> pl.DataFrame:
    """ACPBench planning competencies, pooled across MCQ task-types, stratified by competency.

    Schema (confirmed): id, group, context, question, choices={'label':[...],'text':[...]},
    query, answer (the correct LABEL letter). We pool reach/prog/app/val/land (each ~130 test)
    -> ~650 available, stratify by the `group` field so the 500 sample spans planning sub-skills.
    Performance = MCQ letter accuracy (mechanical). We re-shuffle choices (seeded) so option
    position can't leak, and store the canonical correct letter.
    """
    from datasets import load_dataset
    instr = "Reason step by step, then end with your final answer as a single letter."
    rows = []
    for cfg in configs:
        try:
            d = load_dataset(hf_id, cfg)
        except Exception as e:  # skip a config that isn't present rather than abort the pool
            print(f"[acp] skipping {cfg}: {e}")
            continue
        split = d["test"] if "test" in d else d[list(d.keys())[0]]
        for ex in split:
            ch = ex.get("choices") or {}
            labels, texts = ch.get("label", []), ch.get("text", [])
            ans = ex.get("answer")
            if not texts or ans is None or ans not in labels:
                continue
            correct_idx = labels.index(ans)
            raw_question = ex.get("question", "")
            question = _strip_embedded_mcq_options(raw_question, list(labels))
            stem = f"{ex.get('context','')}\n\n{question}".strip()
            iid = ex.get("id")
            option_seed = _stable_seed("acp", cfg, iid, base_seed=seed)
            prompt, correct_letter, order = _format_mcq(
                stem, list(texts), correct_idx, seed=option_seed, instruction=instr)
            rows.append({
                "instance_id": f"planning:{cfg}:{iid}", "task_type": "planning", "prompt": prompt,
                "reference_answer": correct_letter, "difficulty_raw": ex.get("group"),
                "_strata": ex.get("group"),
                "metadata": json.dumps({
                    "acp_config": cfg, "group": ex.get("group"), "n_options": len(texts),
                    "is_mcq": True, "source_dataset": hf_id,
                    "source_correct_label": ans, "option_seed": option_seed,
                    "option_permutation": order,
                    "embedded_option_block_removed": question.strip() != raw_question.strip(),
                }, ensure_ascii=False),
            })
    return _filter_and_sample(pl.DataFrame(rows), n=n, seed=seed, strata_col="_strata").drop("_strata")


# ----------------------------------------------------------------- science MCQ (GPQA)
def load_gpqa(n=500, seed=42, *, hf_id="Idavidrein/gpqa",
              primary_config="gpqa_diamond", fill_config="gpqa_extended") -> pl.DataFrame:
    """GPQA hard-science 4-way MCQ. Take ALL of Diamond (198, expert-validated), then top up to
    n from `fill_config` MINUS Diamond Record IDs (main/extended are supersets — dedup needed).

    Schema (confirmed): 'Question', 'Correct Answer', 'Incorrect Answer 1..3', 'Subdomain',
    'High-level domain', 'Record ID'. Performance = MCQ letter accuracy. NOTE: 8B models floor
    near chance here — analyze trace STRUCTURE regardless of correctness; the correct/incorrect
    bifurcation is underpowered-by-design for GPQA (documented in PREREG).
    """
    from datasets import load_dataset, get_dataset_config_names
    instr = "Reason step by step, then end with your final answer as a single letter."

    def _rows_from(cfg, gpqa_subset):
        d = load_dataset(hf_id, cfg)
        split = d["test"] if "test" in d else d[list(d.keys())[0]]
        out = []
        for ex in split:
            q = ex.get("Question"); corr = ex.get("Correct Answer")
            wrongs = [ex.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
            if not q or not corr or any(w is None for w in wrongs):
                continue
            opts = [corr] + wrongs  # correct at idx 0 before shuffle
            rid = ex.get("Record ID") or _hash_norm(q)
            option_seed = _stable_seed("gpqa", rid, base_seed=seed)
            prompt, correct_letter, order = _format_mcq(
                q, opts, 0, seed=option_seed, instruction=instr)
            out.append({
                "instance_id": f"gpqa:{rid}", "task_type": "gpqa", "prompt": prompt,
                "reference_answer": correct_letter,
                "difficulty_raw": ex.get("High-level domain"),
                "_strata": ex.get("High-level domain"), "_rid": rid,
                "metadata": json.dumps({
                    "gpqa_subset": gpqa_subset, "subdomain": ex.get("Subdomain"),
                    "high_level_domain": ex.get("High-level domain"), "is_mcq": True,
                    "writer_difficulty": ex.get("Writer's Difficulty Estimate"),
                    "source_dataset": hf_id, "option_seed": option_seed,
                    "option_permutation": order, "n_options": len(opts),
                }, ensure_ascii=False),
            })
        return out

    diamond_rows = _rows_from(primary_config, "diamond")
    diamond = _filter_dedup(pl.DataFrame(diamond_rows))
    if n <= diamond.height:
        return _filter_and_sample(diamond, n=n, seed=seed, strata_col="_strata").drop(
            ["_strata", "_rid"])

    seen = set(diamond["_rid"].to_list())
    fill_rows = []
    for row in _rows_from(fill_config, "extended"):
        if row["_rid"] not in seen:
            fill_rows.append(row)
            seen.add(row["_rid"])
    fill = _filter_dedup(pl.DataFrame(fill_rows))
    need = n - diamond.height
    sampled_fill = _filter_and_sample(fill, n=need, seed=seed, strata_col="_strata")
    out = pl.concat([diamond, sampled_fill], how="diagonal_relaxed")
    out = out.sample(fraction=1.0, seed=seed, shuffle=True)
    if out.filter(pl.col("metadata").str.contains('"gpqa_subset": "diamond"')).height != diamond.height:
        raise RuntimeError("GPQA Diamond rows were lost during fill sampling")
    return out.drop(["_strata", "_rid"])


# ----------------------------------------------------------------- abductive (LiveIdeaBench)
def load_liveideabench(n=500, seed=42, *, hf_id="6cf/LiveIdeaBench") -> pl.DataFrame:
    """LiveIdeaBench abductive/divergent: dataset rows are OTHER models' ideas; the TASK is to
    generate a scientific idea from a `keywords` seed. We dedup to unique keywords (1180 avail),
    sample n, and wrap in the idea-generation instruction. Performance = LLM-judged originality
    + feasibility (rubric judge, held-out family) — NOT ground-truth; documented in PREREG.
    """
    from datasets import load_dataset
    d = load_dataset(hf_id)
    split = d["train"] if "train" in d else d[list(d.keys())[0]]
    keywords = sorted(set(k for k in split["keywords"] if k))
    instr = ("You are a scientist. Propose one novel, specific research idea related to the "
             "keyword below. State the idea and briefly justify its originality and feasibility.")
    rows = [{
        "instance_id": f"idea:{_hash_norm(kw)}", "task_type": "idea",
        "prompt": f"Keyword: {kw}\n\n{instr}", "reference_answer": None,
        "difficulty_raw": None, "_strata": None,
        "metadata": json.dumps({"keyword": kw, "judge_scored": True,
                                "rubric": ["originality", "feasibility"],
                                "source_dataset": hf_id}, ensure_ascii=False),
    } for kw in keywords]
    return _filter_and_sample(pl.DataFrame(rows), n=n, seed=seed, strata_col=None).drop("_strata")


LOADERS = {
    "math": load_math500, "code": load_livecodebench, "moral": load_morebench,
    "planning": load_acpbench, "gpqa": load_gpqa, "idea": load_liveideabench,
}
