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
import re
from typing import Optional

import polars as pl

MAX_PROMPT_CHARS = 16000  # ~4 chars/token, ~4K-token prompt cap


def _hash_norm(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _filter_and_sample(df: pl.DataFrame, *, n: int, seed: int,
                       strata_col: Optional[str] = None) -> pl.DataFrame:
    df = df.filter(pl.col("prompt").str.len_chars() <= MAX_PROMPT_CHARS)
    df = df.with_columns(
        pl.col("prompt").map_elements(_hash_norm, return_dtype=pl.Utf8).alias("_hash")
    ).unique(subset=["_hash"], maintain_order=True).drop("_hash")
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
    quotas, allocated = {}, 0
    for row in counts.iter_rows(named=True):
        q = min(int(round((row["len"] / total) * n)), row["len"])
        quotas[row[strata_col]] = q
        allocated += q
    remainder = n - allocated
    if remainder > 0:
        for st in counts.sort("len", descending=True)[strata_col].to_list():
            if remainder <= 0:
                break
            avail = df_s.filter(pl.col(strata_col) == st).height - quotas[st]
            if avail > 0:
                add = min(remainder, avail)
                quotas[st] += add
                remainder -= add
    parts = []
    for st, k in quotas.items():
        if k > 0:
            parts.append(df_s.filter(pl.col(strata_col) == st).sample(n=k, seed=seed, shuffle=True))
    return pl.concat(parts).sample(fraction=1.0, seed=seed, shuffle=True)


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
                "fn_name": (ex.get("metadata") or {}).get("func_name") if isinstance(ex.get("metadata"), dict) else None,
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
                instruction: str) -> tuple[str, str]:
    """Shuffle options into a stable A/B/C/D order (seeded by instance), return (prompt, correct_letter)."""
    import random
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    letters = [chr(65 + i) for i in range(len(options))]  # A,B,C,D
    correct_letter = letters[order.index(correct_idx)]
    lines = [f"{letters[i]}. {options[oi]}" for i, oi in enumerate(order)]
    prompt = f"{stem.strip()}\n\n" + "\n".join(lines) + f"\n\n{instruction}"
    return prompt, correct_letter


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
            stem = f"{ex.get('context','')}\n\n{ex.get('question','')}".strip()
            iid = ex.get("id")
            prompt, correct_letter = _format_mcq(stem, list(texts), correct_idx,
                                                 seed=hash((cfg, iid)) & 0xFFFFFFFF, instruction=instr)
            rows.append({
                "instance_id": f"planning:{cfg}:{iid}", "task_type": "planning", "prompt": prompt,
                "reference_answer": correct_letter, "difficulty_raw": ex.get("group"),
                "_strata": ex.get("group"),
                "metadata": json.dumps({
                    "acp_config": cfg, "group": ex.get("group"), "n_options": len(texts),
                    "is_mcq": True, "source_dataset": hf_id,
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
            prompt, correct_letter = _format_mcq(q, opts, 0, seed=hash(rid) & 0xFFFFFFFF, instruction=instr)
            out.append({
                "instance_id": f"gpqa:{rid}", "task_type": "gpqa", "prompt": prompt,
                "reference_answer": correct_letter,
                "difficulty_raw": ex.get("High-level domain"),
                "_strata": ex.get("High-level domain"), "_rid": rid,
                "metadata": json.dumps({
                    "gpqa_subset": gpqa_subset, "subdomain": ex.get("Subdomain"),
                    "high_level_domain": ex.get("High-level domain"), "is_mcq": True,
                    "writer_difficulty": ex.get("Writer's Difficulty Estimate"),
                    "source_dataset": hf_id,
                }, ensure_ascii=False),
            })
        return out

    diamond = _rows_from(primary_config, "diamond")
    seen = {r["_rid"] for r in diamond}
    need = max(0, n - len(diamond))
    fill = []
    if need > 0:
        for r in _rows_from(fill_config, "extended"):
            if r["_rid"] not in seen:
                fill.append(r); seen.add(r["_rid"])
    df = pl.DataFrame(diamond + fill).drop("_rid")
    # take all diamond + stratified fill; if pool < n, _filter_and_sample raises (informative)
    return _filter_and_sample(df, n=min(n, df.height), seed=seed, strata_col="_strata").drop("_strata")


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
