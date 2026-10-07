"""Free-text answer extraction and grading (D-47, option b).

The repo extractor (src/judge/run_answer_extraction.py) forces a JSON grammar whose first field is "status"; under that
grammar gemma-4-31B labels almost every trace "no_answer" before it has looked for the answer (D-46). Without the grammar
it extracts answers correctly. This module keeps the repo's prompt (build_prompt), row format and validation
(_score_rows / validate_extraction) and changes one thing: generation is unconstrained, and the JSON in the reply is
normalised to the schema (status "success" -> "ok", missing optional fields filled, numeric confidence mapped to
high/medium/low) before validation. The raw reply is kept in raw_output.

Subcommands
  extract  run gemma over data/extract_in traces (sharded, resumable) -> data/extract_free/answer_extractions*.parquet
  grade    per-trace grades -> data/extract_free/grades.parquet. A validated "ok" row is used only if its evidence quote
           is verbatim in the final answer (after </think>) and contains the extracted answer; then math is graded with the
           repo grader on the compact extracted answer, MCQ by letter. A validated "no_answer" row grades 0. Anything
           else falls back to the direct grader (bold stripped) and is flagged in grade_source.
  sample   write a stratified sample of 100 traces for hand checking -> data/extract_free/check_sample.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import polars as pl

from .common import DATA

import os
IN = Path(os.environ.get("XF_IN", DATA / "extract_in"))      # override to extract another staged set
OUT = Path(os.environ.get("XF_OUT", DATA / "extract_free"))
MODEL = "google/gemma-4-31B-it"


def lenient_json(text: str | None):
    """safe_json, then a retry with lone backslashes escaped (gemma often writes LaTeX like "3\\sqrt{13}" unescaped)."""
    from src.judge.vllm_engine import safe_json
    d = safe_json(text)
    if isinstance(d, dict) or not isinstance(text, str):
        return d
    t = text.replace("\\", "\\\\").replace("\\\\\\\\", "\\\\").replace('\\\\"', '\\"')
    return safe_json(t)


def normalise(text: str | None) -> str | None:
    d = lenient_json(text)
    if not isinstance(d, dict):
        return text
    st = str(d.get("status", "")).strip().lower()
    fa = d.get("final_answer")
    if st in ("success", "answered", "found", "complete", "completed"):
        st = "ok"
    if st not in ("ok", "no_answer", "ambiguous"):
        st = "ok" if isinstance(fa, str) and fa.strip() else "no_answer"
    conf = d.get("confidence")
    if isinstance(conf, (int, float)):
        conf = "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low"
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    def unctl(x):   # JSON read "\\frac" as form feed + "rac" and "\\begin" as backspace + "egin": restore the LaTeX
        return re.sub(r"\t(?=[A-Za-z])", r"\\t", x.replace("\x0c", "\\f").replace("\x08", "\\b")) if isinstance(x, str) else x
    fa, d["evidence"] = unctl(fa), unctl(d.get("evidence"))
    out = {"status": st, "final_answer": fa if isinstance(fa, str) else ("" if fa is None else str(fa)),
           "evidence": d.get("evidence") if isinstance(d.get("evidence"), str) else "",
           "answer_start_quote": d.get("answer_start_quote") if isinstance(d.get("answer_start_quote"), str) else "",
           "answer_end_quote": d.get("answer_end_quote") if isinstance(d.get("answer_end_quote"), str) else "",
           "code_block_index": d.get("code_block_index") if isinstance(d.get("code_block_index"), int) else -1,
           "confidence": conf}
    if st != "ok":
        out.update(final_answer="", evidence="", answer_start_quote="", answer_end_quote="", code_block_index=-1)
    return json.dumps(out)


def cmd_extract(shard: int, num_shards: int, batch: int, max_tokens: int, max_model_len: int):
    import src.judge.run_answer_extraction as X
    from src.judge.vllm_engine import EngineConfig, build_llm, chat_batch, make_sampling
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"answer_extractions.shard{shard:02d}of{num_shards:02d}.parquet"
    tr = pl.concat([pl.read_parquet(p) for p in sorted(IN.glob("traces_*.parquet"))], how="diagonal_relaxed").sort("trace_id")
    tr = tr.filter(pl.int_range(pl.len()) % num_shards == shard)
    prev = pl.read_parquet(out_path) if out_path.exists() else None
    done = set(prev["trace_id"].to_list()) if prev is not None else set()
    pending = tr.filter(~pl.col("trace_id").is_in(sorted(done)))
    print(f"[free-extract] shard {shard}/{num_shards}: {tr.height} total, {len(done)} done, {pending.height} pending", flush=True)
    if pending.height == 0:
        return
    fingerprint = X._fingerprint(MODEL, max_tokens, max_model_len) + "|free-text-v1"
    llm = build_llm(EngineConfig(model=MODEL, dtype="bfloat16", max_model_len=max_model_len, max_num_seqs=256))
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sampling = make_sampling(temperature=0, max_tokens=max_tokens)
    max_input = max_model_len - max_tokens - 32
    combined, rows = prev, pending.to_dicts()
    t0 = time.time()
    for s in range(0, len(rows), batch):
        b = rows[s:s + batch]
        prompts, metas = [], []
        for t in b:
            p, m = X.build_prompt(t, tok, max_input); prompts.append(p); metas.append(m)
        raw, trunc = chat_batch(llm, tok, prompts, sampling, max_input_tokens=max_input, return_truncation=True)
        for m, tt in zip(metas, trunc):
            m["hard_prompt_truncated"] = tt
        new = X._score_rows([normalise(r) for r in raw], b, metas, MODEL, fingerprint).with_columns(pl.Series("raw_output", raw))
        combined = pl.concat([combined, new], how="diagonal_relaxed") if combined is not None else new
        combined = combined.unique("trace_id", keep="last")
        tmp = out_path.with_suffix(".tmp.parquet"); combined.write_parquet(tmp); tmp.replace(out_path)
        ok = new.filter(pl.col("validated") & (pl.col("extraction_status") == "ok")).height
        print(f"[free-extract] {min(s + len(b), len(rows))}/{len(rows)}; this batch {ok}/{new.height} validated ok; {time.time() - t0:.0f}s", flush=True)


def _norm(s: str) -> str:
    s = (s or "").replace("**", "")
    return re.sub(r"[\s\$]|\\left|\\right|\\!|\\,", "", s).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def norm_math(x: str) -> str:
    """Formatting-only normalisation of a math answer or reference before matching (D-47 hand check): dfrac/tfrac,
    unbraced frac, a leading "x in" or "x =", dollar signs, thin spaces, thousands commas, degree marks."""
    x = (x or "").strip().strip("$").strip()
    x = x.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\!", "").replace("\\,", "").replace("\\$", "").replace("$", "")
    x = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", x)
    x = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", x)
    x = re.sub(r"^[a-zA-Z]\s*(\\in|=)\s*", "", x)
    x = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", x)
    x = re.sub(r"\^\s*\{?\\circ\}?|°", "", x)
    return x.strip()


def math_correct(fa: str, ref) -> int:
    from src.perf.answer_grading import grade_math_answer, math_answers_equivalent
    if grade_math_answer(fa, ref).get("success") == 1:
        return 1
    a, r = norm_math(fa), norm_math(str(ref or ""))
    if a and r and (a.replace(" ", "") == r.replace(" ", "") or grade_math_answer(a, r).get("success") == 1 or math_answers_equivalent(a, r)):
        return 1
    return 0


def grade_row(t: dict, e: dict | None) -> dict:
    """Grade from gemma's free-text extraction. The extraction is used only if it is grounded: the extracted answer must
    appear in the model's final answer section (after </think>), as text or as an equivalent math expression (repo
    candidate parser + math_verify equivalence); for MCQ the letter must be the choice the repo parser reads from the
    final answer, or stand alone in its last 200 characters. Unfinished generations grade 0 (repo policy)."""
    from src.perf.answer_grading import (extract_choice, extract_math_candidates, grade_math_answer, grade_mcq_answer,
                                         math_answers_equivalent)
    task, ref = t["task_type"], t.get("reference_answer")
    ans_text = str(t.get("answer_text") or "")
    def direct():
        if not ans_text.strip():
            return 0
        if task == "math":
            return int(bool(grade_math_answer(ans_text.replace("**", ""), ref).get("success")))
        return int(bool(grade_mcq_answer(ans_text, ref, t.get("prompt") or "").get("success")))
    res = {"trace_id": t["trace_id"], "direct": direct()}
    if e is None:
        return res | {"success": res["direct"], "grade_source": "no_extraction_row", "extracted": None}
    if e.get("extraction_method") in ("incomplete_generation_policy", "empty_source"):
        return res | {"success": 0, "grade_source": "unfinished_generation", "extracted": None}
    try:
        j = json.loads(normalise(e.get("raw_output")) or "") if e.get("raw_output") else json.loads(e.get("judge_output") or "")
    except Exception:
        j = None
    if not isinstance(j, dict):
        return res | {"success": res["direct"], "grade_source": "fallback_unparsed_output", "extracted": None}
    st, fa = j.get("status"), (j.get("final_answer") or "").strip()
    if st != "ok" or not fa:
        if not ans_text.strip():
            return res | {"success": 0, "grade_source": "no_answer", "extracted": None}
        return res | {"success": res["direct"], "grade_source": f"fallback_extractor_{st}", "extracted": None}
    if task == "math":
        fa_c = fa.strip().strip("$").strip()
        grounded = _norm(fa_c) in _norm(ans_text)
        if not grounded:
            cands = [c[0] if isinstance(c, (list, tuple)) else c for c in extract_math_candidates(ans_text.replace("**", ""))]
            grounded = any(math_answers_equivalent(fa_c, c) for c in cands[-8:])
        if not grounded:
            return res | {"success": res["direct"], "grade_source": "fallback_ungrounded", "extracted": fa}
        return res | {"success": math_correct(fa_c, ref), "grade_source": "extracted", "extracted": fa}
    letter = (extract_choice(fa, t.get("prompt") or "")[0] or fa.strip().strip("().*").upper()[:1])
    read = extract_choice(ans_text, t.get("prompt") or "")[0]
    tail = ans_text[-200:]
    grounded = (read == letter) or bool(re.search(r"(?<![A-Za-z])\(?" + re.escape(letter) + r"\)?(?![A-Za-z])", tail))
    if not grounded:
        return res | {"success": res["direct"], "grade_source": "fallback_ungrounded", "extracted": fa}
    return res | {"success": int(letter == str(ref).strip().upper()), "grade_source": "extracted", "extracted": fa}


def cmd_grade():
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "task_type", "reference_answer", "full_text", "answer_text", "prompt"])
                    for p in sorted(IN.glob("traces_*.parquet"))], how="diagonal_relaxed")
    ex = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("answer_extractions.shard*.parquet"))], how="diagonal_relaxed").unique("trace_id", keep="last")
    idx = {r["trace_id"]: r for r in ex.iter_rows(named=True)}
    g = pl.DataFrame([grade_row(t, idx.get(t["trace_id"])) for t in tr.iter_rows(named=True)])
    g = g.join(tr.select(["trace_id", "task_type"]), on="trace_id")
    g.write_parquet(OUT / "grades.parquet")
    pl.Config.set_tbl_rows(40)
    print(g.group_by(["task_type", "grade_source"]).len().sort(["task_type", "len"], descending=[False, True]))
    print(g.group_by("task_type").agg(pl.col("success").mean().alias("new"), pl.col("direct").mean().alias("direct_grader")))
    print(g.group_by(["task_type", "direct", "success"]).len().sort(["task_type", "direct", "success"]))


def cmd_sample(n: int):
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "task_type", "condition", "reference_answer", "answer_text", "finish_reason"]).with_columns(pl.lit(p.stem).alias("file"))
                    for p in sorted(IN.glob("traces_*.parquet"))], how="diagonal_relaxed")
    g = pl.read_parquet(OUT / "grades.parquet").select(["trace_id", "success", "direct", "grade_source", "extracted"])
    d = tr.join(g, on="trace_id")
    math = d.filter(pl.col("task_type") == "math")
    parts = [math.filter(pl.col("direct") != pl.col("success")).sample(min(40, math.filter(pl.col("direct") != pl.col("success")).height), seed=0),
             math.filter((pl.col("direct") == pl.col("success")) & (pl.col("answer_text").str.len_chars() > 0)).sample(30, seed=0),
             math.filter(pl.col("answer_text").str.len_chars() == 0).sample(10, seed=0),
             d.filter(pl.col("task_type") != "math").sample(20, seed=0)]
    s = pl.concat(parts, how="diagonal_relaxed").head(n)
    s.write_parquet(OUT / "check_sample.parquet"); print("sample", s.height, "->", OUT / "check_sample.parquet")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--shard", type=int, default=0); e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--batch", type=int, default=128); e.add_argument("--max-tokens", type=int, default=768); e.add_argument("--max-model-len", type=int, default=65536)
    sub.add_parser("grade")
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "extract": cmd_extract(a.shard, a.num_shards, a.batch, a.max_tokens, a.max_model_len)
    elif a.cmd == "grade": cmd_grade()
    else: cmd_sample(a.n)


if __name__ == "__main__":
    main()
