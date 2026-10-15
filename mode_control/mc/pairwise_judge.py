"""Pairwise comparative judging of idea quality (D-63).

For every keyword (instance) and sampling seed, each steered arm's final idea is compared with the unsteered idea of the same
run, keyword and seed. Two questions, asked in separate calls, each in both presentation orders: which idea is more original,
which is more feasible. The judge answers A, B or tie under guided JSON at temperature 0. Pairs with an empty answer on either
side are skipped and counted.

Scores. Per (keyword, seed, arm, axis): +1 if the steered idea wins, -1 if the unsteered idea wins, 0 for a tie, averaged
over the two orders; then averaged over seeds; net win per arm and axis = mean over keywords, with a 2000-draw keyword
bootstrap CI. Frontier expansion (the D-63 test) = an arm whose net win is positive with the CI clear of zero on BOTH axes.
Also reported: tie rate, position bias (share of A wins by order), number of pairs.

Usage:
  python -m mode_control.mc.pairwise_judge --runs tx_idea,power1,sched1 --judge google/gemma-4-31B-it
  python -m mode_control.mc.pairwise_judge --runs tx_idea --arms sig_math,sig_idea --judge Qwen/Qwen3.5-27B --smoke 10
Outputs under data/pairwise/: <run>__<judge tag>.parquet (every call), summary__<judge tag>.csv (run x arm x axis),
frontier__<judge tag>.csv. The prompt text is the one ratified with D-63.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import polars as pl

from .common import DATA

PW = DATA / "pairwise"
SYSTEM = ("You compare two research ideas written for the same keyword. Judge only the idea text, not its length or "
          "formatting. Answer with a JSON object and nothing else.")
QUESTIONS = {
    "originality": "Which research idea is more original (novel, surprising, not an obvious extension of existing work)? "
                   "Answer tie only if they are equally original.",
    "feasibility": "Which research idea is more feasible (a competent research group could carry it out with realistic "
                   "resources, methods and data)? Answer tie only if they are equally feasible.",
}
USER = ('Keyword: "{keyword}"\n\nIdea A:\n{a}\n\nIdea B:\n{b}\n\nQuestion: {q}\n'
        'Answer with a JSON object {{"winner": "A" | "B" | "tie"}}.')
SCHEMA = {"type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}}, "required": ["winner"]}
MAX_IDEA_CHARS = 6000


def keyword_of(r: dict) -> str:
    iid = r.get("instance_id") or ""
    if ":" in iid:
        tail = iid.split(":", 1)[1]
        if tail and len(tail) < 80:
            return tail
    m = re.search(r"[Kk]eyword[^A-Za-z0-9]{0,4}([^\n\"]+)", r.get("prompt") or "")
    return m.group(1).strip(" .:*") if m else (r.get("prompt") or "")[:80]


def load_run(run: str, arms: list[str] | None) -> tuple[pl.DataFrame, list[str]]:
    rd = DATA / "runs" / run
    tr = pl.concat([pl.read_parquet(p, columns=["trace_id", "instance_id", "seed", "condition", "prompt", "answer_text", "hard_loop"])
                    for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    conds = sorted(set(tr["condition"].to_list()) - {"baseline"})
    if arms:
        conds = [c for c in conds if c in arms]
    return tr, conds


def build_pairs(tr: pl.DataFrame, conds: list[str], smoke: int) -> tuple[list[dict], dict]:
    base = {(r["instance_id"], r["seed"]): r for r in tr.filter(pl.col("condition") == "baseline").iter_rows(named=True)}
    pairs, skipped = [], {"no_baseline": 0, "empty_answer": 0}
    for c in conds:
        rows = tr.filter(pl.col("condition") == c)
        if smoke:
            rows = rows.head(smoke)
        for r in rows.iter_rows(named=True):
            b = base.get((r["instance_id"], r["seed"]))
            if b is None:
                skipped["no_baseline"] += 1; continue
            ia, ib = (r["answer_text"] or "").strip(), (b["answer_text"] or "").strip()
            if not ia or not ib:
                skipped["empty_answer"] += 1; continue
            pairs.append({"condition": c, "instance_id": r["instance_id"], "seed": int(r["seed"]), "keyword": keyword_of(r),
                          "steered": ia[:MAX_IDEA_CHARS], "unsteered": ib[:MAX_IDEA_CHARS], "trace_id": r["trace_id"], "base_trace_id": b["trace_id"]})
    return pairs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tx_idea,power1,sched1")
    ap.add_argument("--arms", default=None, help="comma list of conditions (default: every non-baseline arm)")
    ap.add_argument("--judge", default="google/gemma-4-31B-it")
    ap.add_argument("--smoke", type=int, default=0, help="first N traces per arm")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    a = ap.parse_args()
    PW.mkdir(parents=True, exist_ok=True)
    tag = a.judge.replace("/", "_") + ("_smoke" if a.smoke else "")
    arms = a.arms.split(",") if a.arms else None

    from src.judge.vllm_engine import EngineConfig, build_llm, chat_batch, make_sampling, safe_json
    from transformers import AutoTokenizer
    mns = a.max_num_seqs or (128 if "Qwen3.5" in a.judge else 256)
    llm = build_llm(EngineConfig(model=a.judge, dtype="bfloat16", max_model_len=8192, max_num_seqs=mns))
    tok = AutoTokenizer.from_pretrained(a.judge, trust_remote_code=True)
    samp = make_sampling(temperature=0, max_tokens=24, json_schema=SCHEMA)

    all_summary, all_frontier = [], []
    for run in a.runs.split(","):
        tr, conds = load_run(run, arms)
        pairs, skipped = build_pairs(tr, conds, a.smoke)
        print(f"[pairwise] {run}: {len(conds)} arms, {len(pairs)} pairs, skipped {skipped}", flush=True)
        if not pairs:
            continue
        prompts, meta = [], []
        for i, p in enumerate(pairs):
            for axis, q in QUESTIONS.items():
                for order in ("steered_first", "unsteered_first"):
                    A, B = (p["steered"], p["unsteered"]) if order == "steered_first" else (p["unsteered"], p["steered"])
                    prompts.append(USER.format(keyword=p["keyword"], a=A, b=B, q=q))
                    meta.append((i, axis, order))
        t0 = time.time()
        texts = chat_batch(llm, tok, prompts, samp, system=SYSTEM, max_input_tokens=7000)
        print(f"[pairwise] {run}: {len(prompts)} judge calls in {time.time() - t0:.0f}s", flush=True)
        recs = []
        for (i, axis, order), t in zip(meta, texts):
            j = safe_json(t) or {}
            w = j.get("winner")
            if w == "A":
                steered_wins = 1 if order == "steered_first" else -1
            elif w == "B":
                steered_wins = -1 if order == "steered_first" else 1
            elif w == "tie":
                steered_wins = 0
            else:
                steered_wins = None
            p = pairs[i]
            recs.append({"run": run, "condition": p["condition"], "instance_id": p["instance_id"], "seed": p["seed"], "keyword": p["keyword"],
                         "axis": axis, "order": order, "winner_raw": w, "score": steered_wins, "trace_id": p["trace_id"], "base_trace_id": p["base_trace_id"],
                         "judge": a.judge})
        df = pl.DataFrame(recs, schema_overrides={"score": pl.Int64, "winner_raw": pl.Utf8}, infer_schema_length=None)
        df.write_parquet(PW / f"{run}__{tag}.parquet")
        # scores: average orders, then seeds, then keywords; bootstrap over keywords
        rng = np.random.default_rng(0)
        per_kw = (df.drop_nulls("score").group_by(["condition", "axis", "instance_id", "seed"]).agg(pl.col("score").mean())
                    .group_by(["condition", "axis", "instance_id"]).agg(pl.col("score").mean()))
        for (cond, axis), g in per_kw.group_by(["condition", "axis"]):
            x = g["score"].to_numpy()
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            sub = df.filter((pl.col("condition") == cond) & (pl.col("axis") == axis))
            all_summary.append({"run": run, "condition": cond, "axis": axis, "n_keywords": int(len(x)), "n_calls": int(sub.height),
                                "net_win": float(x.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                                "tie_rate": float((sub["winner_raw"] == "tie").mean()), "unparsed": int(sub["score"].null_count()),
                                "A_wins_steered_first": float((sub.filter(pl.col("order") == "steered_first")["winner_raw"] == "A").mean()),
                                "A_wins_unsteered_first": float((sub.filter(pl.col("order") == "unsteered_first")["winner_raw"] == "A").mean())})
        S = pl.DataFrame([s for s in all_summary if s["run"] == run])
        for cond in sorted(set(S["condition"].to_list())):
            o = S.filter((pl.col("condition") == cond) & (pl.col("axis") == "originality")).row(0, named=True)
            f = S.filter((pl.col("condition") == cond) & (pl.col("axis") == "feasibility")).row(0, named=True)
            all_frontier.append({"run": run, "condition": cond, "orig_net": o["net_win"], "orig_lo": o["lo"], "orig_hi": o["hi"],
                                 "feas_net": f["net_win"], "feas_lo": f["lo"], "feas_hi": f["hi"],
                                 "frontier_expansion": bool(o["lo"] > 0 and f["lo"] > 0), "both_worse": bool(o["hi"] < 0 and f["hi"] < 0)})
    if all_summary:
        pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220)
        S = pl.DataFrame(all_summary).sort(["run", "condition", "axis"]); S.write_csv(PW / f"summary__{tag}.csv"); print(S)
        F = pl.DataFrame(all_frontier).sort(["run", "condition"]); F.write_csv(PW / f"frontier__{tag}.csv"); print(F)
    (PW / f"config__{tag}.json").write_text(json.dumps({"runs": a.runs, "arms": a.arms, "judge": a.judge, "system": SYSTEM,
                                                        "questions": QUESTIONS, "user_template": USER, "max_idea_chars": MAX_IDEA_CHARS}, indent=1))


if __name__ == "__main__":
    main()
