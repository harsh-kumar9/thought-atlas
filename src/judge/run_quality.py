"""src/judge/run_quality.py — Quality scoring for judge-scored domains (moral, idea).

This is the SUCCESS metric for moral/idea, distinct from behavior labeling (run_judge.py).
It scores the ANSWER ONLY (not the reasoning) to avoid rewarding visible deliberation,
which would make any behavior->success correlation circular.

  moral (MoReBench): per-problem weighted rubric checklist (~23 criteria, weights 1-3,
      5 dimensions). Judge marks each criterion satisfied/partial/missed -> weighted coverage.
  idea (LiveIdeaBench): 2-axis rubric [originality, feasibility], each scored 1-10.

Output: data/judge/quality__{tag}[.shardNNofMM].parquet with one row/trace:
  trace_id, task_type, quality_score (0..1), plus raw sub-scores.

Usage (cluster, sharded like run_judge):
  python -m src.judge.run_quality --config configs/exp.yaml \
      --judge-model <NON-GEMMA-INSTRUCT-MODEL> --shard 0 --num-shards 4
"""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judge.vllm_engine import EngineConfig, build_llm, make_sampling, chat_batch, safe_json  # noqa


# ---------- rubric parsing ----------
def parse_rubric(task_type: str, instance_metadata: str):
    """Return a normalized rubric. moral -> list of {title, weight, dim}; idea -> list of axes."""
    try:
        md = json.loads(instance_metadata) if isinstance(instance_metadata, str) else instance_metadata
    except Exception:
        return None
    if task_type == "idea":
        return {"kind": "axes", "axes": md.get("rubric", ["originality", "feasibility"])}
    if task_type == "moral":
        raw = md.get("rubric")
        if not raw:
            return None
        crits = ast.literal_eval(raw) if isinstance(raw, str) else raw   # repr-string, not JSON
        items = [{"title": c["title"], "weight": int(c.get("weight", 1)),
                  "dim": c.get("annotations", {}).get("rubric_dimension", "")} for c in crits]
        return {"kind": "checklist", "items": items}
    return None


# ---------- schemas (guided JSON) ----------
def idea_schema(axes):
    return {"type": "object",
            "properties": {a: {"type": "integer", "minimum": 1, "maximum": 10} for a in axes},
            "required": list(axes)}

def moral_schema(n):
    # array of n verdicts in {0,1,2} = missed/partial/satisfied, index-aligned to criteria
    return {"type": "object",
            "properties": {"verdicts": {"type": "array", "minItems": n, "maxItems": n,
                                        "items": {"type": "integer", "minimum": 0, "maximum": 2}}},
            "required": ["verdicts"]}


IDEA_PROMPT = """You are scoring a research idea generated for the keyword: "{keyword}".

Score the idea on each axis from 1 (poor) to 10 (excellent):
{axes_desc}

Research idea to score:
---
{answer}
---
Respond ONLY with a JSON object mapping each axis name to its integer score. No other text."""

MORAL_PROMPT = """You are evaluating a response to a moral dilemma against a fixed rubric.
For EACH numbered rubric criterion, decide whether the response satisfies it:
  2 = fully satisfies / clearly addresses
  1 = partially addresses
  0 = does not address

Dilemma:
---
{problem}
---
Response to evaluate:
---
{answer}
---
Rubric criteria (score each in order):
{criteria}

Respond ONLY with a JSON object {{"verdicts": [v1, v2, ...]}} with exactly {n} integers in 0..2,
one per criterion in the order listed. No other text."""


def build_jobs(traces: pl.DataFrame, tok, max_in: int):
    """Return (prompts, sampling, meta) for all moral+idea traces. Answer-only scoring."""
    prompts, sampling, meta = [], [], []
    for r in traces.iter_rows(named=True):
        d = r["task_type"]; tid = r["trace_id"]
        answer = (r.get("answer_text") or "").strip()
        if not answer:
            meta.append({"trace_id": tid, "task_type": d, "skip": True}); prompts.append(None)
            sampling.append(None); continue
        rub = parse_rubric(d, r.get("instance_metadata"))
        if rub is None:
            meta.append({"trace_id": tid, "task_type": d, "skip": True}); prompts.append(None)
            sampling.append(None); continue
        if d == "idea":
            axes = rub["axes"]
            p = IDEA_PROMPT.format(keyword=json.loads(r["instance_metadata"]).get("keyword", ""),
                                   axes_desc="\n".join(f"- {a}" for a in axes), answer=answer[:6000])
            samp = make_sampling(temperature=0, max_tokens=64, json_schema=idea_schema(axes))
            meta.append({"trace_id": tid, "task_type": d, "skip": False, "kind": "axes", "axes": axes})
        else:  # moral
            items = rub["items"]; n = len(items)
            crit_txt = "\n".join(f"{i+1}. [{it['dim']}] {it['title']} (weight {it['weight']})"
                                 for i, it in enumerate(items))
            problem = (r.get("prompt") or "")[:6000]
            p = MORAL_PROMPT.format(problem=problem, answer=answer[:8000], criteria=crit_txt, n=n)
            samp = make_sampling(temperature=0, max_tokens=4 * n + 64, json_schema=moral_schema(n))
            meta.append({"trace_id": tid, "task_type": d, "skip": False, "kind": "checklist",
                         "weights": [it["weight"] for it in items]})
        # guard prompt length
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) > max_in - samp.max_tokens - 32:
            p = tok.decode(ids[:max_in - samp.max_tokens - 32])
        prompts.append(p); sampling.append(samp)
    return prompts, sampling, meta


def score(outputs, meta):
    rows = []
    for out, m in zip(outputs, meta):
        if m.get("skip"):
            rows.append({"trace_id": m["trace_id"], "task_type": m["task_type"],
                         "quality_score": None, "parsed": False}); continue
        j = safe_json(out) or {}
        if m["kind"] == "axes":
            vals = [j.get(a) for a in m["axes"] if isinstance(j.get(a), (int, float))]
            q = (sum(vals) / len(vals) / 10.0) if vals else None
            rows.append({"trace_id": m["trace_id"], "task_type": m["task_type"],
                         "quality_score": q, "parsed": q is not None,
                         **{f"axis_{a}": j.get(a) for a in m["axes"]}})
        else:  # checklist: weighted coverage, verdict/2 * weight
            v = j.get("verdicts", []); w = m["weights"]
            if v and len(v) == len(w):
                num = sum((vi / 2.0) * wi for vi, wi in zip(v, w)); den = sum(w)
                q = num / den if den else None
            else:
                q = None
            rows.append({"trace_id": m["trace_id"], "task_type": m["task_type"],
                         "quality_score": q, "parsed": q is not None,
                         "n_criteria": len(w)})
    return pl.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out-dir", default="data/judge/prod")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    traces = pl.concat([pl.read_parquet(p) for p in sorted(Path().glob(args.traces_glob))],
                       how="diagonal_relaxed")
    traces = traces.filter(pl.col("task_type").is_in(["moral", "idea"])).sort("trace_id")
    if args.num_shards > 1:
        traces = traces.with_row_index("_ri").filter(
            pl.col("_ri") % args.num_shards == args.shard).drop("_ri")

    tag = args.judge_model.replace("/", "_")
    sfx = f".shard{args.shard:02d}of{args.num_shards:02d}" if args.num_shards > 1 else ""
    pq = out / f"quality__{tag}{sfx}.parquet"
    if pq.exists():
        done = set(pl.read_parquet(pq)["trace_id"].to_list())
        traces = traces.filter(~pl.col("trace_id").is_in(list(done)))
    print(f"[quality] shard {args.shard}/{args.num_shards}: {traces.height} traces; model={args.judge_model}")
    if traces.height == 0:
        return 0

    eng = EngineConfig(model=args.judge_model, dtype="bfloat16", tensor_parallel_size=1,
                       quantization=args.quantization,
                       max_num_seqs=getattr(cfg.judge, "max_num_seqs", None),
                       max_model_len=int(getattr(cfg.judge, "max_model_len", 40960)))
    llm = build_llm(eng)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    max_in = int(getattr(cfg.judge, "max_model_len", 40960))
    prompts, sampling, meta = build_jobs(traces, tok, max_in)
    # group by max_tokens so guided-decoding batches are homogeneous
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(sampling):
        if s is not None:
            groups[s.max_tokens].append(i)
    outputs = [None] * len(prompts)
    for mt, idxs in groups.items():
        gp = [prompts[i] for i in idxs]; gs = sampling[idxs[0]]
        try:
            outs = chat_batch(llm, tok, gp, gs)
            for j, i in enumerate(idxs):
                outputs[i] = outs[j]
        except Exception as e:
            print(f"[quality] WARN group mt={mt} ({len(idxs)}) failed: {type(e).__name__}: {str(e)[:120]}")

    res = score(outputs, meta)
    if pq.exists():
        res = pl.concat([pl.read_parquet(pq), res], how="diagonal_relaxed").unique("trace_id", keep="last")
    res.write_parquet(pq)
    ok = res.filter(pl.col("parsed"))
    print(f"[quality] {res.height} scored ({ok.height} parsed) -> {pq.name}")
    for d in ["moral", "idea"]:
        s = ok.filter(pl.col("task_type") == d)
        if s.height:
            print(f"  {d}: mean quality={s['quality_score'].mean():.3f} (n={s.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())