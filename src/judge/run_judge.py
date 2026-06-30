"""src/judge/run_judge.py — Two-track self-hosted judging on vLLM.

Track A (aggregate, non-temporal): verbatim Kim + Gandhi prompts on the whole
reasoning_text -> 8 counts/trace. Comparable to Kim/Gandhi prior work.

Track B (temporal): per-sentence multi-label presence via the ThinkARM indexed-batch
mechanic (batch=20 + rolling previous-context). Two taxonomy passes (Kim, Gandhi) x two
context modes (full-context pass-1, isolated pass-2). pass1-pass2 = context-dependence.

All judge calls: temperature 0, guided-JSON (no parse failures), single self-hosted model.
Replicate-don't-shard: the model fits one card, so the SLURM wrapper launches one
single-GPU process per GPU and shards the dataset across those processes.

Usage:
    python -m src.judge.run_judge --config configs/exp.yaml --judge-model google/gemma-4-31B-it \
        --track A            # or B, or both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judge.vllm_engine import EngineConfig, build_llm, make_sampling, chat_batch, safe_json  # noqa
from src.judge import schemas as S  # noqa
from src.segment.segmenter import segment_text  # noqa


def _load_prompt(p):
    return Path(p).read_text()


def _truncate_to_tokens(tok, text: str, max_tokens: int) -> str:
    """Keep a trace within the judge's budget. Whole-trace counting tolerates middle-truncation;
    we keep head+tail (where opening framing and conclusions/verification live) over the middle."""
    if not text:
        return ""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = ids[: max_tokens // 2]
    tail = ids[-(max_tokens - len(head)):]
    return (tok.decode(head) + "\n...[trace truncated for judge context]...\n" + tok.decode(tail))


def run_track_A(llm, tok, traces: pl.DataFrame, cfg) -> pl.DataFrame:
    kim_t = _load_prompt(cfg.judge.prompts.kim_whole)
    gan_t = _load_prompt(cfg.judge.prompts.gandhi_whole)
    # Budget: judge ctx minus prompt template (~1k) minus output (256) minus margin.
    budget = int(getattr(cfg.judge, "max_model_len", 32768)) - 2048
    texts = [_truncate_to_tokens(tok, t or "", budget)
             for t in traces["reasoning_text_for_analysis"].to_list()]
    ids = traces["trace_id"].to_list()

    kim_prompts = [kim_t.format(chain_of_thought=t) for t in texts]
    gan_prompts = [gan_t.format(chain_of_thought=t) for t in texts]
    samp_kim = make_sampling(temperature=0, max_tokens=256, json_schema=S.KIM_WHOLE_SCHEMA)
    samp_gan = make_sampling(temperature=0, max_tokens=256, json_schema=S.GANDHI_WHOLE_SCHEMA)

    kim_out = chat_batch(llm, tok, kim_prompts, samp_kim)
    gan_out = chat_batch(llm, tok, gan_prompts, samp_gan)

    rows = []
    for tid, ko, go in zip(ids, kim_out, gan_out):
        kj = safe_json(ko) or {}
        gj = safe_json(go) or {}
        row = {"trace_id": tid}
        for b in S.KIM_BEHAVIORS:
            row[b] = int(kj.get(b, 0) or 0)
        for b in S.GANDHI_BEHAVIORS:
            row[b] = int(gj.get(b, 0) or 0)
        rows.append(row)
    return pl.DataFrame(rows)


def _batched(seq, k):
    for i in range(0, len(seq), k):
        yield i, seq[i:i + k]


def run_track_B(llm, tok, traces: pl.DataFrame, cfg, *, isolated: bool) -> pl.DataFrame:
    """Per-sentence labels. isolated=False -> full-context (rolling prior); True -> sentence-only.

    BATCHED: builds every (trace, sentence-batch, taxonomy) prompt up front and sends them through
    vLLM in large batches, instead of one chat_batch call per unit. vLLM batches internally, so this
    is ~10-50x faster than the per-call loop (which left the GPU idle between 15s requests).
    """
    kim_t = _load_prompt(cfg.judge.prompts.kim_per_sentence)
    gan_t = _load_prompt(cfg.judge.prompts.gandhi_per_sentence)
    bs = int(cfg.judge.batch_size_sentences)

    # ---- Phase 1: segment every trace, build the full prompt list with routing metadata ----
    seg_cache: dict = {}            # tid -> list of segment dicts
    sent_cache: dict = {}           # tid -> list of sentence strings
    jobs = []                       # (tid, start, key, schema) parallel to prompts
    prompts = []
    sampling = []
    # Budget for the rolling prior context so the full prompt fits the judge window.
    # prompt = problem(<=4000 chars) + previous_context + batch + template boilerplate + JSON room.
    # Reserve headroom; keep the RECENT tail of prior context (most relevant to current sentences).
    judge_window = int(getattr(cfg.judge, "max_model_len", 65536))
    # leave ~6k tokens for problem+batch+template+output; budget the rest for prior context (chars~tok*3.5)
    prev_char_budget = max(4000, (judge_window - 6000) * 3)
    for r in traces.iter_rows(named=True):
        tid = r["trace_id"]
        problem = (r.get("prompt") or "")[:4000]
        segs = segment_text(r.get("full_text") or r.get("reasoning_text_for_analysis") or "")
        if not segs:
            continue
        sentences = [s["sentence"] for s in segs]
        seg_cache[tid] = segs
        sent_cache[tid] = sentences
        for start, batch in _batched(sentences, bs):
            if isolated or start == 0:
                prev = ""
            else:
                prev = "\n".join(sentences[:start])
                if len(prev) > prev_char_budget:           # keep the most-recent tail within budget
                    prev = "...[earlier context truncated]...\n" + prev[-prev_char_budget:]
            idx_input = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(batch))
            for key, tmpl, schema in (("kim", kim_t, S.KIM_SENT_SCHEMA),
                                      ("gandhi", gan_t, S.GANDHI_SENT_SCHEMA)):
                prompts.append(tmpl.format(
                    problem=problem,
                    previous_context=(prev or "There are no previous sentences."),
                    indexed_input=idx_input, n_sentences=len(batch)))
                sampling.append(make_sampling(temperature=0, max_tokens=64 * len(batch) + 128,
                                              json_schema=schema))
                jobs.append((tid, start, key))

    if not prompts:
        return pl.DataFrame()

    # ---- Phase 2: one big batched generate. All prompts share temp=0; schema differs per prompt,
    # so group by (schema-key, max_tokens) to keep guided decoding correct, then run each group. ----
    print(f"[trackB] {len(prompts)} judge calls across {len(sent_cache)} traces "
          f"({'isolated' if isolated else 'full-context'}) — batching")
    outputs = [None] * len(prompts)
    # group indices by (key, max_tokens) so each chat_batch call is homogeneous in sampling params
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (tid, start, key) in enumerate(jobs):
        groups[(key, sampling[i].max_tokens)].append(i)
    for (key, mt), idxs in groups.items():
        gp = [prompts[i] for i in idxs]
        gs = sampling[idxs[0]]            # identical within group
        # hard input clamp: judge window minus this group's output reservation, minus a small margin
        clamp = judge_window - int(gs.max_tokens) - 256
        outs = chat_batch(llm, tok, gp, gs, max_input_tokens=clamp)
        for j, i in enumerate(idxs):
            outputs[i] = outs[j]

    # ---- Phase 3: scatter parsed labels back to per-sentence slots ----
    labels = {tid: {i: {"kim": [], "gandhi": []} for i in range(len(s))}
              for tid, s in sent_cache.items()}
    for i, (tid, start, key) in enumerate(jobs):
        parsed = safe_json(outputs[i]) or {"sentences": []}
        for item in parsed.get("sentences", []):
            try:
                local = int(item["index"]) - 1
            except Exception:
                continue
            gi = start + local
            if 0 <= gi < len(sent_cache[tid]):
                labels[tid][gi][key] = list(item.get("behaviors", []))

    # ---- Phase 4: emit per-sentence rows (unchanged schema) ----
    out_rows = []
    for tid, segs in seg_cache.items():
        n = len(segs)
        for i, s in enumerate(segs):
            row = {"trace_id": tid, "seg_idx": i, "n_segments": n,
                   "norm_pos": (i / (n - 1)) if n > 1 else 0.0,
                   "section_type": s["section_type"], "context_mode": "isolated" if isolated else "full"}
            for b in S.KIM_BEHAVIORS:
                row[b] = int(b in labels[tid][i]["kim"])
            for b in S.GANDHI_BEHAVIORS:
                row[b] = int(b in labels[tid][i]["gandhi"])
            out_rows.append(row)
    return pl.DataFrame(out_rows) if out_rows else pl.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--out-dir", default="data/judge")
    ap.add_argument("--quantization", default=None)  # e.g. fp8; validate vs bf16 first
    ap.add_argument("--shard", type=int, default=0, help="this replica's index")
    ap.add_argument("--num-shards", type=int, default=1, help="total replicas (one per GPU)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    traces = pl.concat([pl.read_parquet(p) for p in sorted(Path().glob(args.traces_glob))],
                       how="diagonal_relaxed")
    # only judge non-empty reasoning text
    traces = traces.filter(pl.col("reasoning_text_for_analysis").is_not_null()
                           & (pl.col("reasoning_text_for_analysis").str.len_chars() > 20))
    # deterministic shard by row order so replicas cover disjoint, exhaustive trace sets
    traces = traces.sort("trace_id")
    if args.num_shards > 1:
        traces = traces.with_row_index("_ri").filter(
            pl.col("_ri") % args.num_shards == args.shard).drop("_ri")

    tag = args.judge_model.replace("/", "_")
    sfx = f".shard{args.shard:02d}of{args.num_shards:02d}" if args.num_shards > 1 else ""

    # resume: drop traces already judged in this shard's output file(s)
    def _pending(df, path):
        if not path.exists():
            return df
        done = set(pl.read_parquet(path)["trace_id"].to_list())
        return df.filter(~pl.col("trace_id").is_in(list(done)))

    print(f"[judge] shard {args.shard}/{args.num_shards}: {traces.height} traces; model={args.judge_model}")
    if traces.height == 0:
        return 0

    eng = EngineConfig(model=args.judge_model, dtype="bfloat16",
                       tensor_parallel_size=1,
                       quantization=args.quantization,
                       max_num_seqs=getattr(cfg.judge, "max_num_seqs", None),
                       max_model_len=int(getattr(cfg.judge, "max_model_len", 40960)))
    llm = build_llm(eng)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    if args.track in ("A", "both"):
        pa = out / f"trackA_counts__{tag}{sfx}.parquet"
        todo = _pending(traces, pa)
        print(f"[judge] Track A: {todo.height} pending (of {traces.height})")
        if todo.height:
            a = run_track_A(llm, tok, todo, cfg)
            if pa.exists():
                a = pl.concat([pl.read_parquet(pa), a], how="diagonal_relaxed").unique("trace_id", keep="last")
            a.write_parquet(pa)
            print(f"[judge] Track A -> {a.height} traces -> {pa.name}")
    if args.track in ("B", "both"):
        pb = out / f"trackB_full__{tag}{sfx}.parquet"
        todo = _pending(traces, pb)
        print(f"[judge] Track B: {todo.height} pending (of {traces.height})")
        # Chunk over traces with incremental writes. Rationale: (1) holding all prompts for all
        # 64k-token traces in RAM at once OOM-kills the process; chunking caps peak memory.
        # (2) writing after each chunk means a crash leaves a resumable checkpoint, so _pending
        # skips done traces on restart instead of re-rendering all ~72k prompts from zero.
        TRACE_CHUNK = int(getattr(cfg.judge, "trace_chunk", 100))
        if todo.height:
            n_chunks = (todo.height + TRACE_CHUNK - 1) // TRACE_CHUNK
            for ci in range(n_chunks):
                chunk = todo.slice(ci * TRACE_CHUNK, TRACE_CHUNK)
                if chunk.height == 0:
                    continue
                bf = run_track_B(llm, tok, chunk, cfg, isolated=False)
                if bf.height == 0:
                    continue
                if pb.exists():
                    bf = pl.concat([pl.read_parquet(pb), bf], how="diagonal_relaxed").unique(
                        ["trace_id", "seg_idx"], keep="last")
                bf.write_parquet(pb)
                print(f"[judge] Track B chunk {ci+1}/{n_chunks}: "
                      f"+{chunk.height} traces -> {pb.name} ({bf.height} total labels)", flush=True)
            print(f"[judge] Track B complete -> {pb.name}")
        if cfg.judge.passes.per_sentence_isolated:
            pbi = out / f"trackB_isolated__{tag}{sfx}.parquet"
            todo_i = _pending(traces, pbi)
            if todo_i.height:
                n_chunks = (todo_i.height + TRACE_CHUNK - 1) // TRACE_CHUNK
                for ci in range(n_chunks):
                    chunk = todo_i.slice(ci * TRACE_CHUNK, TRACE_CHUNK)
                    if chunk.height == 0:
                        continue
                    bi = run_track_B(llm, tok, chunk, cfg, isolated=True)
                    if bi.height == 0:
                        continue
                    if pbi.exists():
                        bi = pl.concat([pl.read_parquet(pbi), bi], how="diagonal_relaxed").unique(
                            ["trace_id", "seg_idx"], keep="last")
                    bi.write_parquet(pbi)
                    print(f"[judge] Track B isolated chunk {ci+1}/{n_chunks}: "
                          f"+{chunk.height} traces -> {pbi.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
