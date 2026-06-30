"""src/generate/generate_traces.py — vLLM data-parallel trace generation (Blackwell).

For each gen model x task instance x seed: render the prompt with the model's chat
template, generate with shared decoding (don't confound), parse think/answer per model
kind, characterize failure mode post-hoc, write parquet. Replicate-don't-shard: one model
on as few GPUs as fit (8B trivially fits one card), data-parallel across the rest.

Resumable: skips (gen_model, instance_id, seed) already present in the output shard.
Failures are KEPT (exp1 lesson): no retries; first-attempt distribution + post-hoc
failure_mode label. Stage task parquets to local NVMe before iterating.

Usage:
    python -m src.generate.generate_traces --config configs/exp.yaml \
        --gen-model reasoner --tasks math code gpqa planning moral idea
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judge.vllm_engine import EngineConfig, build_llm, make_sampling  # noqa: E402
from src.utils.parse import parse_generation, is_completed, reasoning_text_for_analysis  # noqa: E402
from src.utils.characterize import (soft_loop_detect, loop_collapse_detect,  # noqa: E402
                                    classify_failure_mode)


def build_work(tasks_dir: Path, task_filter, seeds) -> list[dict]:
    work = []
    for path in sorted(tasks_dir.glob("*.parquet")):
        if task_filter and path.stem not in task_filter:
            continue
        df = pl.read_parquet(path)
        for row in df.iter_rows(named=True):
            for seed in seeds:
                work.append({**row, "seed": seed})
    if not work:
        raise FileNotFoundError(f"no task parquets in {tasks_dir} matching {task_filter}")
    return work


def done_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    df = pl.read_parquet(out_path)
    return set(zip(df["gen_model"].to_list(), df["instance_id"].to_list(), df["seed"].to_list()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--gen-model", required=True,
                    help="key under gen_models in the config (e.g. anchor, reasoner, qwen35_9b)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--shard", type=int, default=0, help="this replica's index (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--num-shards", type=int, default=1, help="total replicas (one per GPU)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.gen_model not in cfg.gen_models:
        raise SystemExit(f"--gen-model '{args.gen_model}' not in config gen_models; "
                         f"valid: {list(cfg.gen_models.keys())}")
    gm = cfg.gen_models[args.gen_model]
    gen = cfg.generation
    seeds = list(cfg.sample_seeds)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Each replica writes its OWN shard file -> no cross-process write collisions. Merge after.
    if args.num_shards > 1:
        out_path = out_dir / f"traces_{args.gen_model}.shard{args.shard:02d}of{args.num_shards:02d}.parquet"
    else:
        out_path = out_dir / f"traces_{args.gen_model}.parquet"

    work = build_work(Path(args.tasks_dir), args.tasks, seeds)
    # Deterministic round-robin shard: replica s handles work units where idx % num_shards == s.
    if args.num_shards > 1:
        work = [w for i, w in enumerate(work) if i % args.num_shards == args.shard]
    done = done_keys(out_path)
    pending = [w for w in work if (args.gen_model, w["instance_id"], w["seed"]) not in done]
    print(f"[gen:{args.gen_model}] shard {args.shard}/{args.num_shards}: "
          f"{len(work)} units, {len(done)} done, {len(pending)} pending")
    if not pending:
        return 0

    # Build ONE replica on this task's single visible GPU (SLURM/CUDA_VISIBLE_DEVICES pins it).
    def _cfg_get(obj, key, default):
        # OmegaConf-safe: .get() if available, else getattr, else default (never raises on missing)
        try:
            if hasattr(obj, "get"):
                v = obj.get(key, default)
                return v if v is not None else default
        except Exception:
            pass
        return getattr(obj, key, default)
    gmu = float(_cfg_get(gm, "gpu_memory_utilization",
                         _cfg_get(gen, "gpu_memory_utilization", 0.90)))
    mnbt = _cfg_get(gm, "max_num_batched_tokens",
                    _cfg_get(gen, "max_num_batched_tokens", None))
    # max_num_seqs: hybrid-Mamba/GDN models (Qwen3.5, esp. 27B) cap decode concurrency at the
    # number of Mamba cache blocks; vLLM's default 1024 can exceed it -> CUDA-graph capture fails.
    # Allow a per-model override (gm.max_num_seqs) first, then a global default (gen.max_num_seqs).
    mns = _cfg_get(gm, "max_num_seqs", _cfg_get(gen, "max_num_seqs", None))
    eng = EngineConfig(model=gm.hf_id, dtype=gen.dtype,
                       tensor_parallel_size=1,
                       max_model_len=int(gen.max_model_len),
                       gpu_memory_utilization=gmu,
                       max_num_batched_tokens=(int(mnbt) if mnbt is not None else None),
                       max_num_seqs=(int(mns) if mns is not None else None))
    llm = build_llm(eng)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(gm.hf_id, trust_remote_code=True)

    max_new = (gen.decode.max_new_tokens_reasoning if gm.kind == "reasoning"
               else gen.decode.max_new_tokens_anchor)
    # per-model thinking spec: delimiters, enable mechanism, recommended sampling.
    # config sets gen_models.<name>.thinking_style (r1|gemma4|qwen35|none); default by kind.
    from src.generate.thinking_spec import get_spec, build_messages, template_kwargs, normalize_to_canonical
    style = getattr(gm, "thinking_style", None) or ("r1" if gm.kind == "reasoning" else "none")
    spec = get_spec(style)
    # sampling: prefer the model card's recommended params (spec) but allow config to override
    samp_kwargs = dict(temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k,
                       max_tokens=int(max_new), seed=cfg.seed)
    samp_kwargs.update(spec.extra_sampling)            # e.g. qwen presence_penalty=1.5
    if getattr(gen.decode, "override_spec_sampling", False):
        samp_kwargs.update(temperature=gen.decode.temperature, top_p=gen.decode.top_p, top_k=gen.decode.top_k)
    sampling = make_sampling(**samp_kwargs)
    print(f"[gen:{args.gen_model}] thinking_style={style} enable_thinking={spec.enable_thinking} "
          f"sampling=temp{samp_kwargs['temperature']}/top_k{samp_kwargs['top_k']}"
          f"{'/pp'+str(spec.extra_sampling['presence_penalty']) if 'presence_penalty' in spec.extra_sampling else ''}")

    # Render prompts (vLLM preserves order). build_messages injects the thinking-enable system prefix
    # where required (gemma4); apply_chat_template gets the spec's template kwargs (qwen enable_thinking).
    tkw = template_kwargs(spec)
    rendered = []
    for w in pending:
        msgs = build_messages(w["prompt"], spec)
        try:
            r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw)
        except TypeError:
            r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # tok ignores unknown kwargs
        rendered.append(r)
    t0 = time.time()
    outs = llm.generate(rendered, sampling)
    print(f"[gen:{args.gen_model}] generated {len(outs)} in {time.time()-t0:.1f}s")

    from src.generate.repair_byte_bpe import repair as _repair_bpe
    rows = []
    for w, o in zip(pending, outs):
        raw_text = _repair_bpe(o.outputs[0].text)  # some tokenizer/vLLM combos return pre-byte-decode text
        # normalize model-specific think delimiters (gemma4 channel tags) -> canonical <think></think>
        gen_text = normalize_to_canonical(raw_text, spec)
        finish = "stop" if o.outputs[0].finish_reason == "stop" else "length"
        think_text, answer_text = parse_generation(gen_text, gm.kind)
        has_close = ("</think>" in gen_text) if gm.kind == "reasoning" else True
        completed = is_completed(kind=gm.kind, finish_reason=finish,
                                 has_close_tag=has_close, has_answer=bool(answer_text))
        rtext = reasoning_text_for_analysis(analysis_source=gm.analysis_source,
                                            think_text=think_text, answer_text=answer_text)
        failure_mode = classify_failure_mode(text=gen_text, completed=completed, finish_reason=finish)
        rows.append({
            "trace_id": str(uuid.uuid4()), "gen_model": args.gen_model, "gen_model_id": gm.hf_id,
            "instance_id": w["instance_id"], "task_type": w["task_type"], "seed": w["seed"],
            "difficulty_raw": w.get("difficulty_raw"), "reference_answer": w.get("reference_answer"),
            "instance_metadata": w.get("metadata", "{}"), "prompt": w["prompt"],
            "full_text": gen_text, "think_text": think_text, "answer_text": answer_text,
            "reasoning_text_for_analysis": rtext,
            "n_new_tokens": len(o.outputs[0].token_ids), "completed": completed,
            "finish_reason": finish, "failure_mode": failure_mode,
            "hard_loop": loop_collapse_detect(gen_text), "soft_loop": soft_loop_detect(gen_text),
            "decode_temperature": float(gen.decode.temperature),
        })
    new = pl.DataFrame(rows)
    combined = pl.concat([pl.read_parquet(out_path), new], how="diagonal_relaxed") if out_path.exists() else new
    tmp = out_path.with_suffix(".parquet.tmp"); combined.write_parquet(tmp); tmp.replace(out_path)
    print(f"[gen:{args.gen_model}] wrote {len(rows)} -> {out_path} (total {combined.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
