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
        --gen-model reasoner --tasks math code gpqa planning security safety moral idea

    reasoner = DeepSeek-R1-Distill-Llama-8B; anchor = Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judge.vllm_engine import EngineConfig, build_llm, make_sampling  # noqa: E402
from src.utils.parse import parse_generation_detailed, is_completed, reasoning_text_for_analysis  # noqa: E402
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


GENERATION_VERSION = "generation-v3-special-tokens"


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_fingerprint(row: dict) -> str:
    return _json_hash({k: row.get(k) for k in (
        "instance_id", "task_type", "prompt", "reference_answer", "metadata")})


def _trace_id(model: str, instance_id: str, seed: int, prompt: str, output: str) -> str:
    material = "\0".join([model, instance_id, str(seed),
                           hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                           hashlib.sha256(output.encode("utf-8")).hexdigest()])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material))


def done_keys(out_path: Path, generation_fingerprint: str) -> set:
    if not out_path.exists():
        return set()
    df = pl.read_parquet(out_path)
    if "generation_fingerprint" not in df.columns:
        raise RuntimeError(
            f"{out_path} is a legacy trace file without a generation fingerprint; "
            "write v2 traces to a fresh output directory instead of mixing runs")
    fingerprints = set(df["generation_fingerprint"].drop_nulls().to_list())
    if fingerprints != {generation_fingerprint}:
        raise RuntimeError(
            f"{out_path} belongs to generation fingerprint(s) {sorted(fingerprints)}; "
            f"current run is {generation_fingerprint}. Use a fresh output directory.")
    return set(zip(df["gen_model"].to_list(), df["instance_id"].to_list(), df["seed"].to_list()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--gen-model", required=True,
                    help="key under gen_models in the config, e.g. anchor=Llama-3.1-8B-Instruct, "
                         "reasoner=DeepSeek-R1-Distill-Llama-8B, qwen35_9b")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--shard", type=int, default=0, help="this replica's index (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--num-shards", type=int, default=1, help="total replicas (one per GPU)")
    ap.add_argument("--work-batch-size", type=int, default=None,
                    help="checkpoint after this many requests (default: config generation.work_batch_size)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.gen_model not in cfg.gen_models:
        raise SystemExit(f"--gen-model '{args.gen_model}' not in config gen_models; "
                         f"valid: {list(cfg.gen_models.keys())}")
    gm = cfg.gen_models[args.gen_model]
    gen = cfg.generation
    seeds = list(cfg.sample_seeds)
    # Fingerprint the full, unsharded task set.  Resume safety must notice task
    # prompt/reference changes even when model and decode settings are unchanged.
    work = build_work(Path(args.tasks_dir), args.tasks, seeds)
    task_set_fingerprint = _json_hash(sorted({
        _task_fingerprint(w) for w in work
    }))

    from src.generate.thinking_spec import get_spec, build_messages, template_kwargs, normalize_to_canonical
    style = getattr(gm, "thinking_style", None) or ("r1" if gm.kind == "reasoning" else "none")
    spec = get_spec(style)
    policy = str(getattr(gen.decode, "sampling_policy", "shared"))
    if policy == "shared":
        base_sampling = {
            "temperature": float(gen.decode.temperature),
            "top_p": float(gen.decode.top_p),
            "top_k": int(gen.decode.top_k),
        }
    elif policy == "model_recommended":
        base_sampling = {"temperature": spec.temperature, "top_p": spec.top_p,
                         "top_k": spec.top_k, **spec.extra_sampling}
    else:
        raise ValueError("generation.decode.sampling_policy must be shared or model_recommended")
    max_new = int(gen.decode.max_new_tokens_reasoning if gm.kind == "reasoning"
                  else gen.decode.max_new_tokens_anchor)
    generation_fingerprint = _json_hash({
        "version": GENERATION_VERSION,
        "model_key": args.gen_model,
        "model_id": str(gm.hf_id),
        "kind": str(gm.kind),
        "analysis_source": str(gm.analysis_source),
        "thinking_style": style,
        "sampling_policy": policy,
        "sampling": base_sampling,
        "skip_special_tokens": False,
        "max_new_tokens": max_new,
        "max_model_len": int(gen.max_model_len),
        "dtype": str(gen.dtype),
        "sample_seeds": seeds,
        "task_set_fingerprint": task_set_fingerprint,
    })

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Each replica writes its OWN shard file -> no cross-process write collisions. Merge after.
    if args.num_shards > 1:
        out_path = out_dir / f"traces_{args.gen_model}.shard{args.shard:02d}of{args.num_shards:02d}.parquet"
    else:
        out_path = out_dir / f"traces_{args.gen_model}.parquet"

    # Deterministic round-robin shard: replica s handles work units where idx % num_shards == s.
    if args.num_shards > 1:
        work = [w for i, w in enumerate(work) if i % args.num_shards == args.shard]
    done = done_keys(out_path, generation_fingerprint)
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

    print(f"[gen:{args.gen_model}] thinking_style={style} enable_thinking={spec.enable_thinking} "
          f"sampling_policy={policy} sampling={base_sampling} fingerprint={generation_fingerprint[:12]}")

    # Render prompts (vLLM preserves order). build_messages injects the thinking-enable system prefix
    # where required (gemma4); apply_chat_template gets the spec's template kwargs (qwen enable_thinking).
    tkw = template_kwargs(spec)
    from src.generate.repair_byte_bpe import repair as _repair_bpe
    combined = pl.read_parquet(out_path) if out_path.exists() else None
    batch_size = int(args.work_batch_size or getattr(gen, "work_batch_size", 64))
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        rendered, sampling = [], []
        for w in batch:
            msgs = build_messages(w["prompt"], spec)
            try:
                rendered_prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **tkw)
            except TypeError:
                rendered_prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            input_tokens = len(tok(rendered_prompt, add_special_tokens=False)["input_ids"])
            output_budget = min(max_new, int(gen.max_model_len) - input_tokens)
            if output_budget <= 0:
                raise ValueError(f"prompt exceeds model context for {w['instance_id']}: {input_tokens}")
            rendered.append(rendered_prompt)
            # Think delimiters are part of the data contract. vLLM otherwise
            # strips tokenizer-registered special tokens before we can separate
            # reasoning from the final answer.
            sampling.append(make_sampling(**base_sampling, max_tokens=output_budget,
                                          seed=int(w["seed"]), skip_special_tokens=False))

        t0 = time.time()
        outs = llm.generate(rendered, sampling)
        rows = []
        for w, o, sp in zip(batch, outs, sampling):
            raw_text = _repair_bpe(o.outputs[0].text)
            gen_text = normalize_to_canonical(raw_text, spec)
            finish = str(o.outputs[0].finish_reason or "unknown")
            parsed = parse_generation_detailed(gen_text, gm.kind, finish_reason=finish)
            think_text, answer_text = parsed["think_text"], parsed["answer_text"]
            has_close = parsed["close_tag_count"] > 0 if gm.kind == "reasoning" else True
            completed = is_completed(kind=gm.kind, finish_reason=finish,
                                     has_close_tag=has_close or parsed["parse_status"] == "direct_answer_no_close",
                                     has_answer=bool(answer_text))
            rtext = reasoning_text_for_analysis(analysis_source=gm.analysis_source,
                                                think_text=think_text, answer_text=answer_text)
            if (gm.kind == "reasoning" and
                    parsed["parse_status"] not in {
                        "single_close", "multiple_close_last_suffix"}):
                # Answer extraction may still recover these rows, but temporal
                # reasoning analyses require an observed, clean boundary.
                rtext = None
            failure_mode = classify_failure_mode(
                text=gen_text, completed=completed, finish_reason=finish)
            sampling_payload = {**base_sampling, "max_tokens": int(sp.max_tokens),
                                "skip_special_tokens": False,
                                "seed": int(w["seed"])}
            rows.append({
                "trace_id": _trace_id(args.gen_model, w["instance_id"], w["seed"],
                                      w["prompt"], gen_text),
                "gen_model": args.gen_model, "gen_model_id": gm.hf_id,
                "generation_kind": str(gm.kind),
                "instance_id": w["instance_id"], "task_type": w["task_type"], "seed": w["seed"],
                "difficulty_raw": w.get("difficulty_raw"), "reference_answer": w.get("reference_answer"),
                "instance_metadata": w.get("metadata", "{}"), "prompt": w["prompt"],
                "full_text": gen_text, "think_text": think_text, "answer_text": answer_text,
                "reasoning_text_for_analysis": rtext,
                "n_new_tokens": len(o.outputs[0].token_ids), "completed": completed,
                "finish_reason": finish, "failure_mode": failure_mode,
                "hard_loop": loop_collapse_detect(gen_text), "soft_loop": soft_loop_detect(gen_text),
                "decode_temperature": float(base_sampling["temperature"]),
                "decode_top_p": float(base_sampling["top_p"]),
                "decode_top_k": int(base_sampling["top_k"]),
                "sampling_seed": int(w["seed"]),
                "sampling_params": json.dumps(sampling_payload, sort_keys=True),
                "thinking_style": style, "parse_status": parsed["parse_status"],
                "answer_source": parsed["answer_source"],
                "close_tag_count": parsed["close_tag_count"],
                "task_fingerprint": _task_fingerprint(w),
                "task_set_fingerprint": task_set_fingerprint,
                "generation_fingerprint": generation_fingerprint,
                "generation_version": GENERATION_VERSION,
            })
        new = pl.DataFrame(rows)
        combined = (pl.concat([combined, new], how="diagonal_relaxed")
                    if combined is not None else new)
        if combined.select(pl.struct(["gen_model", "instance_id", "seed"]).n_unique()).item() != combined.height:
            raise RuntimeError("duplicate natural keys detected while checkpointing generation")
        tmp = out_path.with_suffix(".parquet.tmp")
        combined.write_parquet(tmp); tmp.replace(out_path)
        print(f"[gen:{args.gen_model}] checkpoint {batch_start + len(batch)}/{len(pending)} "
              f"in {time.time()-t0:.1f}s -> {out_path} ({combined.height} total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
