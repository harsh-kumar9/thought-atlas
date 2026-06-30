"""src/judge/vllm_engine.py — Data-parallel vLLM backbone for Blackwell (sm_120).

This is the shared inference engine used by BOTH the trace generator and the judge.
It follows the Blackwell rule: a model that fits on one card is REPLICATED, not
sharded — one vLLM engine per GPU, dataset sharded across them, zero cross-GPU comms.
(For a model that doesn't fit one card, set tensor_parallel_size within one NUMA
island 0-3 or 4-7; never straddle.)

Two ways to use it:
  * `vllm serve` persistent OpenAI-compatible servers (one per GPU) + an OpenAI client.
    Best when you want the existing OpenAI-client code path. See scripts/serve_judges.sh.
  * This offline batch engine: simplest for a fixed dataset. The SLURM wrapper
    launches N single-GPU processes pinned by CUDA_VISIBLE_DEVICES and shards the
    dataset across them.

Guided JSON: judging forces a JSON schema via vLLM guided decoding (xgrammar), which
removes parse failures — the #1 judge headache at scale. temp=0 for reproducibility.

NOTE: the judge's *schema* and *prompt* are intentionally injected by the caller
(src/judge/run_judge.py), because how the Kim+Gandhi dual taxonomy maps onto
per-sentence labels is a design decision still being finalized. This engine is
taxonomy-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EngineConfig:
    model: str
    dtype: str = "bfloat16"          # bf16 if it fits one card (<=~30B in 64GB) -> no quant risk
    quantization: Optional[str] = None   # "fp8" to fit bigger / go faster; validate vs bf16 first
    kv_cache_dtype: Optional[str] = None # "fp8_e4m3" only helps long context
    tensor_parallel_size: int = 1     # >1 ONLY if model doesn't fit one card; keep within an island
    data_parallel_size: int = 1       # = number of GPUs to replicate across (the common case)
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.92
    max_num_batched_tokens: Optional[int] = None  # GDN/Mamba (Qwen3.5) requires >= 2096; set 4096
    max_num_seqs: Optional[int] = None  # cap batch concurrency; hybrid Mamba models (Qwen3.6) need <= Mamba cache blocks
    enforce_eager: bool = False       # keep CUDA graphs ON on Blackwell
    seed: int = 0
    extra: dict = field(default_factory=dict)  # passthrough (e.g. enable_expert_parallel=True for MoE)


def build_llm(cfg: EngineConfig):
    """Construct ONE vLLM replica (single GPU by default; TP only if a model won't fit one card).

    NOTE: vLLM >=0.22 dropped in-process `data_parallel_size` for the offline LLM() API. Data
    parallelism here = many of THESE single-GPU replicas, one per GPU, each over a dataset shard
    (the 'replicate-don't-shard' pattern). See generate_traces.py --shard/--num-shards, driven by
    a SLURM array. Do NOT pass data_parallel_size to LLM().
    """
    from vllm import LLM
    kwargs: dict[str, Any] = dict(
        model=cfg.model,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tensor_parallel_size,   # 1 for 8B; >1 only if it won't fit one card
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
        seed=cfg.seed,
        trust_remote_code=True,
    )
    if cfg.max_num_seqs is not None:
        kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
    if cfg.quantization:
        kwargs["quantization"] = cfg.quantization
    if cfg.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = cfg.kv_cache_dtype
        kwargs["calculate_kv_scales"] = True
    kwargs.update(cfg.extra)
    return LLM(**kwargs)


def make_sampling(*, temperature: float = 0.0, max_tokens: int = 1024,
                  top_p: float = 1.0, top_k: int = -1, seed: Optional[int] = None,
                  json_schema: Optional[dict] = None, **extra):
    """SamplingParams with optional guided-JSON. temp=0 => deterministic judge.

    `extra` forwards model-specific sampling knobs (e.g. presence_penalty for Qwen3.5, min_p,
    repetition_penalty); unknown keys are dropped defensively so a stray kwarg can't crash a run.

    vLLM renamed the structured-output knob: 0.22 uses `structured_outputs=StructuredOutputsParams`,
    older versions used `guided_decoding=GuidedDecodingParams`. Try the new API first, fall back.
    """
    from vllm import SamplingParams
    base = dict(temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k, seed=seed)
    # forward only sampling kwargs SamplingParams actually accepts (defensive against version drift)
    import inspect
    allowed = set(inspect.signature(SamplingParams).parameters) | set(getattr(SamplingParams, "__init__", object).__code__.co_varnames if hasattr(getattr(SamplingParams, "__init__", object), "__code__") else [])
    for k, v in extra.items():
        if k in allowed or k in {"presence_penalty", "frequency_penalty", "repetition_penalty", "min_p"}:
            base[k] = v
    if json_schema is None:
        return SamplingParams(**base)

    # 0.22+: structured_outputs / StructuredOutputsParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**base,
                              structured_outputs=StructuredOutputsParams(json=json_schema))
    except (ImportError, TypeError):
        pass
    # legacy: guided_decoding / GuidedDecodingParams
    from vllm.sampling_params import GuidedDecodingParams
    return SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=json_schema))


def chat_batch(llm, tokenizer, prompts: list[str], sampling, *,
               system: Optional[str] = None, max_input_tokens: Optional[int] = None) -> list[str]:
    """Render chat prompts with the model's template and run a batched generate.

    Passes enable_thinking=False where the template supports it: a thinking judge under guided
    JSON opens <think> and the grammar then forces the JSON out *before* it reasons -> all-zeros.
    Suppressing the think reflex makes the judgment land in the JSON itself. Harmless no-op for
    instruct models whose template ignores the kwarg.

    max_input_tokens: hard safety clamp. Char-budgeting upstream is approximate (symbol-dense text
    tokenizes denser than chars/3.5), so as a last guard we token-truncate any rendered prompt that
    still exceeds the window — one over-budget prompt must not abort the whole (expensive) batch.
    """
    rendered = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": p}]
        try:
            r = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            r = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        if max_input_tokens is not None:
            ids = tokenizer(r, add_special_tokens=False)["input_ids"]
            if len(ids) > max_input_tokens:
                # keep head (problem + instructions live at the top of these templates) + recent tail
                head = ids[: max_input_tokens // 4]
                tail = ids[-(max_input_tokens - len(head)):]
                r = tokenizer.decode(head) + "\n...[prompt truncated to fit context]...\n" + tokenizer.decode(tail)
        rendered.append(r)
    outs = llm.generate(rendered, sampling)
    # vLLM preserves input order
    return [o.outputs[0].text for o in outs]


def safe_json(text: str) -> Optional[dict]:
    """Guided decoding should already guarantee valid JSON; this is the belt-and-suspenders."""
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                return None
        return None
