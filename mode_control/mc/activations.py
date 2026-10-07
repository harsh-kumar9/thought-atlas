"""Stage A/B: residual-stream activations under teacher forcing (single GPU).

Stage A (baseline): for every valid Math/Idea baseline trace, feed the original prompt and the
first K=64 tokens of the stored generation; pool the residual at every layer at
  last_prompt : the last prompt position (the '\\n' after '<think>')
  mean32      : mean over generated tokens 1..32
  mean64      : mean over generated tokens 1..64
Stage B (paired strategy prompts): for a subset of tasks and each paraphrase pair, teacher-force
the SAME first 32 stored tokens under the checking and the exploratory system prompt and store
the per-layer difference (exploratory - checking) at last_prompt and mean32.

Outputs (float16 npz):
  data/acts/stageA_k64.npz : acts [n, L, 3, d], trace_id, instance_id, task_type
  data/acts/stageB_k32.npz : diffs [m, P, L, 2, d], plus per-condition means
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
import torch

from .common import (DATA, Capture, decoder_layers, hidden_size, load_model, load_strategy_prompts, load_tokenizer,
                     render_prompt, stable_fold)
from .phenotype import load_baseline_traces

PHASES_A = {"last_prompt": None, "mean32": None, "mean64": None}


def _batch_forward(model, tok, rendered: list[str], gen_ids: list[list[int]], K: int, positions: dict):
    """Right-aligned batch: prompt tokens + exactly K generation tokens; left padding."""
    ids = []
    for p, g in zip(rendered, gen_ids):
        pid = tok(p, add_special_tokens=False)["input_ids"]
        ids.append(pid + g[:K])
    T = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(ids), T), pad, dtype=torch.long)
    attn = torch.zeros((len(ids), T), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, T - len(x):] = torch.tensor(x); attn[i, T - len(x):] = 1
    layers = decoder_layers(model)
    caps = [Capture(positions) for _ in layers]
    hs = [l.register_forward_hook(c) for l, c in zip(layers, caps)]
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device), use_cache=False)
    finally:
        for h in hs: h.remove()
    # -> [B, L, n_phase, d]
    names = list(positions)
    return np.stack([np.stack([c.out[n] for n in names], axis=1) for c in caps], axis=1)


def stage_a(model, tok, K: int = 64, batch: int = 8):
    """K=64: phases last_prompt, mean32, mean64. K>64 adds 'late' = mean over generated tokens 65..K."""
    tr = load_baseline_traces().sort("trace_id")
    gen_ids = [tok(t, add_special_tokens=False)["input_ids"][:K] for t in tr["full_text"].to_list()]
    ok = [len(g) == K for g in gen_ids]
    print(f"[A] {tr.height} baseline traces, {sum(ok)} with >= {K} generated tokens")
    tr = tr.filter(pl.Series(ok)); gen_ids = [g for g, o in zip(gen_ids, ok) if o]
    rendered = [render_prompt(tok, p) for p in tr["prompt"].to_list()]
    positions = {"last_prompt": (-K - 1, -K), "mean32": (-K, -K + 32), "mean64": (-K, -K + 64)}
    if K > 64:
        positions["late"] = (-K + 64, 10**9)
    out = []
    t0 = time.time()
    order = np.argsort([len(r) for r in rendered])
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        acts = _batch_forward(model, tok, [rendered[i] for i in idx], [gen_ids[i] for i in idx], K, positions)
        out.append((idx, acts))
        if (s // batch) % 10 == 0:
            print(f"[A] {s + len(idx)}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    acts = np.zeros((len(order),) + out[0][1].shape[1:], dtype=np.float16)
    for idx, a in out: acts[idx] = a
    d = DATA / "acts"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"stageA_k{K}.npz", acts=acts, trace_id=np.array(tr["trace_id"].to_list()),
             instance_id=np.array(tr["instance_id"].to_list()), task_type=np.array(tr["task_type"].to_list()),
             phases=np.array(list(positions)))
    print(f"[A] saved {acts.shape} -> stageA_k{K}.npz in {time.time() - t0:.0f}s")


def stage_b(model, tok, K: int = 32, n_per_domain: int = 150, batch: int = 8, seed: int = 20260903):
    """K=32: phases last_prompt, mean32. K>64 adds 'late' = tokens 65..K (same teacher-forced tokens)."""
    prompts = load_strategy_prompts()
    P = len(prompts["checking"]); assert P == len(prompts["exploratory"])
    tr = load_baseline_traces().sort("trace_id")
    gen_ids = [tok(t, add_special_tokens=False)["input_ids"][:K] for t in tr["full_text"].to_list()]
    tr = tr.filter(pl.Series([len(g) == K for g in gen_ids])); gen_ids = [g for g in gen_ids if len(g) == K]
    rng = np.random.default_rng(seed)
    pick = []
    for dom in ["math", "idea"]:
        ids = np.where((tr["task_type"] == dom).to_numpy())[0]
        pick.extend(rng.choice(ids, size=min(n_per_domain, len(ids)), replace=False).tolist())
    pick = sorted(pick)
    sub = tr[pick]; gsub = [gen_ids[i] for i in pick]
    positions = {"last_prompt": (-K - 1, -K), "mean32": (-K, -K + 32)}
    if K > 64:
        positions["late"] = (-K + 64, 10**9)
    NP = len(positions)
    L = len(decoder_layers(model)); d = hidden_size(model)
    diffs = np.zeros((sub.height, P, L, NP, d), dtype=np.float16)
    means = {"checking": np.zeros((P, L, NP, d), np.float32), "exploratory": np.zeros((P, L, NP, d), np.float32)}
    t0 = time.time()
    for p in range(P):
        acts = {}
        for cond in ["checking", "exploratory"]:
            rendered = [render_prompt(tok, u, system=prompts[cond][p]) for u in sub["prompt"].to_list()]
            a = np.zeros((sub.height, L, NP, d), np.float16)
            order = np.argsort([len(r) for r in rendered])
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                a[idx] = _batch_forward(model, tok, [rendered[i] for i in idx], [gsub[i] for i in idx], K, positions)
            acts[cond] = a; means[cond][p] = a.astype(np.float32).mean(0)
        diffs[:, p] = (acts["exploratory"].astype(np.float32) - acts["checking"].astype(np.float32)).astype(np.float16)
        print(f"[B] paraphrase {p + 1}/{P} done {time.time() - t0:.0f}s", flush=True)
    dd = DATA / "acts"; dd.mkdir(parents=True, exist_ok=True)
    np.savez(dd / f"stageB_k{K}.npz", diffs=diffs, mean_checking=means["checking"], mean_exploratory=means["exploratory"],
             trace_id=np.array(sub["trace_id"].to_list()), instance_id=np.array(sub["instance_id"].to_list()),
             task_type=np.array(sub["task_type"].to_list()), phases=np.array(list(positions)),
             prompts=json.dumps(prompts))
    print(f"[B] saved {diffs.shape} -> stageB_k{K}.npz")


def smoke(model, tok):
    """Load check, template check, hook check, and a tiny steered generation for speed."""
    from .common import Steerer
    layers = decoder_layers(model)
    print("layers", len(layers), "hidden", hidden_size(model), "device", model.device)
    r = render_prompt(tok, "If $2^8=4^x$, what is the value of $x$?")
    print("rendered tail:", repr(r[-80:]))
    ids = tok(r, return_tensors="pt", add_special_tokens=False).to(model.device)
    d = hidden_size(model)
    U = np.zeros((d, 1), np.float32); U[0, 0] = 1.0
    st = Steerer(U, np.zeros(d, np.float32), "add", alpha=np.zeros(1, np.float32), pulse_k=8)
    h = layers[16].register_forward_hook(st)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    h.remove()
    n = out.shape[1] - ids["input_ids"].shape[1]
    print(f"generated {n} tokens in {time.time() - t0:.1f}s ({n / (time.time() - t0):.1f} tok/s, batch 1); "
          f"steerer calls={st.calls} steered_positions={st.n_steered_positions}")
    print(repr(tok.decode(out[0, -n:], skip_special_tokens=False)[:400]))
    # batched speed probe
    B = 16
    ids = tok([r] * B, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    n = out.shape[1] - ids["input_ids"].shape[1]
    print(f"batch {B}: {n} steps in {time.time() - t0:.1f}s -> {B * n / (time.time() - t0):.0f} tok/s aggregate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "A", "B", "AB"])
    ap.add_argument("--n-per-domain", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=None, help="teacher-forced tokens (default 64 for A, 32 for B)")
    a = ap.parse_args()
    tok = load_tokenizer(); model = load_model()
    if a.stage == "smoke": smoke(model, tok)
    if a.stage in ("A", "AB"): stage_a(model, tok, K=a.k or 64, batch=a.batch)
    if a.stage in ("B", "AB"): stage_b(model, tok, K=a.k or 32, n_per_domain=a.n_per_domain, batch=a.batch)


if __name__ == "__main__":
    main()
