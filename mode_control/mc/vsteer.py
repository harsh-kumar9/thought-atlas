"""Fixed-vector steering with vLLM (D-40, D-41).

Open-loop steering adds one constant vector to the residual stream after a decoder layer. That does not need
steer.py's HF generate loop: a forward hook on the vLLM decoder layer (eager, no torch.compile) makes the same
edit with continuous batching. Position rules:
  all    every token, prompt and generated (steer.py's sustained open-loop target mode)
  seal   the newest token of each request when it is a paragraph delimiter (any token containing "\\n\\n"),
         as SEAL's modeling_qwen2.py does (github.com/VITA-Group/SEAL). SEAL also stops after </think>; the
         flattened vLLM batch has no request identity, so delimiters in the final answer are steered too.
  delims every delimiter token in the batch (test only: lets `check` compare against HF in one forward)

Subcommands
  vectors  write data/vsteer/vec_<name>.npz from steer.py 'target' open-loop specs (A2, random, archetypes)
  seal     teacher-force corpus traces (HF); mean residual at delimiter tokens by the class of the thought that
           follows (SEAL keyword rules); S = mean(reflection + transition) - mean(execution), per layer
  check    HF vs vLLM teacher-forced log-probs, unsteered and steered (gate before any run)
  gen      generate a run from run.json with streaming checkpoints (resume skips done instance x seed)

A vector file holds v [d] and layer = index of the decoder layer whose OUTPUT is edited. SEAL's "layer L" is
the input to decoder layer L, i.e. the output of layer L - 1.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import polars as pl

from .common import DATA, DECODE, GEN_MODEL_KEY, MODEL_ID, decoder_layers, json_hash, render_prompt

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")          # hooks and _S must live in this process
VS = DATA / "vsteer"

# SEAL thought classes, verbatim from SEAL's hidden_analysis.py (including its "differenly" typo)
SEAL_CHECK_WORDS = ["verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect", "Let me check", "seems right"]
SEAL_CHECK_PREFIX = ["Wait"]
SEAL_SWITCH_WORDS = ["think differenly", "another way", "another approach", "another method", "another solution",
                     "another strategy", "another technique"]
SEAL_SWITCH_PREFIX = ["Alternatively"]


def seal_class(step: str) -> str:
    s = step.strip(" ").strip("\n").lower()
    if any(s.startswith(p.lower()) for p in SEAL_CHECK_PREFIX) or any(w.lower() in s for w in SEAL_CHECK_WORDS):
        return "check"
    if any(s.startswith(p.lower()) for p in SEAL_SWITCH_PREFIX) or any(w.lower() in s for w in SEAL_SWITCH_WORDS):
        return "switch"
    return "other"


def delimiter_ids(tok) -> list[int]:
    return sorted(i for t, i in tok.get_vocab().items() if "ĊĊ" in t)


def load_vector(name: str) -> dict:
    z = np.load(VS / f"vec_{name}.npz", allow_pickle=True)
    return {"v": z["v"].astype(np.float32), "layer": int(z["layer"]), "meta": json.loads(str(z["meta"]))}


# ---------------------------------------------------------------- vectors from the signature controller
def cmd_vectors(spec_path: str):
    """spec: list of {"name", "cond": steer.py target cond (open_loop), "ref_domain"}. Reproduces the exact vector
    Steerer adds in open-loop mode: gain * open_norm * unit((phi* - domain mean) * mask @ G^T)."""
    from .steer import build_steerers
    VS.mkdir(parents=True, exist_ok=True)
    for s in json.loads(Path(spec_path).read_text()):
        (layer, st), = build_steerers(s["cond"], None)
        R = st.readout
        phi = R["phi_star_abs"] - np.asarray(R["domain_means"][s["ref_domain"]], np.float32)
        u = (phi * R["mask"].numpy()) @ R["G"].numpy().T
        v = R["gain"] * R["open_norm"] * u / (np.linalg.norm(u) + 1e-6)
        meta = {"source": "signature_controller", "cond": s["cond"], "ref_domain": s["ref_domain"], "norm": float(np.linalg.norm(v))}
        np.savez(VS / f"vec_{s['name']}.npz", v=v.astype(np.float32), layer=layer, meta=json.dumps(meta))
        print(f"[vectors] {s['name']}: layer {layer}, |v| {np.linalg.norm(v):.3f}")


# ---------------------------------------------------------------- SEAL vector extraction (HF, teacher forcing)
def cmd_seal(runs: list[str], layers: list[int], max_think_tokens: int):
    import torch
    from .common import load_model, load_tokenizer, _layer_output
    VS.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(); model = load_model(); dl = decoder_layers(model)
    split = set(delimiter_ids(tok))
    tr = pl.concat([pl.read_parquet(p) for r in runs for p in sorted((DATA / "runs" / r).glob("traces_*.parquet"))], how="diagonal_relaxed")
    tr = tr.filter((pl.col("condition") == "baseline") & pl.col("think_text").is_not_null() & (pl.col("think_text").str.len_chars() > 20))
    d = int(getattr(model.config, "text_config", model.config).hidden_size)
    sums = {l: {c: np.zeros(d, np.float64) for c in ("check", "switch", "other")} for l in layers}
    counts = {c: 0 for c in ("check", "switch", "other")}
    grab = {"pos": None}; got = {}
    def hook(l):
        def f(module, args, output):
            got[l] = _layer_output(output)[0, grab["pos"], :].float().cpu().numpy()
        return f
    hs = [dl[l].register_forward_hook(hook(l)) for l in layers]
    ck = VS / "seal_partial.npz"; start = 0
    if ck.exists():                                                   # resume after preemption
        z = np.load(ck, allow_pickle=True); start = int(z["next"]); counts = json.loads(str(z["counts"]))
        sums = {l: {c: z[f"{l}|{c}"].astype(np.float64) for c in ("check", "switch", "other")} for l in layers}
        print(f"[seal] resuming at trace {start}", flush=True)
    def save_partial(nxt):
        np.savez(VS / "seal_partial.tmp.npz", next=nxt, counts=json.dumps(counts), **{f"{l}|{c}": sums[l][c] for l in layers for c in sums[l]})
        (VS / "seal_partial.tmp.npz").replace(ck)
    t0 = time.time()
    try:
        for n, r in enumerate(tr.iter_rows(named=True)):
            if n < start:
                continue
            if n % 50 == 0 and n > start:
                save_partial(n)
            p_ids = tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            t_ids = tok(r["think_text"], add_special_tokens=False)["input_ids"][:max_think_tokens]
            idx = [j for j, t in enumerate(t_ids) if t in split]
            if not idx:
                continue
            bounds = idx + [len(t_ids)]
            cls = [seal_class(tok.decode(t_ids[bounds[k] + 1:bounds[k + 1]])) for k in range(len(idx))]
            grab["pos"] = torch.tensor([len(p_ids) + j for j in idx])
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + t_ids], device=model.device), logits_to_keep=1)
            for l in layers:
                H = got[l]
                for c in ("check", "switch", "other"):
                    m = np.array([x == c for x in cls])
                    if m.any():
                        sums[l][c] += H[m].sum(0)
            for c in cls:
                counts[c] += 1
            if n % 100 == 0:
                print(f"[seal] {n}/{tr.height} traces, counts {counts}, {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in hs: h.remove()
    out = {"counts": json.dumps(counts), "layers": np.array(layers), "runs": json.dumps(runs), "max_think_tokens": max_think_tokens}
    for l in layers:
        m_rt = (sums[l]["check"] + sums[l]["switch"]) / max(counts["check"] + counts["switch"], 1)
        m_e = sums[l]["other"] / max(counts["other"], 1)
        out[f"S|{l}"] = (m_rt - m_e).astype(np.float32)          # SEAL's steer_vec; applied with coef -1
        out[f"mean_rt|{l}"] = m_rt.astype(np.float32); out[f"mean_e|{l}"] = m_e.astype(np.float32)
    np.savez(VS / "seal_means.npz", **out)
    ck.unlink(missing_ok=True)
    print(f"[seal] done: counts {counts}; |S| by output layer " + ", ".join(f"{l}: {np.linalg.norm(out[f'S|{l}']):.2f}" for l in layers))


def write_seal_vector(name: str, layer_out: int, coef: float = -1.0, norm: float | None = None):
    """vec_<name>.npz from seal_means.npz: coef * S at the output of `layer_out` (SEAL layer layer_out + 1).
    With `norm`, the vector is rescaled to that norm (dose-matched variant)."""
    z = np.load(VS / "seal_means.npz", allow_pickle=True)
    v = coef * z[f"S|{layer_out}"]
    if norm is not None:
        v = v / np.linalg.norm(v) * norm
    meta = {"source": "seal", "seal_layer": layer_out + 1, "coef": coef, "rescaled_to": norm, "counts": json.loads(str(z["counts"]))}
    np.savez(VS / f"vec_{name}.npz", v=v.astype(np.float32), layer=layer_out, meta=json.dumps(meta))
    print(f"[seal vec] {name}: output of layer {layer_out}, |v| {np.linalg.norm(v):.3f}")


# ---------------------------------------------------------------- prompt-contrast directions (D-44 step 2)
def cmd_promptdir(run: str, cond: str, name: str, n_pos: int, seeds: list[int], scales: list[float]):
    """The D-04 recipe for a strategy instruction: teacher-force the same generated tokens with and without the
    instruction as system message, and average the layer-16 difference over the positions that produce generated tokens
    1..n_pos. Token sources: the unsteered traces and the prompted traces of `run` (both, for symmetry). Writes
    vec_<name>_x<scale>.npz (the raw mean difference times scale) and a norm-matched random vector for the largest scale."""
    import torch
    from .common import load_model, load_tokenizer, _layer_output
    rd = DATA / "runs" / run
    spec = json.loads((rd / "run.json").read_text())
    system = next(c["system"] for c in spec["conditions"] if c["name"] == cond)
    tok = load_tokenizer(); model = load_model(); dl = decoder_layers(model)
    tr = pl.concat([pl.read_parquet(rd / f"traces_{c}.parquet", columns=["instance_id", "seed", "prompt", "gen_token_ids", "condition"])
                    for c in ("baseline", cond)]).filter(pl.col("seed").is_in(seeds))
    grab = {}
    h = dl[16].register_forward_hook(lambda m, a, o: grab.__setitem__("h", _layer_output(o)[0].float()))
    diffs = []
    try:
        for r in tr.iter_rows(named=True):
            gen = list(r["gen_token_ids"])[:n_pos]
            if len(gen) < n_pos:
                continue
            out = []
            for sysmsg in (system, None):
                p = tok(render_prompt(tok, r["prompt"], system=sysmsg), add_special_tokens=False)["input_ids"]
                with torch.no_grad():
                    model(input_ids=torch.tensor([p + gen], device=model.device), logits_to_keep=1)
                out.append(grab["h"][len(p) - 1:len(p) - 1 + n_pos].mean(0).cpu().numpy())
            diffs.append(out[0] - out[1])
    finally:
        h.remove()
    D = np.stack(diffs); S = D.mean(0)
    cons = float(np.mean([x @ S / (np.linalg.norm(x) * np.linalg.norm(S) + 1e-9) for x in D]))
    print(f"[promptdir] {cond}: {len(D)} traces, |S| {np.linalg.norm(S):.3f}, mean cosine of per-trace differences with S {cons:.3f}")
    meta = {"source": "prompt_contrast", "run": run, "cond": cond, "system": system, "n_pos": n_pos, "seeds": seeds, "n": len(D),
            "raw_norm": float(np.linalg.norm(S)), "consistency": cons}
    for sc in scales:
        np.savez(VS / f"vec_{name}_x{sc:g}.npz", v=(sc * S).astype(np.float32), layer=16, meta=json.dumps(meta | {"scale": sc}))
    rng = np.random.default_rng(11); u = rng.standard_normal(S.shape[0]); u /= np.linalg.norm(u)
    top = max(scales) * np.linalg.norm(S)
    np.savez(VS / f"vec_rand_{name}_x{max(scales):g}.npz", v=(u * top).astype(np.float32), layer=16, meta=json.dumps({"source": "random", "norm": float(top)}))
    for sc in scales:
        print(f"[promptdir] vec_{name}_x{sc:g}: |v| {sc * np.linalg.norm(S):.3f}")


# ---------------------------------------------------------------- vLLM steering hooks (in-process engine)
_S = {"v": None, "layer": None, "rule": "all", "split": None, "ids": None, "n_steered": 0, "n_mismatch": 0}


def _query_start_loc():
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    md = md[0] if isinstance(md, list) else md
    for m in (md or {}).values():
        q = getattr(m, "query_start_loc", None)
        if q is not None:
            return q
    return None


def _install(model):
    """Called once through LLM.apply_model: a pre-hook on the token embedding keeps this step's token ids; a hook
    on every decoder layer edits the output of the layer named in _S."""
    import torch
    layers = decoder_layers(model)
    embs = [(n, m) for n, m in model.named_modules() if n.endswith("embed_tokens")]
    embs.sort(key=lambda x: "language_model" not in x[0])
    assert embs, "embed_tokens not found"
    emb = embs[0][1]

    def keep_ids(mod, args):
        _S["ids"] = args[0]
    emb.register_forward_pre_hook(keep_ids)

    def make(idx):
        def hook(mod, args, out):
            if _S["v"] is None or _S["layer"] != idx:
                return None
            h, res = out
            if _S["rule"] == "all":
                _S["n_steered"] += h.shape[0]
                return ((h.float() + _S["v"]).to(h.dtype), res)
            ids = _S["ids"]
            if ids is None or ids.shape[0] != h.shape[0]:
                _S["n_mismatch"] += 1
                return None
            isdel = torch.isin(ids, _S["split"])
            if _S["rule"] == "delims":
                mask = isdel
            else:                                                       # seal: newest token of each request
                q = _query_start_loc()
                if q is None:
                    _S["n_mismatch"] += 1
                    return None
                last = (q[1:] - 1).clamp(min=0, max=h.shape[0] - 1)
                mask = torch.zeros_like(isdel); mask[last] = isdel[last]
            n = int(mask.sum())
            if n == 0:
                return None
            _S["n_steered"] += n
            h = h.clone(); h[mask] = (h[mask].float() + _S["v"]).to(h.dtype)
            return (h, res)
        return hook
    for i, lay in enumerate(layers):
        lay.register_forward_hook(make(i))
    return len(layers)


def _set_steer(llm, tok, vec: dict | None, rule: str = "all", coef: float = 1.0):
    import torch
    dev = torch.device("cuda:0")
    if vec is None:
        _S.update(v=None, layer=None)
    else:
        _S.update(v=torch.as_tensor(coef * vec["v"], dtype=torch.float32, device=dev), layer=vec["layer"], rule=rule,
                  split=torch.as_tensor(delimiter_ids(tok), device=dev))
    _S.update(n_steered=0, n_mismatch=0)


def build_vllm(max_model_len: int, gpu_mem: float = 0.88, max_num_seqs: int = 256, hooks: bool = True):
    """hooks=False: plain engine with CUDA graphs and torch.compile, for runs with no steering vector."""
    from vllm import LLM
    kw = dict(enforce_eager=True, compilation_config={"mode": 0}) if hooks else {}
    llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
              max_num_seqs=max_num_seqs, max_num_batched_tokens=8192, enable_prefix_caching=False, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    if hooks:
        n = llm.apply_model(_install)
        print(f"[vllm] hooks installed on {n} layers", flush=True)
    return llm


# ---------------------------------------------------------------- equivalence check
def cmd_check(a2: str, seal: str, n_traces: int, think_tokens: int):
    """Teacher-forced log-probs of prompt + first `think_tokens` reasoning tokens of corpus Math traces, in HF
    (original Steerer for A2; plain hook for SEAL) and in vLLM (hooks above), unsteered and steered. Pass: the
    steering effect per token agrees across engines (corr > 0.9, norm ratio 0.8-1.25). Then a short seal-rule
    generation checks that steered events match delimiter tokens fed back."""
    import torch
    from .common import load_model, load_tokenizer, _layer_output, _replace_output
    from .steer import build_steerers
    tok = load_tokenizer()
    tr = pl.read_parquet(DATA / "runs" / "corpus1_math_a" / "traces_baseline.parquet").filter(pl.col("think_text").is_not_null()).head(n_traces)
    seqs = []
    for r in tr.iter_rows(named=True):
        p = tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        t = tok(r["think_text"], add_special_tokens=False)["input_ids"][:think_tokens]
        seqs.append((p, t))
    va, vs = load_vector(a2), load_vector(seal)
    split = set(delimiter_ids(tok))
    # HF
    model = load_model(); dl = decoder_layers(model)
    a2_cond = va["meta"]["cond"]
    def hf_lp(ids, hooks):
        hs = [dl[l].register_forward_hook(f) for l, f in hooks]
        try:
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([ids], device=model.device)).logits[0].float()
        finally:
            for h in hs: h.remove()
        lp = torch.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:], device=lg.device)[:, None])[:, 0]
        return lp.cpu().numpy()
    hf = {"base": [], "a2": [], "seal": []}
    for p, t in seqs:
        ids = p + t
        hf["base"].append(hf_lp(ids, []))
        (l, st), = build_steerers(a2_cond, None)
        st.readout["phi_star"] = torch.as_tensor((st.readout["phi_star_abs"] - np.asarray(st.readout["domain_means"][va["meta"]["ref_domain"]], np.float32))[None])
        st.device = None; st.reset()
        hf["a2"].append(hf_lp(ids, [(l, st)]))
        mask = torch.tensor([x in split for x in ids], device=model.device)
        v = torch.as_tensor(vs["v"], device=model.device)                      # the file already carries SEAL's coef
        def seal_hook(module, args, output, mask=mask, v=v):
            h = _layer_output(output).clone(); h[0, mask] = (h[0, mask].float() + v).to(h.dtype)
            return _replace_output(output, h)
        hf["seal"].append(hf_lp(ids, [(vs["layer"], seal_hook)]))
    del model; torch.cuda.empty_cache()
    # vLLM
    from vllm import SamplingParams
    llm = build_vllm(max_model_len=max(len(p) + len(t) for p, t in seqs) + 1024, gpu_mem=0.6)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    def vl_lp():
        outs = llm.generate([{"prompt_token_ids": p + t} for p, t in seqs], sp, use_tqdm=False)
        res = []
        for (p, t), o in zip(seqs, outs):
            ids = p + t
            res.append(np.array([o.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids))], np.float32))
        return res
    vl = {}
    _set_steer(llm, tok, None); vl["base"] = vl_lp()
    _set_steer(llm, tok, va, "all"); vl["a2"] = vl_lp(); n_a2 = _S["n_steered"]
    _set_steer(llm, tok, vs, "delims"); vl["seal"] = vl_lp(); n_seal = _S["n_steered"]
    report = {"n_steered_a2": n_a2, "n_steered_seal_delims": n_seal, "mismatch": _S["n_mismatch"]}
    for k in ("base", "a2", "seal"):
        diffs = [np.abs(h[len(p) - 1:] - v[len(p) - 1:]) for h, v, (p, _) in zip(hf[k], vl[k], seqs)]
        report[f"abs_diff_{k}"] = float(np.mean(np.concatenate(diffs)))
    for k in ("a2", "seal"):
        dh = np.concatenate([h[len(p) - 1:] - b[len(p) - 1:] for h, b, (p, _) in zip(hf[k], hf["base"], seqs)])
        dv = np.concatenate([h[len(p) - 1:] - b[len(p) - 1:] for h, b, (p, _) in zip(vl[k], vl["base"], seqs)])
        report[f"effect_corr_{k}"] = float(np.corrcoef(dh, dv)[0, 1])
        report[f"effect_norm_ratio_{k}"] = float(np.linalg.norm(dv) / (np.linalg.norm(dh) + 1e-9))
        report[f"effect_mean_abs_hf_{k}"] = float(np.mean(np.abs(dh)))
    # seal rule during decoding: steered events should equal delimiter tokens that were fed back as inputs
    _set_steer(llm, tok, vs, "seal")
    outs = llm.generate([{"prompt_token_ids": p} for p, _ in seqs], SamplingParams(max_tokens=512, temperature=0.6, top_p=0.95, top_k=20, seed=1), use_tqdm=False)
    fed = sum(sum(1 for x in o.outputs[0].token_ids[:-1] if x in split) for o in outs)
    fed += sum(1 for p, _ in seqs if p[-1] in split)
    report.update(seal_rule_steered=_S["n_steered"], seal_rule_delims_fed=fed, seal_rule_mismatch=_S["n_mismatch"])
    report["pass"] = bool(all(report[f"effect_corr_{k}"] > 0.9 and 0.8 < report[f"effect_norm_ratio_{k}"] < 1.25 for k in ("a2", "seal"))
                          and report["seal_rule_steered"] == report["seal_rule_delims_fed"])
    VS.mkdir(parents=True, exist_ok=True)
    (VS / "check.json").write_text(json.dumps(report, indent=1))
    print("[check]", json.dumps(report, indent=1))


# ---------------------------------------------------------------- generation
def _tasks(spec: dict, rd: Path) -> pl.DataFrame:
    if "tasks_from" in spec:
        return pl.read_parquet(DATA / "runs" / spec["tasks_from"] / "tasks.parquet")
    from .steer import select_tasks
    excl = set(spec.get("exclude_instances", []))
    for r in spec.get("exclude_runs", []):
        excl |= set(pl.read_parquet(DATA / "runs" / r / "tasks.parquet")["instance_id"].to_list())
    return select_tasks(spec["n_per_domain"], spec["task_seed"], valid_only=spec.get("valid_only", True),
                        domains=spec["domains"], exclude=sorted(excl))


def cmd_gen(run: str, conds_filter: str | None, flush_s: float):
    from vllm import SamplingParams
    from .common import load_tokenizer
    from src.utils.parse import parse_generation_detailed, is_completed
    from src.utils.characterize import soft_loop_detect, loop_collapse_detect
    rd = DATA / "runs" / run
    spec = json.loads((rd / "run.json").read_text())
    conds = spec["conditions"]
    if conds_filter:
        want = conds_filter.split(","); conds = [c for c in conds if c["name"] in want]
    tasks = _tasks(spec, rd)
    if not (rd / "tasks.parquet").exists():
        tasks.write_parquet(rd / "tasks.parquet")
    tok = load_tokenizer()
    max_new, window = int(spec["max_new_tokens"]), int(spec["window"])
    run_fp = json_hash({k: spec[k] for k in ["max_new_tokens", "window", "seeds"]} | {"tasks": sorted(tasks["instance_id"].to_list())})
    llm = None
    for cond in conds:
        out_path = rd / f"traces_{cond['name']}.parquet"
        existing = pl.read_parquet(out_path) if out_path.exists() else None
        done = set(zip(existing["instance_id"].to_list(), existing["seed"].to_list())) if existing is not None else set()
        # optional per-condition subset (dose arms): the first n_tasks tasks in run order and/or a subset of the seeds.
        # Sampling seeds still come from the full run fingerprint, so a subset arm pairs with the full arms.
        c_tasks = tasks.head(int(cond["n_tasks"])) if cond.get("n_tasks") else tasks
        c_seeds = [s for s in spec["seeds"] if s in cond["seeds"]] if cond.get("seeds") else spec["seeds"]
        work = [(r, s) for r in c_tasks.iter_rows(named=True) for s in c_seeds if (r["instance_id"], s) not in done]
        print(f"[vgen:{cond['name']}] {len(work)} pending of {tasks.height * len(spec['seeds'])}", flush=True)
        if not work:
            continue
        needs_hooks = any(c.get("vector") for c in conds)
        if llm is None:
            llm = build_vllm(max_model_len=max_new + 4096, hooks=needs_hooks)
        system = cond.get("system")
        vec = load_vector(cond["vector"]) if cond.get("vector") else None
        if needs_hooks:
            _set_steer(llm, tok, vec, cond.get("positions", "all"), float(cond.get("coef", 1.0)))
        eng = llm.llm_engine
        # partial generations saved at each flush; after a preemption, unfinished traces continue from their saved
        # tokens (prompt + partial as the new prompt, remaining budget, a fresh seed). Not for the seal rule: a
        # re-prefill would not re-steer the delimiters already generated, so those traces restart instead.
        part_path = rd / f"partial_{cond['name']}.parquet"
        partial = {}
        if part_path.exists() and cond.get("positions", "all") != "seal":
            pp = pl.read_parquet(part_path)
            partial = {(i, int(sd)): list(t) for i, sd, t in zip(pp["instance_id"].to_list(), pp["seed"].to_list(), pp["partial_ids"].to_list())}
        meta, latest = {}, {}
        n_resumed = 0
        for r, s in work:
            ids = tok(render_prompt(tok, r["prompt"], system=system), add_special_tokens=False)["input_ids"]
            rid = f"{cond['name']}|{r['instance_id']}|{s}"
            seed = int(json_hash([run_fp, r["instance_id"], s])[:8], 16)           # common random numbers across conditions
            prev = partial.get((r["instance_id"], int(s)), [])
            if len(prev) >= max_new:
                prev = []
            n_resumed += bool(prev)
            sp = SamplingParams(max_tokens=max_new - len(prev), seed=seed + len(prev), skip_special_tokens=False, **DECODE)
            eng.add_request(rid, {"prompt_token_ids": ids + prev}, sp)
            meta[rid] = (r, s, seed, prev)
        if n_resumed:
            print(f"[vgen:{cond['name']}] resuming {n_resumed} partial traces", flush=True)
        rows, t0, t_flush, n_done, n_tok = [], time.time(), time.time(), 0, 0
        def flush():
            nonlocal existing, rows
            if rows:
                new = pl.DataFrame(rows)
                existing = pl.concat([existing, new], how="diagonal_relaxed") if existing is not None else new
                tmp = out_path.with_suffix(".parquet.tmp"); existing.write_parquet(tmp); tmp.replace(out_path)
                rows = []
            if latest:
                pp = pl.DataFrame({"instance_id": [meta[k][0]["instance_id"] for k in latest], "seed": [int(meta[k][1]) for k in latest],
                                   "partial_ids": [meta[k][3] + list(v) for k, v in latest.items()]}, schema_overrides={"partial_ids": pl.List(pl.Int32)})
                tmp = part_path.with_suffix(".parquet.tmp"); pp.write_parquet(tmp); tmp.replace(part_path)
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not o.finished:
                    latest[o.request_id] = o.outputs[0].token_ids
                    continue
                latest.pop(o.request_id, None)
                r, s, seed, prev = meta[o.request_id]
                g = o.outputs[0]
                gen_ids = prev + list(g.token_ids)
                text = tok.decode(gen_ids, skip_special_tokens=False) if prev else g.text
                for t in (tok.eos_token, tok.pad_token, "<|im_end|>"):
                    if t and text.endswith(t): text = text[: -len(t)]
                finish = "stop" if g.finish_reason == "stop" else "length"
                parsed = parse_generation_detailed(text, "reasoning", finish_reason=finish)
                think, answer = parsed["think_text"], parsed["answer_text"]
                completed = is_completed(kind="reasoning", finish_reason=finish, has_close_tag=parsed["close_tag_count"] > 0 or parsed["parse_status"] == "direct_answer_no_close", has_answer=bool(answer))
                rtext = None
                if think:
                    tids = tok(think, add_special_tokens=False)["input_ids"]
                    rtext = think if len(tids) <= window else tok.decode(tids[:window])
                rows.append({
                    "trace_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "\0".join([run_fp, cond["name"], r["instance_id"], str(s), text]))),
                    "run_fingerprint": run_fp, "condition": cond["name"], "condition_spec": json.dumps(cond, sort_keys=True),
                    "gen_model": GEN_MODEL_KEY, "gen_model_id": MODEL_ID, "generation_kind": "reasoning", "engine": "vllm",
                    "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": int(s),
                    "difficulty_raw": r.get("difficulty_raw"), "reference_answer": r.get("reference_answer"),
                    "instance_metadata": r.get("metadata", "{}"), "prompt": r["prompt"], "system_prompt": system,
                    "full_text": text, "think_text": think, "answer_text": answer, "reasoning_text_for_analysis": rtext,
                    "gen_token_ids": gen_ids, "n_new_tokens": len(gen_ids), "completed": completed, "finish_reason": finish,
                    "resumed_at": len(prev),
                    "parse_status": parsed["parse_status"], "close_tag_count": parsed["close_tag_count"],
                    "hard_loop": loop_collapse_detect(text), "soft_loop": soft_loop_detect(text),
                    "steered_positions": None, "steered_layers": json.dumps([vec["layer"]] if vec else []),
                    "decode": json.dumps({**DECODE, "max_new_tokens": max_new, "phenotype_window": window, "request_seed": seed}),
                })
                n_done += 1; n_tok += len(g.token_ids)
            if time.time() - t_flush > flush_s:
                flush(); t_flush = time.time()
                print(f"[vgen:{cond['name']}] {n_done}/{len(work)} done, {n_tok / (time.time() - t0):.0f} tok/s (finished traces), "
                      f"steered events {_S['n_steered']}, mismatches {_S['n_mismatch']}", flush=True)
        latest.clear(); flush(); part_path.unlink(missing_ok=True)
        print(f"[vgen:{cond['name']}] finished {n_done} in {time.time() - t0:.0f}s; steered events {_S['n_steered']}, mismatches {_S['n_mismatch']}", flush=True)
        (rd / f"steer_log_{cond['name']}.json").write_text(json.dumps({"n_steered": _S["n_steered"], "n_mismatch": _S["n_mismatch"], "seconds": time.time() - t0}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vectors"); v.add_argument("--spec", required=True)
    s = sub.add_parser("seal"); s.add_argument("--runs", default="corpus1_math_a,corpus1_math_b"); s.add_argument("--layers", default="16,19,22,25")
    s.add_argument("--max-think-tokens", type=int, default=12288)
    sv = sub.add_parser("sealvec"); sv.add_argument("--name", required=True); sv.add_argument("--layer-out", type=int, required=True)
    sv.add_argument("--coef", type=float, default=-1.0); sv.add_argument("--norm", type=float, default=None)
    c = sub.add_parser("check"); c.add_argument("--a2", default="ol_math_A2"); c.add_argument("--seal", default="seal_L23")
    c.add_argument("--n", type=int, default=4); c.add_argument("--think-tokens", type=int, default=1024)
    pd = sub.add_parser("promptdir"); pd.add_argument("--run", required=True); pd.add_argument("--cond", required=True); pd.add_argument("--name", required=True)
    pd.add_argument("--n-pos", type=int, default=64); pd.add_argument("--seeds", default="1"); pd.add_argument("--scales", default="1,2")
    g = sub.add_parser("gen"); g.add_argument("--run", required=True); g.add_argument("--conditions", default=None); g.add_argument("--flush-s", type=float, default=180)
    a = ap.parse_args()
    if a.cmd == "vectors": cmd_vectors(a.spec)
    elif a.cmd == "seal": cmd_seal(a.runs.split(","), [int(x) for x in a.layers.split(",")], a.max_think_tokens)
    elif a.cmd == "sealvec": write_seal_vector(a.name, a.layer_out, a.coef, a.norm)
    elif a.cmd == "check": cmd_check(a.a2, a.seal, a.n, a.think_tokens)
    elif a.cmd == "promptdir": cmd_promptdir(a.run, a.cond, a.name, a.n_pos, [int(x) for x in a.seeds.split(",")], [float(x) for x in a.scales.split(",")])
    else: cmd_gen(a.run, a.conditions, a.flush_s)


if __name__ == "__main__":
    main()
