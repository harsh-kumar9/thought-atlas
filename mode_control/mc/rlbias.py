"""P4a (D-45): train only a steering bias with RL and compare it with A2.

The model is frozen. One trainable vector b [d] is added to the residual stream after decoder layer 16 at every
position, the placement A2 uses. Rollouts come from vLLM with b applied by the vsteer hook; the policy-gradient
step runs in HF with b as the only parameter (the same edit, checked equal across engines in D-40).
Reward: 1 if the answer is correct within the token budget, else 0 (repo grader, bold stripped).
Loss: Dr. GRPO (Liu et al. 2025, arXiv 2503.20783): advantage = r - mean(group r), no std scaling; token log-probs
summed and divided by a constant (prompts x G x budget), so the objective adds no length preference of its own.
One on-policy gradient step per batch, Adam on b.
Train problems: the MATH-500 tasks outside the 199 evaluation tasks of D-36 (readout corpus 200 + SEAL validation 101).
State (b, Adam, step, metrics) is saved after every step, so a requeue resumes.

Subcommands: train (the loop above), export (write vec_rl_<run>_s<step>.npz for vsteer evaluation).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from .common import DATA, DECODE, TASKS_DIR, decoder_layers, json_hash, render_prompt, _layer_output, _replace_output
from .vsteer import VS, build_vllm, load_vector, _set_steer
from .budget import grade

RL = DATA / "rlbias"
LAYER = 16


def train_tasks() -> pl.DataFrame:
    ev = set(pl.read_parquet(DATA / "runs" / "sigpilot3_math" / "tasks.parquet")["instance_id"].to_list())
    return pl.read_parquet(TASKS_DIR / "math.parquet").filter(~pl.col("instance_id").is_in(sorted(ev))).sort("instance_id")


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cmd_train(a):
    import torch
    from vllm import SamplingParams
    from .common import load_model, load_tokenizer
    from src.utils.parse import parse_generation_detailed
    rd = RL / a.run; rd.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(a, k) for k in ("steps", "prompts", "group", "budget", "lr", "init", "seed")} | {"logprob_temperature": DECODE["temperature"]}
    if (rd / "config.json").exists():
        assert json.loads((rd / "config.json").read_text()) == cfg, f"{rd}/config.json differs from these arguments"
    else:
        (rd / "config.json").write_text(json.dumps(cfg, indent=1))
    tok = load_tokenizer()
    tasks = train_tasks(); n = tasks.height
    refs = {k: load_vector(k)["v"] for k in ("ol_math_A2", "sealdir_L17_n6p75", "ol_random") if (VS / f"vec_{k}.npz").exists()}
    dev = torch.device("cuda:0")
    d = len(next(iter(refs.values()))) if refs else 4096
    b = torch.zeros(d, dtype=torch.float32, device=dev)
    if a.init != "zero":
        b = torch.as_tensor(load_vector(a.init)["v"], dtype=torch.float32, device=dev)
    b = torch.nn.Parameter(b)
    opt = torch.optim.Adam([b], lr=a.lr)
    start = 0
    ck = rd / "state.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        with torch.no_grad():
            b.copy_(st["b"])
        opt.load_state_dict(st["opt"]); start = int(st["step"])
        print(f"[rl] resuming at step {start}", flush=True)
    if start >= a.steps:
        print("[rl] done"); return
    llm = build_vllm(max_model_len=a.budget + 2048, gpu_mem=a.vllm_mem, max_num_seqs=128)
    model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    dl = decoder_layers(model)
    def add_b(module, args, output):
        h = _layer_output(output)
        return _replace_output(output, (h.float() + b).to(h.dtype))          # same arithmetic as the vLLM hook
    dl[LAYER].register_forward_hook(add_b)
    norms = [m for name, m in model.named_modules() if name.endswith("language_model.norm")]
    assert norms, "final norm not found"
    grab = {}
    norms[0].register_forward_hook(lambda m, i, o: grab.__setitem__("h", o))
    lm_head = model.lm_head

    # log-probs of the sampling policy: logits divided by the sampling temperature, as TRL and verl do (top-k/top-p
    # truncation ignored, as there). Runs before 2026-09-26 (zero8k, a2init8k) used temperature 1 here: a mismatch.
    T = float(DECODE["temperature"])
    def chunk_lp(h, tgt):
        return torch.log_softmax(lm_head(h).float() / T, -1).gather(-1, tgt[..., None])[..., 0].sum()

    for step in range(start, a.steps):
        t0 = time.time()
        ep, pos = divmod(step * a.prompts, n)
        perm = np.random.default_rng(a.seed + ep).permutation(n)
        idx = [int(perm[(pos + i) % n]) for i in range(a.prompts)]
        batch = tasks[idx]
        p_ids = [tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"] for r in batch.iter_rows(named=True)]
        _set_steer(llm, tok, {"v": b.detach().cpu().numpy(), "layer": LAYER}, "all")
        sp = [SamplingParams(n=a.group, max_tokens=a.budget, seed=int(json_hash([a.run, step, i])[:8], 16), skip_special_tokens=False, **DECODE)
              for i in range(len(p_ids))]
        outs = llm.generate([{"prompt_token_ids": x} for x in p_ids], sp, use_tqdm=False)
        t_gen = time.time() - t0
        rolls = []
        for r, pid, o in zip(batch.iter_rows(named=True), p_ids, outs):
            rs = []
            for g in o.outputs:
                fin = "stop" if g.finish_reason == "stop" else "length"
                ans = parse_generation_detailed(g.text, "reasoning", finish_reason=fin)["answer_text"] or ""
                rew = len(g.token_ids) % 2 if a.smoke_reward else grade(ans, r["reference_answer"], "math", r["prompt"])   # smoke: fake mixed rewards
                rs.append((list(g.token_ids), rew, bool(ans.strip())))
            mean_r = float(np.mean([x[1] for x in rs]))
            for gid, rew, ansd in rs:
                rolls.append({"pid": pid, "gid": gid, "r": rew, "adv": rew - mean_r, "answered": ansd})
        # policy gradient on b (Dr. GRPO); rollouts with zero advantage contribute nothing
        opt.zero_grad(set_to_none=True)
        norm_c = float(a.prompts * a.group * a.budget)
        for x in rolls:
            if x["adv"] == 0 or not x["gid"]:
                continue
            ids = torch.tensor([x["pid"] + x["gid"]], device=dev)
            model(input_ids=ids, logits_to_keep=1, use_cache=False)
            h = grab.pop("h")[0]                                           # [T, d]
            P, T = len(x["pid"]), ids.shape[1]
            tgt = ids[0, P:T]
            hs = h[P - 1:T - 1]
            lp = sum(torch.utils.checkpoint.checkpoint(chunk_lp, hs[s:s + 1024], tgt[s:s + 1024], use_reentrant=False)
                     for s in range(0, hs.shape[0], 1024))
            (-(x["adv"] / norm_c) * lp).backward()
        gnorm = float(b.grad.norm()) if b.grad is not None else 0.0
        if b.grad is not None:
            opt.step()
        bv = b.detach().cpu().numpy()
        m = {"step": step + 1, "reward": float(np.mean([x["r"] for x in rolls])), "answered": float(np.mean([x["answered"] for x in rolls])),
             "tokens": float(np.mean([len(x["gid"]) for x in rolls])),
             "groups_mixed": float(np.mean([len({y["r"] for y in rolls[i * a.group:(i + 1) * a.group]}) > 1 for i in range(len(p_ids))])),
             "grad_norm": gnorm, "b_norm": float(np.linalg.norm(bv)), **{f"cos_{k}": _cos(bv, v) for k, v in refs.items()},
             "t_gen": t_gen, "t_step": time.time() - t0}
        with open(rd / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        np.save(rd / f"b_s{step + 1:03d}.npy", bv)
        torch.save({"b": b.detach(), "opt": opt.state_dict(), "step": step + 1}, rd / "state.tmp.pt")
        (rd / "state.tmp.pt").replace(ck)
        m["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        print("[rl] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)


def cmd_export(run: str, step: int, name: str | None):
    bv = np.load(RL / run / f"b_s{step:03d}.npy")
    name = name or f"rl_{run}_s{step}"
    meta = {"source": "rlbias", "run": run, "step": step, "config": json.loads((RL / run / "config.json").read_text())}
    np.savez(VS / f"vec_{name}.npz", v=bv.astype(np.float32), layer=LAYER, meta=json.dumps(meta))
    print(f"[export] vec_{name}: layer {LAYER}, |v| {np.linalg.norm(bv):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--run", required=True); t.add_argument("--steps", type=int, default=60)
    t.add_argument("--prompts", type=int, default=16); t.add_argument("--group", type=int, default=8); t.add_argument("--budget", type=int, default=8192)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--init", default="zero"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--vllm-mem", type=float, default=0.42); t.add_argument("--smoke-reward", action="store_true")
    e = sub.add_parser("export"); e.add_argument("--run", required=True); e.add_argument("--step", type=int, required=True); e.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.cmd == "train": cmd_train(a)
    else: cmd_export(a.run, a.step, a.name)


if __name__ == "__main__":
    main()
