"""Stage C: steered generation on untouched Math/Idea tasks (single GPU, HF generate + hooks).

A run is a directory data/runs/<run>/ with conditions.json (list of condition specs) and one
traces parquet per condition: traces_<condition>.parquet (resumable per instance x seed).

Condition spec keys:
  name              unique
  mode              baseline | gamma | add | swap | random
  direction         key in directions npz, e.g. prompt_r1 | rawdom_r1 | prompt_r2   (not for baseline/random)
  layer, phase      which fitted direction (phase = center/mu source, e.g. mean32)
  gamma / alpha_sd  gamma for mode=gamma; alpha in units of baseline projection SD for mode=add
  swap_to           "other" (each domain -> the other domain's centroid) for mode=swap
  rank              for random: rank of the random subspace; rms_match: condition name whose mean
                    |Δh| the random perturbation is matched to (computed on baseline acts)
  pulse_k           steered positions (default 64); sustained: bool
  schedule          for mode=add|random (sustained): {"early": alpha_gap, "late": alpha_gap,
                    "switch": {"type": "fixed", "tokens": T} | {"type": "baseline_frac", "frac": f, "min": a, "max": b}}
                    baseline_frac reads the same (instance, seed) trace from traces_baseline.parquet of the run
                    and switches at frac x its n_new_tokens, clipped to [min, max]
  system_prompt     optional strategy prompt (paraphrase index into strategy_prompts.json, e.g. "exploratory:0")
Sampling: shared decode (T=0.6, top_p=0.95, top_k=20), max_new_tokens=window; seed per (batch).
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .common import (DATA, DECODE, GEN_MODEL_KEY, MODEL_ID, Steerer, TASKS_DIR, decoder_layers,
                     json_hash, load_model, load_strategy_prompts, load_tokenizer, render_prompt)
from src.utils.parse import parse_generation_detailed, is_completed  # noqa: E402
from src.utils.characterize import soft_loop_detect, loop_collapse_detect  # noqa: E402


def select_tasks(n_per_domain: int, seed: int, valid_only: bool = True, domains=("math", "idea"), exclude: list | None = None) -> pl.DataFrame:
    """Stratified task sample (math by level, other domains random) among instances with valid baseline labels."""
    from .common import TRACES_BASELINE
    base = pl.read_parquet(TRACES_BASELINE, columns=["instance_id", "task_type", "reasoning_text_for_analysis"]).filter(
        pl.col("reasoning_text_for_analysis").is_not_null() & (pl.col("reasoning_text_for_analysis").str.len_chars() > 20))
    rng = np.random.default_rng(seed)
    picks = []
    for dom in list(domains):
        tasks = pl.read_parquet(TASKS_DIR / f"{dom}.parquet")
        if valid_only:
            tasks = tasks.filter(pl.col("instance_id").is_in(base["instance_id"].to_list()))
        if exclude:
            tasks = tasks.filter(~pl.col("instance_id").is_in(list(exclude)))
        if n_per_domain >= tasks.height:
            picks.append(tasks); continue
        if dom == "math":
            strata = tasks["difficulty_raw"].to_numpy()
            levels = np.unique(strata); per = n_per_domain // len(levels)
            rows = []
            for lv in levels:
                ids = np.where(strata == lv)[0]; rows.extend(rng.choice(ids, size=min(per, len(ids)), replace=False))
            extra = [i for i in range(tasks.height) if i not in set(rows)]
            rows.extend(rng.choice(extra, size=n_per_domain - len(rows), replace=False))
            picks.append(tasks[sorted(int(r) for r in rows)])
        else:
            picks.append(tasks[sorted(rng.choice(tasks.height, size=n_per_domain, replace=False).tolist())])
    return pl.concat(picks)


def _cond_layers(cond: dict) -> list[int]:
    return list(cond["layers"]) if "layers" in cond else [int(cond["layer"])]


def build_steerers(cond: dict, dirs) -> list[tuple[int, Steerer]]:
    """One Steerer per intervened layer. Magnitudes:
    mode gamma : gamma (projection scaling about the training center)
    mode add   : alpha_raw (raw units), alpha_gap (units of the per-layer Idea-minus-Math projection gap),
                 or alpha_sd (units of the pooled within-domain SD of the projection)
    mode random: random orthonormal U; magnitude matched to the add condition with the same alpha_* key
    mode swap  : target = other domain's centroid (set per batch)
    """
    mode = cond["mode"]
    if mode in ("baseline", "prompt_only"):
        return []
    if mode == "target":
        # signature controller: cond = {"map": "<npz under data/sigctrl>", "phi_star": [k] or "archetype:<name>", "gain": g, "lam": l, "mask": [k]}
        z = np.load(DATA / "sigctrl" / cond["map"], allow_pickle=True)
        layer = int(z["layer"]); k = z["W"].shape[1]
        phi = cond["phi_star"]
        if isinstance(phi, str):
            arch = json.loads((DATA / "sigctrl" / "archetypes.json").read_text()); phi = arch[phi]["phi"]
        ro = {"W": z["W"], "b": z["b"], "mu": z["mu"], "sd": z["sd"], "phi_star": np.asarray(phi, np.float32)[:k], "gain": cond.get("gain", 1.0),
              "mask": np.asarray(cond.get("mask", np.ones(k)), np.float32), "cap": cond.get("cap", 0.0),
              "open_loop": cond.get("open_loop", False), "open_norm": cond.get("open_norm", 0.0)}
        if "rel_lam" in cond: ro["rel_lam"] = cond["rel_lam"]
        else: ro["lam"] = cond.get("lam", 1e-3)
        if bool(z["demean_domain"]) if "demean_domain" in z.files else False:
            # the mapping predicts deviations from each domain's mean signature: targets become archetype - domain mean,
            # set per row at batch time (see generate_condition) from the map's stored domain means
            ro["domain_means"] = {str(d): m[:k] for d, m in zip(z["domain_names"].tolist(), z["domain_means"])}
        if cond.get("random_subspace"):
            rng = np.random.default_rng(int(cond.get("seed", 0))); Wr = rng.standard_normal(z["W"].shape).astype(np.float32)
            Wr *= np.linalg.norm(z["W"]) / np.linalg.norm(Wr); ro["W"] = Wr           # same Frobenius norm, random directions
        st = Steerer(np.zeros((z["W"].shape[0], 1), np.float32), np.zeros(z["W"].shape[0], np.float32), "target",
                     pulse_k=cond.get("pulse_k", 64), sustained=cond.get("sustained", True), readout=ro)
        return [(layer, st)]
    ph = cond.get("phase", "mean32"); pk = cond.get("pulse_k", 64); sus = cond.get("sustained", False)
    out = []
    for l in _cond_layers(cond):
        center = dirs[f"center|{l}|{ph}"]
        key = cond.get("direction", "prompt_r1")
        mu = dirs[f"mu|{key}|{l}|{ph}"]                                     # [2, r]: math, idea
        gap = float(mu[1][0] - mu[0][0])
        sd = float(dirs[f"sd|{key}|{l}|{ph}"]) if f"sd|{key}|{l}|{ph}" in dirs.files else None

        def _mag():
            if "alpha_gap" in cond: return float(cond["alpha_gap"]) * gap
            if "alpha_sd" in cond: return float(cond["alpha_sd"]) * sd
            return float(cond["alpha_raw"])
        if mode == "random":
            rng = np.random.default_rng(int(cond.get("seed", 0)) * 1000 + l)
            d = center.shape[0]
            U = np.linalg.qr(rng.standard_normal((d, int(cond.get("rank", 1)))))[0].astype(np.float32)
            if "schedule" in cond:
                sch = cond["schedule"]
                a0 = np.full(U.shape[1], float(sch["early"]) * gap, np.float32) / np.sqrt(U.shape[1])
                a1 = np.full(U.shape[1], float(sch["late"]) * gap, np.float32) / np.sqrt(U.shape[1])
                out.append((l, Steerer(U, center, "add", alpha=a0, alpha_late=a1, pulse_k=pk, sustained=sus)))
                continue
            mag = _mag()
            alpha = np.full(U.shape[1], mag, np.float32) / np.sqrt(U.shape[1])
            out.append((l, Steerer(U, center, "add", alpha=alpha, pulse_k=pk, sustained=sus)))
            continue
        U = dirs[f"{key}|{l}|{ph}"]
        if mode == "gamma":
            out.append((l, Steerer(U, center, "gamma", gamma=float(cond["gamma"]), pulse_k=pk, sustained=sus)))
        elif mode == "add" and "schedule" in cond:
            sch = cond["schedule"]
            a0 = np.zeros(U.shape[1], np.float32); a0[0] = float(sch["early"]) * gap
            a1 = np.zeros(U.shape[1], np.float32); a1[0] = float(sch["late"]) * gap
            out.append((l, Steerer(U, center, "add", alpha=a0, alpha_late=a1, pulse_k=pk, sustained=sus)))
        elif mode == "add":
            alpha = np.zeros(U.shape[1], np.float32)
            alpha[0] = _mag()
            out.append((l, Steerer(U, center, "add", alpha=alpha, pulse_k=pk, sustained=sus)))
        elif mode == "swap":
            st = Steerer(U, center, "swap", target=mu[0], pulse_k=pk, sustained=sus); st.mu = mu
            out.append((l, st))
        else:
            raise ValueError(mode)
    return out


def generate_condition(model, tok, cond: dict, tasks: pl.DataFrame, dirs, *, out_path: Path, window: int,
                       seeds: list[int], batch: int, run_fp: str, max_new_tokens: int | None = None):
    max_new_tokens = int(max_new_tokens or window)
    strategy = load_strategy_prompts()
    system = None
    if cond.get("system_prompt"):
        k, i = cond["system_prompt"].split(":"); system = strategy[k][int(i)]
    layers = decoder_layers(model)
    done = set()
    existing = None
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        done = set(zip(existing["instance_id"].to_list(), existing["seed"].to_list()))
    work = [(r, s) for r in tasks.iter_rows(named=True) for s in seeds if (r["instance_id"], s) not in done]
    print(f"[steer:{cond['name']}] {len(work)} pending of {tasks.height * len(seeds)}", flush=True)
    if not work:
        return
    # group by domain for swap targets; sort by prompt length inside a domain
    work.sort(key=lambda w: (w[0]["task_type"], len(w[0]["prompt"]), w[0]["instance_id"], w[1]))
    steerers = build_steerers(cond, dirs)
    is_swap = bool(steerers) and steerers[0][1].mode == "swap"
    if is_swap:
        # domain-pure chunks (the swap target is the other domain's centroid), nothing dropped
        chunks = []
        for dom_name in sorted({w[0]["task_type"] for w in work}):
            sub = [w for w in work if w[0]["task_type"] == dom_name]
            chunks += [sub[i:i + batch] for i in range(0, len(sub), batch)]
    else:
        chunks = [work[i:i + batch] for i in range(0, len(work), batch)]
    s0 = 0
    sched = cond.get("schedule")
    base_len = None
    if sched:
        sw = sched["switch"]
        if sw["type"] == "baseline_frac":
            bp = out_path.parent / "traces_baseline.parquet"
            if not bp.exists():
                raise FileNotFoundError(f"schedule baseline_frac needs {bp} (run the baseline condition first)")
            bt = pl.read_parquet(bp).select(["instance_id", "seed", "n_new_tokens"])
            base_len = {(i, int(sd)): int(n) for i, sd, n in zip(bt["instance_id"].to_list(), bt["seed"].to_list(), bt["n_new_tokens"].to_list())}
    def _switch_for(w):
        if not sched: return 0
        sw = sched["switch"]
        if sw["type"] == "fixed": return int(sw["tokens"])
        n = base_len.get((w[0]["instance_id"], int(w[1])))
        if n is None: raise KeyError(f"no baseline trace for {w[0]['instance_id']} seed {w[1]}")
        return int(min(max(round(float(sw["frac"]) * n), int(sw.get("min", 1))), int(sw.get("max", 10**9))))
    for chunk in chunks:
        s0 += len(chunk)
        doms = {w[0]["task_type"] for w in chunk}
        switches = [_switch_for(w) for w in chunk]
        if sched:
            for _, st in steerers:
                st.switch = torch.as_tensor(switches, dtype=torch.long)
        for _, st in steerers:
            if st.mode == "target" and st.readout is not None and "domain_means" in st.readout:
                dm = st.readout["domain_means"]; base = st.readout["phi_star_abs"]; ref = cond.get("ref_domain")
                # ref_domain fixes the reference mean (e.g. "math") so the same direction/target is applied to every row,
                # whatever its own domain; otherwise each row's target is relative to its own domain mean
                rows_phi = np.stack([base - np.asarray(dm[ref or w[0]["task_type"]], np.float32) for w in chunk])   # [B, k]
                st.readout["phi_star"] = torch.as_tensor(rows_phi); st.device = None
        if is_swap:
            dom = chunk[0][0]["task_type"]
            for _, st in steerers:
                st.target = torch.as_tensor(st.mu[1 if dom == "math" else 0])   # mu rows [math, idea]: math -> idea centroid, idea -> math
                st.device = None
        rendered = [render_prompt(tok, w[0]["prompt"], system=system) for w in chunk]
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # common random numbers: the torch seed depends on the batch composition, not the condition,
        # so paired conditions share the sampling stream until the intervention makes logits diverge (D-14)
        seed = int(json_hash([run_fp, [w[0]["instance_id"] for w in chunk], [w[1] for w in chunk]])[:8], 16)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        hooks = []
        for l, st in steerers:
            st.reset(); hooks.append(layers[l].register_forward_hook(st))
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, **DECODE,
                                     pad_token_id=tok.pad_token_id)
        finally:
            for h in hooks: h.remove()
        P = enc["input_ids"].shape[1]
        rows = []
        for i, (r, sd) in enumerate(chunk):
            gen = out[i, P:]
            eos_pos = ((gen == tok.eos_token_id) | (gen == tok.pad_token_id)).nonzero()
            n_new = int(eos_pos[0, 0]) + 1 if len(eos_pos) else int(gen.shape[0])
            finish = "stop" if len(eos_pos) else "length"
            text = tok.decode(gen[:n_new], skip_special_tokens=False)
            for t in (tok.eos_token, tok.pad_token):
                if t and text.endswith(t): text = text[: -len(t)]
            parsed = parse_generation_detailed(text, "reasoning", finish_reason=finish)
            think, answer = parsed["think_text"], parsed["answer_text"]
            has_close = parsed["close_tag_count"] > 0
            completed = is_completed(kind="reasoning", finish_reason=finish, has_close_tag=has_close or parsed["parse_status"] == "direct_answer_no_close", has_answer=bool(answer))
            # prefix-window phenotype: the judge sees only the first `window` reasoning tokens (bounds judge
            # cost); the full think block stays in think_text and the answer is graded from the full generation
            rtext = None
            if think:
                tids = tok(think, add_special_tokens=False)["input_ids"]
                rtext = think if len(tids) <= window else tok.decode(tids[:window])
            rows.append({
                "trace_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "\0".join([run_fp, cond["name"], r["instance_id"], str(sd), text]))),
                "run_fingerprint": run_fp, "condition": cond["name"], "condition_spec": json.dumps(cond, sort_keys=True),
                "gen_model": GEN_MODEL_KEY, "gen_model_id": MODEL_ID, "generation_kind": "reasoning",
                "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": int(sd),
                "difficulty_raw": r.get("difficulty_raw"), "reference_answer": r.get("reference_answer"),
                "instance_metadata": r.get("metadata", "{}"), "prompt": r["prompt"], "system_prompt": system,
                "full_text": text, "think_text": think, "answer_text": answer, "reasoning_text_for_analysis": rtext,
                "n_new_tokens": n_new, "completed": completed, "finish_reason": finish,
                "parse_status": parsed["parse_status"], "close_tag_count": parsed["close_tag_count"],
                "hard_loop": loop_collapse_detect(text), "soft_loop": soft_loop_detect(text),
                "steered_positions": (steerers[0][1].n_steered_positions if steerers else 0),
                "switch_at": (switches[i] if sched else None),
                "late_positions": (steerers[0][1].n_late_positions if (steerers and sched) else None),
                "steered_layers": json.dumps([l for l, _ in steerers]),
                "decode": json.dumps({**DECODE, "max_new_tokens": max_new_tokens, "phenotype_window": window, "batch_seed": seed}),
            })
        new = pl.DataFrame(rows)
        existing = pl.concat([existing, new], how="diagonal_relaxed") if existing is not None else new
        tmp = out_path.with_suffix(".parquet.tmp"); existing.write_parquet(tmp); tmp.replace(out_path)
        dt = time.time() - t0
        print(f"[steer:{cond['name']}] {s0}/{len(work)} batch {len(chunk)} x {out.shape[1] - P} steps "
              f"in {dt:.0f}s ({sum(x['n_new_tokens'] for x in rows) / dt:.0f} tok/s) doms={sorted(doms)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under data/runs/")
    ap.add_argument("--conditions", default=None, help="comma list of condition names (default all)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    rd = DATA / "runs" / a.run
    spec = json.loads((rd / "run.json").read_text())
    conds = spec["conditions"]
    if a.conditions:
        want = set(a.conditions.split(",")); conds = [c for c in conds if c["name"] in want]
    tasks = select_tasks(spec["n_per_domain"], spec["task_seed"], domains=spec.get("domains", ["math", "idea"]), exclude=spec.get("exclude_instances"))
    tasks.write_parquet(rd / "tasks.parquet")
    dirs = np.load(DATA / "directions" / spec["directions"], allow_pickle=True) if (DATA / "directions" / spec["directions"]).exists() else None
    run_fp = json_hash({k: spec[k] for k in ["n_per_domain", "task_seed", "window", "seeds", "directions"]})
    tok = load_tokenizer(); model = load_model()
    for c in conds:
        generate_condition(model, tok, c, tasks, dirs, out_path=rd / f"traces_{c['name']}.parquet",
                           window=spec["window"], seeds=spec["seeds"], batch=a.batch, run_fp=run_fp,
                           max_new_tokens=spec.get("max_new_tokens"))


if __name__ == "__main__":
    main()
