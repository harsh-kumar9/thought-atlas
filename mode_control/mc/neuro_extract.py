"""Neuro-style readouts (D-55 to D-58): every component's write onto the mode directions, per token.

The residual stream is a sum (pre-norm): h_L = embed + sum_{l <= L} (attn_l + mlp_l). Each attention output is
o_proj (full attention, 16 heads x 256) or out_proj (linear attention, 32 heads x 128) applied to the concatenated,
gated head outputs, so it is itself a sum over heads. The projection of h_L onto a direction u therefore decomposes
exactly into per-component terms <write_c, u>. Components: the token embedding, every head of layers 0-19, every
MLP block of layers 0-19 (581 components). Layers above 19 write after both readout layers and are skipped.

Directions (k = 11, data/neuro/directions.json):
  dom16, dom19   prompt_r1 at layer 16 / 19, phase mean32, unit, + = Idea-like (directions_w8192_v2w8192.npz)
  sig16_0..7     the 8 left singular vectors of the rank-8 readout operator R = diag(1/sd) W at layer 16
                 (map_corpus1_dm8_L16_r8.npz); phi_hat - b = (h - mu) @ R, so (w @ U8) S V^T is w's readout share
  a2_16          the A2 vector (vec_ol_math_A2.npz), unit

Modes
  corpus  --runs r1,r2   baseline traces, prompt + first N reasoning tokens (N = --n-tokens, 2048)
  prompts --run r --conditions c1,c2   the same for other conditions (power1 prompt arms)
  taskset  the instruction-delay prompts (demand x operation x delay x topic x item): generate 192 tokens, then
           teacher-force the whole sequence; spans (system, filler, task, generated) are stored per prompt
  smoke   four corpus traces and the sum-identity check (sum of component writes = residual projection)
  states  --runs r --conditions c1,c2: the layer-16 residual every --stride tokens of each trace (prompt + first
          --n-tokens reasoning tokens), for the steered-state density test (D-62) -> data/neuro/states/<run>__<cond>.npz

Per trace the shard holds: comp_mean [C, k] (mean write over the window positions), comp_act [C] (RMS of the
8-dim readout write), resid [T_win, 2, k] (residual projections at layers 16 and 19, every window position) and,
when series are requested, series [T_win, C] float16 (every component's dom19 write per position). For taskset
the window is the whole sequence and comp_mean / comp_act are given for the pre-task span and the generated span.
Shards: data/neuro/<mode>/<name>__s<k>.npz, flushed every --flush traces; a rerun skips stored trace ids.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .common import (DATA, DECODE, MC, _layer_output, decoder_layers, json_hash, load_model,
                     load_strategy_prompts, load_tokenizer, render_prompt)

NEURO = DATA / "neuro"
MAX_LAYER = 19
READOUT_LAYERS = (16, 19)
DIR_NAMES = ["dom16", "dom19"] + [f"sig16_{j}" for j in range(8)] + ["a2_16"]
K = len(DIR_NAMES)

# ---------------------------------------------------------------- task-set prompts (E2)
DEMAND_SYS = {
    "solve": "The next message gives you a problem. Your job is to solve it and state the final answer.",
    "generate": "The next message gives you a problem. Your job is not to solve it. Your job is to list genuinely "
                "different ways someone could approach it.",
}
PTYPE = {"solve": "solve_free", "generate": "generate"}
DELAYS = (0, 64, 256)
OPS = ("none", "checking", "exploratory")
GEN_TOKENS = 192
# Neutral filler for the delay: one passage, cut to the delay length in tokens. No numbers, questions or instructions.
FILLER = (
    "The morning in the harbour town begins slowly. Mist sits on the water until the sun has cleared the headland, "
    "and the boats at the quay rock a little on the tide without going anywhere. A baker opens the shutters of the "
    "shop on the corner and props the door with a crate. The smell of bread drifts along the street and meets the "
    "smell of salt coming the other way. Gulls stand along the roof of the fish market, each facing the same "
    "direction, and now and then one lifts off, circles, and settles again in the same spot. The cobbles are still "
    "dark with dew. An old dog walks the length of the quay, stops at the bollards one by one, and walks back. "
    "Across the water the hills are green and brown, with pale lines where the paths cut through the heather. A "
    "ferry waits at the far pier with its ramp down and nobody on it yet. In the square the café puts out its chairs, "
    "and the chairs scrape, and then it is quiet again. The church clock is behind the trees and cannot be seen from "
    "the harbour, but its bell carries well over the water on a still morning. Window boxes along the terrace are "
    "full of geraniums, red and pink, and someone has already been out with a watering can because the stone below "
    "them is wet. Two fishermen sit on upturned buckets mending a net, talking in low voices, their hands moving "
    "without looking. The net is spread across the slipway and the mesh throws a faint shadow on the concrete. "
    "Further along, a painter has set up an easel facing the lighthouse and is squeezing colours onto a board. The "
    "lighthouse itself is white with a red band, and the door at its base is painted the same red. A cat sleeps on "
    "the warm lid of a lobster pot. The tide is turning now, and the water near the steps has started to move, "
    "carrying a line of weed and foam out toward the mouth of the harbour. The sun reaches the quay, and the mist "
    "thins, and the colours of the boats come up one by one: blue, white, a faded green, and a yellow hull that "
    "someone has half repainted. The baker's door swings shut, then opens again. Up on the terrace a window opens "
    "and a curtain moves in the breeze. The day has begun, and it will go on like this for a while yet."
)


def build_directions():
    """[d, K] float32 and the metadata needed to turn readout-space writes back into behaviour targets."""
    dirs = np.load(DATA / "directions" / "directions_w8192_v2w8192.npz", allow_pickle=True)
    d16 = dirs["prompt_r1|16|mean32"][:, 0].astype(np.float32); d19 = dirs["prompt_r1|19|mean32"][:, 0].astype(np.float32)
    z = np.load(DATA / "sigctrl" / "map_corpus1_dm8_L16_r8.npz", allow_pickle=True)
    R = (z["W"].astype(np.float64) / z["sd"].astype(np.float64)[:, None])          # [d, 16], rank 8
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    U8 = U[:, :8]
    for j in range(8):                                                             # sign: + = more of the dominant target
        if (U8[:, j] @ R[:, int(np.argmax(np.abs(Vt[j])))]) < 0:
            U8[:, j] *= -1
    a2 = np.load(DATA / "vsteer" / "vec_ol_math_A2.npz", allow_pickle=True)["v"].astype(np.float32)
    a2 = a2 / np.linalg.norm(a2)
    D = np.concatenate([d16[:, None], d19[:, None], U8.astype(np.float32), a2[:, None]], axis=1).astype(np.float32)
    # readout of a point h: phi_hat = (h - mu) @ R + b = ((h - mu) @ U8) @ (S8 Vt8) + b
    proj_U = U8.T @ R                                                              # [8, 16] = S8 Vt8 (with the sign flips)
    meta = {"names": DIR_NAMES, "readout_layer": 16, "axis_layers": [16, 19], "U8_to_targets": proj_U.tolist(),
            "readout_b": z["b"].tolist(), "readout_mu_proj": (z["mu"].astype(np.float64) @ D.astype(np.float64)).tolist(),
            "target_names": z["target_names"].tolist(), "target_sd": z["target_sd"].tolist(),
            "domain_names": z["domain_names"].tolist(), "domain_means": np.asarray(z["domain_means"]).tolist(),
            "sig16_singular_values": S[:8].tolist(), "cos_dom16_dom19": float(d16 @ d19), "cos_dom16_a2": float(d16 @ a2),
            "a2_norm_in_runs": 6.75, "directions_file": "directions_w8192_v2w8192.npz", "map_file": "map_corpus1_dm8_L16_r8.npz"}
    return D, meta


# ---------------------------------------------------------------- component hooks
class CompHooks:
    """Fills P[B, T, C, K] (float32, device) with every component's write projected onto D during one forward."""

    def __init__(self, model, D: np.ndarray):
        import torch
        self.layers = decoder_layers(model)
        self.device = model.device
        self.D = torch.as_tensor(D, dtype=torch.float32, device=self.device)        # [d, K]
        self.comps: list[dict] = [{"layer": -1, "kind": "embed", "head": -1}]
        self.handles = []
        self.P = None
        self.R = {}                                                                  # layer -> [B, T, K] residual projections
        embs = sorted([(n, m) for n, m in model.named_modules() if n.endswith("embed_tokens")], key=lambda x: "language_model" not in x[0])
        self.handles.append(embs[0][1].register_forward_hook(self._emb_hook))
        for l in range(MAX_LAYER + 1):
            lay = self.layers[l]
            if hasattr(lay, "self_attn"):
                proj, nh = lay.self_attn.o_proj, int(lay.self_attn.o_proj.in_features // lay.self_attn.head_dim)
                kind = "attn"
            else:
                proj, nh = lay.linear_attn.out_proj, int(lay.linear_attn.num_v_heads)
                kind = "lin"
            hd = proj.in_features // nh
            M = (proj.weight.detach().float().T @ self.D).view(nh, hd, K).contiguous()   # W_O[:, head]^T D  -> [nh, hd, K]
            c0 = len(self.comps)
            self.comps += [{"layer": l, "kind": kind, "head": h} for h in range(nh)]
            self.handles.append(proj.register_forward_pre_hook(self._attn_hook(M, nh, hd, c0)))
            cm = len(self.comps); self.comps.append({"layer": l, "kind": "mlp", "head": -1})
            self.handles.append(lay.mlp.register_forward_hook(self._mlp_hook(cm)))
        for l in READOUT_LAYERS:
            self.handles.append(self.layers[l].register_forward_hook(self._resid_hook(l)))
        self.C = len(self.comps)
        self.layer_of = np.array([c["layer"] for c in self.comps])

    def begin(self, B: int, T: int):
        import torch
        self.P = torch.zeros((B, T, self.C, K), dtype=torch.float32, device=self.device)
        self.R = {}

    # Every hook is a no-op while self.P is None (e.g. during model.generate in taskset mode); begin() arms them.
    def _emb_hook(self, module, args, output):
        if self.P is None: return
        self.P[:, :, 0, :] = output.float() @ self.D

    def _attn_hook(self, M, nh, hd, c0):
        import torch
        def f(module, args):
            if self.P is None: return
            x = args[0].float()                                                     # [B, T, nh*hd], gated head outputs
            B, T = x.shape[:2]
            self.P[:, :, c0:c0 + nh, :] = torch.einsum("bthd,hdk->bthk", x.view(B, T, nh, hd), M)
        return f

    def _mlp_hook(self, c):
        def f(module, args, output):
            if self.P is None: return
            self.P[:, :, c, :] = output.float() @ self.D
        return f

    def _resid_hook(self, l):
        def f(module, args, output):
            if self.P is None: return
            self.R[l] = _layer_output(output).float() @ self.D
        return f

    def remove(self):
        for h in self.handles: h.remove()


def _pad_batch(tok, ids: list[list[int]]):
    import torch
    T = max(len(x) for x in ids); pad = tok.pad_token_id
    inp = torch.full((len(ids), T), pad, dtype=torch.long); att = torch.zeros((len(ids), T), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, T - len(x):] = torch.tensor(x); att[i, T - len(x):] = 1
    return inp, att, [T - len(x) for x in ids]


def _forward(model, hooks: CompHooks, inp, att):
    import torch
    hooks.begin(*inp.shape)
    with torch.no_grad():
        model(input_ids=inp.to(model.device), attention_mask=att.to(model.device), use_cache=False)
    P, R = hooks.P, hooks.R
    hooks.P = None                                                                  # disarm until the next begin()
    return P, R


# ---------------------------------------------------------------- shard writer
class Shards:
    def __init__(self, out_dir: Path, name: str, flush: int):
        self.dir = out_dir; self.name = name; self.flush_n = flush
        out_dir.mkdir(parents=True, exist_ok=True)
        self.done = set()
        self.k = 0
        for p in sorted(out_dir.glob(f"{name}__s*.npz")):
            z = np.load(p, allow_pickle=True); self.done |= set(z["trace_id"].tolist()); self.k += 1
        self.rows = []

    def add(self, row: dict):
        self.rows.append(row)
        if len(self.rows) >= self.flush_n:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        keys = self.rows[0].keys()
        out = {}
        for key in keys:
            vals = [r[key] for r in self.rows]
            if key in ("resid", "series", "gen_ids"):                         # variable length: concatenate + offsets
                out[key] = np.concatenate(vals, axis=0)
                out[key + "_off"] = np.concatenate([[0], np.cumsum([len(v) for v in vals])]).astype(np.int64)
            elif isinstance(vals[0], np.ndarray):
                out[key] = np.stack(vals)
            else:
                out[key] = np.array(vals)
        p = self.dir / f"{self.name}__s{self.k:03d}.npz"
        np.savez(p.with_suffix(".tmp.npz"), **out); p.with_suffix(".tmp.npz").replace(p)
        self.done |= {r["trace_id"] for r in self.rows}
        print(f"[neuro] wrote {p.name} ({len(self.rows)} traces)", flush=True)
        self.rows = []; self.k += 1


def _write_meta(hooks: CompHooks, dmeta: dict):
    NEURO.mkdir(parents=True, exist_ok=True)
    (NEURO / "components.json").write_text(json.dumps(hooks.comps))
    (NEURO / "directions.json").write_text(json.dumps(dmeta))


def _summaries(P, R, b: int, sl: slice, series: bool):
    """Window summaries for one sequence: comp_mean [C, K], comp_act [C], resid [T, 2, K], series [T, C] (dom19)."""
    Pw = P[b, sl]                                                                    # [T, C, K]
    comp_mean = Pw.mean(0).cpu().numpy().astype(np.float32)
    comp_act = Pw[:, :, 2:10].norm(dim=-1).pow(2).mean(0).sqrt().cpu().numpy().astype(np.float32)
    resid = np.stack([R[l][b, sl].cpu().numpy() for l in READOUT_LAYERS], axis=1).astype(np.float16)
    ser = Pw[:, :, 1].cpu().numpy().astype(np.float16) if series else np.zeros((0, Pw.shape[1]), np.float16)
    return comp_mean, comp_act, resid, ser


# ---------------------------------------------------------------- corpus / prompts
def _load_traces(run: str, conditions: list[str] | None) -> pl.DataFrame:
    rd = DATA / "runs" / run
    tr = pl.concat([pl.read_parquet(p) for p in sorted(rd.glob("traces_*.parquet"))], how="diagonal_relaxed")
    if conditions:
        tr = tr.filter(pl.col("condition").is_in(conditions))
    else:
        tr = tr.filter(pl.col("condition") == "baseline")
    return tr.filter(pl.col("think_text").is_not_null() & (pl.col("think_text").str.len_chars() > 20))


def run_traces(mode: str, model, tok, hooks: CompHooks, runs: list[str], conditions: list[str] | None, n_tokens: int,
               batch: int, flush: int, series_seeds: list[int] | None, limit: int | None):
    out_dir = NEURO / mode
    for run in runs:
        tr = _load_traces(run, conditions)
        if limit: tr = tr.head(limit)
        name = run if not conditions else f"{run}__{'+'.join(conditions)}"
        sh = Shards(out_dir, name, flush)
        rows = [r for r in tr.iter_rows(named=True) if r["trace_id"] not in sh.done]
        print(f"[neuro:{mode}] {run}: {len(rows)} traces pending of {tr.height}", flush=True)
        items = []
        for r in rows:
            sysm = r.get("system_prompt")
            p = tok(render_prompt(tok, r["prompt"], system=sysm), add_special_tokens=False)["input_ids"]
            t = tok(r["think_text"], add_special_tokens=False)["input_ids"][:n_tokens]
            if len(t) < 16:
                continue
            items.append((r, p, t))
        items.sort(key=lambda x: len(x[1]) + len(x[2]))
        t0 = time.time()
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            inp, att, off = _pad_batch(tok, [p + t for _, p, t in chunk])
            P, R = _forward(model, hooks, inp, att)
            for b, (r, p, t) in enumerate(chunk):
                sl = slice(off[b] + len(p), off[b] + len(p) + len(t))
                ser = series_seeds is None or int(r["seed"]) in series_seeds
                cm, ca, rs, se = _summaries(P, R, b, sl, ser)
                sh.add({"trace_id": r["trace_id"], "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": int(r["seed"]),
                        "condition": r["condition"], "run": run, "n_prompt": len(p), "n_win": len(t),
                        "comp_mean": cm, "comp_act": ca, "resid": rs, "series": se})
            if (i // batch) % 10 == 0:
                print(f"[neuro:{mode}] {run}: {min(i + batch, len(items))}/{len(items)} traces, {time.time() - t0:.0f}s", flush=True)
        sh.flush()


# ---------------------------------------------------------------- task set (E2)
# Varied fillers (D-56 amendment): with one fixed filler every item in a condition sees identical pre-task text, so the
# delay activations are deterministic and carry no item variance. One filler per item: a seeded sample of 28 of these
# neutral sentences in random order, cut to the delay length. Sentences: plain description, no numbers, no questions.
FILLER_POOL = [
    "The morning in the harbour town begins slowly.", "Mist sits on the water until the sun has cleared the headland.",
    "The boats at the quay rock a little on the tide without going anywhere.", "A baker opens the shutters of the shop on the corner and props the door with a crate.",
    "The smell of bread drifts along the street and meets the smell of salt coming the other way.", "Gulls stand along the roof of the fish market, each facing the same direction.",
    "Now and then one lifts off, circles, and settles again in the same spot.", "The cobbles are still dark with dew.",
    "An old dog walks the length of the quay, stops at the bollards one by one, and walks back.", "Across the water the hills are green and brown, with pale lines where the paths cut through the heather.",
    "A ferry waits at the far pier with its ramp down and nobody on it yet.", "In the square the cafe puts out its chairs, and the chairs scrape, and then it is quiet again.",
    "The church clock is behind the trees and cannot be seen from the harbour.", "Its bell carries well over the water on a still morning.",
    "Window boxes along the terrace are full of geraniums, red and pink.", "Someone has already been out with a watering can because the stone below them is wet.",
    "Two fishermen sit on upturned buckets mending a net, talking in low voices.", "Their hands move without looking.",
    "The net is spread across the slipway and the mesh throws a faint shadow on the concrete.", "Further along, a painter has set up an easel facing the lighthouse.",
    "The lighthouse itself is white with a red band, and the door at its base is painted the same red.", "A cat sleeps on the warm lid of a lobster pot.",
    "The tide is turning now, and the water near the steps has started to move.", "It carries a line of weed and foam out toward the mouth of the harbour.",
    "The sun reaches the quay, and the mist thins.", "The colours of the boats come up one by one: blue, white, a faded green, and a yellow hull that someone has half repainted.",
    "The baker's door swings shut, then opens again.", "Up on the terrace a window opens and a curtain moves in the breeze.",
    "A delivery van backs slowly into the lane behind the shops.", "The driver gets out, stretches, and lifts the rear door.",
    "Crates of vegetables go in through a side entrance, and the van pulls away.", "A woman in a grey coat walks a bicycle along the front rather than riding it.",
    "She stops at the railing and looks at the water for a while.", "The ice cream kiosk is still shuttered, its sign turned to the wall.",
    "Along the sea wall the benches face outward, their paint worn pale by the weather.", "A small boy drags a stick along the railings and the sound follows him down the street.",
    "The newsagent stacks the papers outside under a stone to hold them down.", "Inside the chandlery the lights are on and coils of rope hang from the ceiling.",
    "The harbour master's office has its door open and a radio playing softly.", "A row of lobster pots stands drying against the wall of the fish market.",
    "The path up to the headland starts between two cottages and climbs steeply.", "From the top the whole bay can be seen, with the town small below.",
    "Sheep graze on the slope and do not look up when people pass.", "The wind is stronger on the headland and carries the smell of the sea.",
    "Back in the town the cafe has its first customers, sitting outside with their coats on.", "A waiter brings out cups on a tray and sets them down one at a time.",
    "The bus from the next town arrives and three people get off.", "They stand for a moment looking around, then walk toward the harbour.",
    "The post office opens its door and a queue forms almost at once.", "The hardware shop puts a rack of brooms and buckets out on the pavement.",
    "On the slipway a man hoses down the deck of a small boat and the water runs back into the sea.", "A heron stands in the shallows at the edge of the harbour, perfectly still.",
    "The school bell rings somewhere up the hill and the street goes quiet again.", "Clouds move in from the west but the sun stays out over the water.",
    "The painter steps back from the easel, looks, and steps forward again.", "The old dog has found a patch of sun by the harbour wall and lies down in it.",
    "A fishing boat comes in past the lighthouse, low in the water, with gulls following.", "It ties up at the far end of the quay and the crew start unloading boxes.",
    "The tide keeps going out and the mud at the head of the harbour begins to show.", "The day has begun, and it will go on like this for a while yet.",
]


def item_filler(tok, item: str, delay: int) -> str:
    """One filler per item: 28 sentences sampled and ordered by a seed from the item id, cut to `delay` tokens."""
    if delay <= 0:
        return ""
    rng = np.random.default_rng(int(json_hash(["filler", item])[:8], 16))
    sents = [FILLER_POOL[i] for i in rng.choice(len(FILLER_POOL), 28, replace=False)]
    ids = tok(" ".join(sents), add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:delay]).strip()


def taskset_prompts(tok, topics=("math", "gpqa"), limit: int | None = None, varied: bool = False) -> list[dict]:
    strat = load_strategy_prompts()
    f1 = pl.concat([pl.read_parquet(MC / "data" / "f1" / "tasks" / f"{t}.parquet").with_columns(pl.col("difficulty_raw").cast(pl.Utf8)) for t in topics])
    meta = [json.loads(m) for m in f1["metadata"].to_list()]
    f1 = f1.with_columns(pl.Series("item", [m["f1_item"] for m in meta]), pl.Series("ptype", [m["ptype"] for m in meta]),
                         pl.Series("topic", [m["topic"] for m in meta]))
    f1 = f1.filter(pl.col("ptype").is_in(list(PTYPE.values())))
    fill_ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    out = []
    for r in f1.iter_rows(named=True):
        demand = next(k for k, v in PTYPE.items() if v == r["ptype"])
        for op in OPS:
            sysm = DEMAND_SYS[demand] + ("" if op == "none" else " " + strat[op][0])
            for delay in DELAYS:
                filler = item_filler(tok, r["item"], delay) if varied else (tok.decode(fill_ids[:delay]).strip() if delay else "")
                user = (filler + "\n\n" if filler else "") + r["prompt"]
                out.append({"trace_id": f"{r['item']}|{demand}|{op}|{delay}", "item": r["item"], "topic": r["topic"], "demand": demand,
                            "op": op, "delay": delay, "system": sysm, "filler": filler, "task": r["prompt"], "user": user,
                            "reference_answer": r.get("reference_answer")})
    if limit:
        out = out[:limit]
    return out


def _spans(tok, rendered: str, parts: dict[str, str]):
    """Token index ranges [a, b) of each named substring of the rendered prompt."""
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([o[0] for o in enc["offset_mapping"]]); ends = np.array([o[1] for o in enc["offset_mapping"]])
    spans = {}
    pos = 0
    for name, text in parts.items():
        if not text:
            spans[name] = (-1, -1); continue
        c0 = rendered.find(text, pos); assert c0 >= 0, name
        c1 = c0 + len(text); pos = c1
        idx = np.where((ends > c0) & (starts < c1))[0]
        spans[name] = (int(idx[0]), int(idx[-1]) + 1)
    return enc["input_ids"], spans


def run_taskset(model, tok, hooks: CompHooks, batch_gen: int, batch: int, flush: int, limit: int | None, topics: tuple, varied: bool = False):
    import torch
    out_dir = NEURO / ("taskset_varied" if varied else "taskset")
    prompts = taskset_prompts(tok, topics, limit, varied=varied)
    sh = Shards(out_dir, "taskset", flush)
    pending = [p for p in prompts if p["trace_id"] not in sh.done]
    print(f"[neuro:taskset] {len(pending)} prompts pending of {len(prompts)}", flush=True)
    for p in pending:
        p["rendered"] = render_prompt(tok, p["user"], system=p["system"])
        p["ids"], p["spans"] = _spans(tok, p["rendered"], {"system": p["system"], "filler": p["filler"], "task": p["task"]})
    pending.sort(key=lambda p: len(p["ids"]))
    t0 = time.time()
    for i in range(0, len(pending), batch_gen):
        chunk = pending[i:i + batch_gen]
        inp, att, off = _pad_batch(tok, [p["ids"] for p in chunk])
        seed = int(json_hash(["taskset", [p["trace_id"] for p in chunk]])[:8], 16)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            gen = model.generate(input_ids=inp.to(model.device), attention_mask=att.to(model.device), max_new_tokens=GEN_TOKENS,
                                 do_sample=True, **DECODE, pad_token_id=tok.pad_token_id)
        Pn = inp.shape[1]
        for b, p in enumerate(chunk):
            g = gen[b, Pn:].tolist()
            stop = [j for j, x in enumerate(g) if x in (tok.eos_token_id, tok.pad_token_id)]
            p["gen"] = g[:stop[0]] if stop else g
        # teacher-force prompt + generated tokens in smaller batches and store every position
        for j in range(0, len(chunk), batch):
            sub = [p for p in chunk[j:j + batch] if len(p["gen"]) >= 8]
            if not sub:
                continue
            inp2, att2, off2 = _pad_batch(tok, [p["ids"] + p["gen"] for p in sub])
            P, R = _forward(model, hooks, inp2, att2)
            for b, p in enumerate(sub):
                n = len(p["ids"]) + len(p["gen"]); sl = slice(off2[b], off2[b] + n)
                cm, ca, rs, se = _summaries(P, R, b, sl, True)
                sp = p["spans"]; pre = slice(off2[b] + sp["system"][0], off2[b] + sp["task"][0]); gs = slice(off2[b] + len(p["ids"]), off2[b] + n)
                cm_pre, ca_pre, _, _ = _summaries(P, R, b, pre, False)
                cm_gen, ca_gen, _, _ = _summaries(P, R, b, gs, False)
                sh.add({"trace_id": p["trace_id"], "item": p["item"], "topic": p["topic"], "demand": p["demand"], "op": p["op"], "delay": int(p["delay"]),
                        "n_prompt": len(p["ids"]), "n_gen": len(p["gen"]), "n_win": n,
                        "span_system": np.array(sp["system"]), "span_filler": np.array(sp["filler"]), "span_task": np.array(sp["task"]),
                        "gen_text": tok.decode(p["gen"]), "gen_ids": np.array(p["gen"], np.int32),
                        "comp_mean": cm, "comp_act": ca, "comp_mean_pre": cm_pre, "comp_act_pre": ca_pre, "comp_mean_gen": cm_gen, "comp_act_gen": ca_gen,
                        "resid": rs, "series": se})
        print(f"[neuro:taskset] {min(i + batch_gen, len(pending))}/{len(pending)} prompts, {time.time() - t0:.0f}s", flush=True)
    sh.flush()


# ---------------------------------------------------------------- D-62: strided residual states of (steered) traces
def run_states(model, tok, runs: list[str], conditions: list[str], stride: int, n_tokens: int, batch: int, limit: int | None, layer: int = 16):
    import torch
    out_dir = NEURO / "states"; out_dir.mkdir(parents=True, exist_ok=True)
    dl = decoder_layers(model); store = {}
    h = dl[layer].register_forward_hook(lambda m, a, o: store.__setitem__("h", _layer_output(o).detach()))
    try:
        for run in runs:
            for cond in conditions:
                out = out_dir / f"{run}__{cond}.npz"
                if out.exists():
                    print(f"[states] {run}/{cond}: exists, skipping"); continue
                if not (DATA / "runs" / run / f"traces_{cond}.parquet").exists():
                    print(f"[states] {run}/{cond}: no traces file yet, skipping"); continue
                if (DATA / "runs" / run / f"partial_{cond}.parquet").exists():
                    print(f"[states] {run}/{cond}: generation still in progress (partial file present), skipping"); continue
                tr = _load_traces(run, [cond])
                if limit: tr = tr.head(limit)
                items = []
                for r in tr.iter_rows(named=True):
                    p_ = tok(render_prompt(tok, r["prompt"], system=r.get("system_prompt")), add_special_tokens=False)["input_ids"]
                    t_ = tok(r["think_text"], add_special_tokens=False)["input_ids"][:n_tokens]
                    if len(t_) >= stride:
                        items.append((r, p_, t_))
                items.sort(key=lambda x: len(x[1]) + len(x[2]))
                H, off, meta = [], [0], []
                t0 = time.time()
                for i in range(0, len(items), batch):
                    chunk = items[i:i + batch]
                    inp, att, pad = _pad_batch(tok, [p_ + t_ for _, p_, t_ in chunk])
                    with torch.no_grad():
                        model(input_ids=inp.to(model.device), attention_mask=att.to(model.device), use_cache=False)
                    for b, (r, p_, t_) in enumerate(chunk):
                        pos = list(range(pad[b] + len(p_), pad[b] + len(p_) + len(t_), stride))
                        H.append(store["h"][b, pos, :].float().cpu().numpy().astype(np.float16)); off.append(off[-1] + len(pos))
                        meta.append({"trace_id": r["trace_id"], "condition": cond, "run": run, "instance_id": r["instance_id"], "task_type": r["task_type"], "seed": int(r["seed"]),
                                     "hard_loop": int(bool(r.get("hard_loop"))), "soft_loop": int(bool(r.get("soft_loop"))), "answered": int(bool((r.get("answer_text") or "").strip())),
                                     "n_new_tokens": int(r["n_new_tokens"]), "finish_reason": r.get("finish_reason") or ""})
                    if (i // batch) % 20 == 0:
                        print(f"[states] {run}/{cond}: {min(i + batch, len(items))}/{len(items)} traces, {time.time() - t0:.0f}s", flush=True)
                if not H:
                    continue
                cols = {k: np.array([m[k] for m in meta]) for k in meta[0]}
                np.savez(out, H=np.concatenate(H), off=np.array(off, np.int64), layer=layer, stride=stride, **cols)
                print(f"[states] {run}/{cond}: {len(meta)} traces, {off[-1]} states -> {out.name}", flush=True)
    finally:
        h.remove()


# ---------------------------------------------------------------- smoke
def smoke(model, tok, hooks: CompHooks, D):
    import torch
    tr = _load_traces("corpus1_math_a", None).head(2)
    tr = pl.concat([tr, _load_traces("corpus1_idea_a", None).head(2)], how="diagonal_relaxed")
    items = []
    for r in tr.iter_rows(named=True):
        p = tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        t = tok(r["think_text"], add_special_tokens=False)["input_ids"][:512]
        items.append((r, p, t))
    inp, att, off = _pad_batch(tok, [p + t for _, p, t in items])
    t0 = time.time(); P, R = _forward(model, hooks, inp, att); dt = time.time() - t0
    print(f"[smoke] forward {tuple(inp.shape)} with hooks in {dt:.1f}s; P {tuple(P.shape)} ({P.numel() * 4 / 1e6:.0f} MB)")
    lo = hooks.layer_of
    for l in READOUT_LAYERS:
        m = torch.as_tensor(lo <= l, device=P.device)
        S = P[:, :, m, :].sum(2)                                                    # [B, T, K]
        for b, (r, p, t) in enumerate(items):
            sl = slice(off[b] + len(p), off[b] + len(p) + len(t))
            a, c = S[b, sl], R[l][b, sl]
            err = (a - c).abs().mean().item(); scale = c.abs().mean().item()
            print(f"[smoke] L{l} {r['task_type']:5s} sum-identity |err| {err:.4f} vs |resid proj| {scale:.3f} -> rel {err / scale:.4f}")
    j = DIR_NAMES.index("dom19")
    for b, (r, p, t) in enumerate(items):
        sl = slice(off[b] + len(p), off[b] + len(p) + len(t))
        print(f"[smoke] {r['task_type']:5s} mean dom19 proj over {len(t)} tokens: {R[19][b, sl, j].mean().item():+.3f}; "
              f"top components: " + ", ".join(f"L{hooks.comps[c]['layer']}{hooks.comps[c]['kind']}{hooks.comps[c]['head']}={P[b, sl, c, j].mean().item():+.2f}"
                                               for c in P[b, sl, :, j].mean(0).abs().topk(5).indices.tolist()))
    # task-set prompt rendering and spans
    ps = taskset_prompts(tok, ("math",), limit=18)
    p = ps[-1]; ids, sp = _spans(tok, render_prompt(tok, p["user"], system=p["system"]), {"system": p["system"], "filler": p["filler"], "task": p["task"]})
    print(f"[smoke] taskset prompt {p['trace_id']}: {len(ids)} tokens, spans {sp}; filler tokens {sp['filler'][1] - sp['filler'][0]}")
    print("[smoke] rendered head:", repr(tok.decode(ids[:60]))[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "corpus", "prompts", "taskset", "states"])
    ap.add_argument("--stride", type=int, default=32, help="states: token stride between stored residuals")
    ap.add_argument("--runs", default="corpus1_math_a,corpus1_math_b,corpus1_idea_a,corpus1_idea_b")
    ap.add_argument("--run", default="power1"); ap.add_argument("--conditions", default="baseline,prompt_checking,prompt_exploratory")
    ap.add_argument("--n-tokens", type=int, default=2048); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--batch-gen", type=int, default=32)
    ap.add_argument("--flush", type=int, default=40); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--series-seeds", default="1", help="corpus: seeds whose per-position component series are stored ('all' = every trace)")
    ap.add_argument("--topics", default="math,gpqa")
    ap.add_argument("--varied-filler", action="store_true", help="taskset: one filler per item (D-56 amendment) -> data/neuro/taskset_varied")
    a = ap.parse_args()
    tok = load_tokenizer(); model = load_model()
    if a.mode == "states":                                                          # no component hooks needed
        run_states(model, tok, a.runs.split(","), a.conditions.split(","), a.stride, a.n_tokens, a.batch, a.limit); return
    D, dmeta = build_directions()
    hooks = CompHooks(model, D)
    _write_meta(hooks, dmeta)
    print(f"[neuro] {hooks.C} components, K={K} directions; cos(dom16, dom19)={dmeta['cos_dom16_dom19']:.3f}", flush=True)
    try:
        if a.mode == "smoke":
            smoke(model, tok, hooks, D)
        elif a.mode == "corpus":
            ss = None if a.series_seeds == "all" else [int(x) for x in a.series_seeds.split(",")]
            run_traces("corpus", model, tok, hooks, a.runs.split(","), None, a.n_tokens, a.batch, a.flush, ss, a.limit)
        elif a.mode == "prompts":
            run_traces("prompts", model, tok, hooks, [a.run], a.conditions.split(","), min(a.n_tokens, 1536), a.batch, a.flush, None, a.limit)
        else:
            run_taskset(model, tok, hooks, a.batch_gen, a.batch, a.flush, a.limit, tuple(a.topics.split(",")), varied=a.varied_filler)
    finally:
        hooks.remove()


if __name__ == "__main__":
    main()
