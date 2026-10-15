"""D-64 stage 2 smoke for Gemma-4-12B (run with MC_MODEL=gemma4_12b on one GPU).

Part A (HF): load the unified checkpoint, find the text decoder layers, teacher-force one readout trace with hooks at the
candidate layers and print the residual norms (the dose rule in D-64 needs the median norm at the chosen layer).
Part B (vLLM): build the steering engine (unified plugin, hooks installed), render three held-out math prompts through the
atlas thinking spec, generate unsteered, with a zero vector (must reproduce the unsteered text) and with a random vector at
12.5% of the layer-24 norm (must change the text and count steered events); parse each output with the canonical think
normaliser and report think/answer lengths. Exit 2 on any failed check.
Output: data/gemma4_12b/smoke.json and the printed checks.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import polars as pl

from .common import DATA, GEN_MODEL_KEY, MODEL_ID, canon_text, decoder_layers, load_model, load_tokenizer, render_prompt, _layer_output

LAYERS = [16, 20, 24, 28, 32]


def part_a(tok) -> dict:
    import torch
    model = load_model(); dl = decoder_layers(model)
    info = {"n_layers": len(dl), "layer_class": dl[0].__class__.__name__, "hidden": int(model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size)}
    print("[smoke A]", info, flush=True)
    tr = pl.read_parquet(DATA / "runs" / "atlas_math" / "traces_baseline.parquet").filter(pl.col("reasoning_text_for_analysis").is_not_null()).head(2)
    store = {}
    hs = [dl[l].register_forward_hook((lambda l: (lambda m, a, o: store.__setitem__(l, _layer_output(o).detach().float())))(l)) for l in LAYERS]
    norms = {}
    try:
        for r in tr.iter_rows(named=True):
            p = tok(render_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            t = tok(r["reasoning_text_for_analysis"], add_special_tokens=False)["input_ids"][:2048]
            with torch.no_grad():
                model(input_ids=torch.tensor([p + t], device=model.device))
            for l in LAYERS:
                h = store[l][0, len(p):, :]
                norms.setdefault(l, []).append(float(h.norm(dim=-1).median()))
    finally:
        for h in hs: h.remove()
    info["median_norm"] = {str(l): float(np.mean(v)) for l, v in norms.items()}
    print("[smoke A] median residual norm by layer:", info["median_norm"], flush=True)
    del model; torch.cuda.empty_cache()
    return info


def part_b(tok, norm24: float) -> dict:
    from vllm import SamplingParams
    from .vsteer import _S, _set_steer, build_vllm
    from src.utils.parse import parse_generation_detailed
    llm = build_vllm(max_model_len=8192, hooks=True)
    tasks = pl.read_parquet(DATA / "tasks_heldout_math.parquet").head(3)
    prompts = [render_prompt(tok, r["prompt"]) for r in tasks.iter_rows(named=True)]
    print("[smoke B] rendered prompt tail:", repr(prompts[0][-160:]), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1500, seed=0, skip_special_tokens=False)
    out = {}
    rng = np.random.default_rng(0)
    for name, vec in (("none", None), ("zero", {"v": np.zeros(3840, np.float32), "layer": 24}),
                      ("random", {"v": (rng.standard_normal(3840) / np.sqrt(3840) * 0.125 * norm24).astype(np.float32), "layer": 24})):
        _set_steer(llm, tok, vec, rule="all")
        t0 = time.time()
        res = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in res]
        parsed = [parse_generation_detailed(canon_text(t), "reasoning", finish_reason="stop" if o.outputs[0].finish_reason == "stop" else "length") for t, o in zip(texts, res)]
        out[name] = {"n_steered": int(_S["n_steered"]), "n_mismatch": int(_S["n_mismatch"]), "seconds": round(time.time() - t0, 1),
                     "think_chars": [len(p["think_text"] or "") for p in parsed], "answer_chars": [len(p["answer_text"] or "") for p in parsed],
                     "parse_status": [p["parse_status"] for p in parsed], "head": texts[0][:200]}
        print(f"[smoke B] {name}: {out[name]}", flush=True)
    return out


def main():
    assert GEN_MODEL_KEY == "gemma4_12b", "run with MC_MODEL=gemma4_12b"
    tok = load_tokenizer()
    a = part_a(tok)
    b = part_b(tok, a["median_norm"]["24"])
    checks = {
        "48_layers": a["n_layers"] == 48,
        "zero_vector_counts_events": b["zero"]["n_steered"] > 0 and b["zero"]["n_mismatch"] == 0,
        "random_vector_counts_events": b["random"]["n_steered"] > 0,
        "think_parsed_unsteered": all(c > 0 for c in b["none"]["think_chars"]),
        "zero_matches_unsteered_lengths": b["zero"]["think_chars"] == b["none"]["think_chars"] and b["zero"]["answer_chars"] == b["none"]["answer_chars"],
    }
    res = {"model": MODEL_ID, "part_a": a, "part_b": b, "checks": checks}
    (DATA / "smoke.json").write_text(json.dumps(res, indent=1))
    print("[smoke] checks:", checks, flush=True)
    if not all(checks.values()):
        print("[smoke] FAILED", flush=True); sys.exit(2)
    print("[smoke] passed", flush=True)


if __name__ == "__main__":
    main()
