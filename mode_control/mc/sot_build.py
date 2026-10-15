"""D-67 build: the Society of Thought feature on R1-Distill-Llama-8B, as vsteer vectors and a Countdown run spec.

1. Download the Llama Scope SAE for the layer-15 residual (fnlp/Llama3_1-8B-Base-LXR-8x, Llama3_1-8B-Base-L15R-8x) into the HF
   cache; read the decoder matrix; take feature 30939 and two seeded random features (not 30939).
2. Write vsteer vectors under data/reasoner/vsteer/: f30939 at s = +10, +5, -10; two random features at +10; a Gaussian
   direction at the norm of 10 d_30939. All at decoder layer 15 (0-based, output), the SAE's hook point.
3. Write data/reasoner/runs/sot_countdown/{tasks.parquet, run.json}: the 200 cd_countdown tasks, 2 seeds, 32k budget.
Run with MC_MODEL=reasoner on the ada login node (internet for the download).
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .common import DATA, GEN_MODEL_KEY, MC

assert GEN_MODEL_KEY == "reasoner", "run with MC_MODEL=reasoner"
REPO, SUB = "fnlp/Llama3_1-8B-Base-LXR-8x", "Llama3_1-8B-Base-L15R-8x"
FEAT, LAYER, SEEDS = 30939, 15, [1, 2]
VS = DATA / "vsteer"; VS.mkdir(parents=True, exist_ok=True)

hp = json.load(open(hf_hub_download(REPO, f"{SUB}/hyperparams.json")))
print("[sot] hyperparams keys:", list(hp.keys())[:20])
print("[sot] hook / d_model / d_sae:", hp.get("hook_point_in") or hp.get("hook_point"), hp.get("d_model"), hp.get("d_sae"))
print("[sot] normalisation:", {k: hp.get(k) for k in ("norm_activation", "dataset_average_activation_norm", "decoder_exactly_fixed_norm", "sparsity_include_decoder_norm", "act_fn", "jump_relu_threshold", "use_decoder_bias")})
path = hf_hub_download(REPO, f"{SUB}/checkpoints/final.safetensors")
W = {k: v.float().numpy() for k, v in load_file(path).items()}
for k, v in W.items():
    print(f"[sot] tensor {k}: {v.shape} {v.dtype}")
dec_key = next(k for k, v in W.items() if "decoder" in k.lower() and v.ndim == 2)
D = W[dec_key].astype(np.float32)
if D.shape[0] != 32768 and D.shape[1] == 32768:
    D = D.T
assert D.shape == (32768, 4096), D.shape
norms = np.linalg.norm(D, axis=1)
print(f"[sot] decoder {dec_key}: {D.shape}; row norms mean {norms.mean():.3f} sd {norms.std():.3f}; feature {FEAT} norm {norms[FEAT]:.3f}")
d = D[FEAT]
rng = np.random.default_rng(20261014)
rand_feats = [int(x) for x in rng.choice([i for i in range(32768) if i != FEAT], 2, replace=False)]
meta = {"source": "llama_scope_sae", "repo": REPO, "sub": SUB, "decoder_key": dec_key, "hook": "blocks.15.hook_resid_post", "decision": "D-67"}


def save(name, v, extra):
    np.savez(VS / f"vec_{name}.npz", v=v.astype(np.float32), layer=LAYER, meta=json.dumps({**meta, **extra, "norm": float(np.linalg.norm(v))}))
    print(f"[sot] vec_{name}: |v| {np.linalg.norm(v):.3f}")


# Units. The SAE is trained on dataset-wise normalised activations: x_norm = x * sqrt(d_model) / dataset_average_activation_norm
# (10.8125 at layer 15). A feature steer of strength s inside the SAE's space is s * d * (avg_norm / sqrt(d_model)) in raw
# residual units. The paper's formula h' = h + s d does not say which space; the literal raw reading at s = 10 adds a vector
# 1.7x the typical residual norm, so the normalised reading is primary and the literal one is a single dose-check arm.
scale = float(hp["dataset_average_activation_norm"]["out"]) / np.sqrt(4096.0)
meta["raw_scale_per_unit_s"] = scale
print(f"[sot] normalised-unit scale: s=1 in SAE space = {scale:.4f} * d in raw units; |d_{FEAT}| = {np.linalg.norm(d):.3f}; raw norm at s=10: {10 * scale * np.linalg.norm(d):.3f}")
for s in (10, 5, -10):
    save(f"f{FEAT}_{'p' if s > 0 else 'm'}{abs(s)}", s * scale * d, {"feature": FEAT, "s": s, "units": "sae_normalised"})
save(f"f{FEAT}_p10raw", 10 * d, {"feature": FEAT, "s": 10, "units": "raw_literal"})
for j, f in enumerate(rand_feats):
    save(f"randfeat{'ab'[j]}_p10", 10 * scale * D[f], {"feature": int(f), "s": 10, "units": "sae_normalised", "random_feature": True})
g = rng.standard_normal(4096); g = g / np.linalg.norm(g) * 10 * scale * np.linalg.norm(d)
save("gauss_n10d", g, {"random_gaussian": True, "matched_to": f"10 * scale * |d_{FEAT}|"})

rd = DATA / "runs" / "sot_countdown"; rd.mkdir(parents=True, exist_ok=True)
src = MC / "data" / "runs" / "cd_countdown" / "tasks.parquet"
shutil.copy(src, rd / "tasks.parquet")
t = pl.read_parquet(rd / "tasks.parquet"); print(f"[sot] tasks: {t.height} ({t['task_type'].unique().to_list()})")
conds = [{"name": "baseline", "mode": "baseline"}] + [
    {"name": n, "mode": "vector", "vector": n, "positions": "all"}
    for n in (f"f{FEAT}_p10", f"f{FEAT}_p5", f"f{FEAT}_m10", f"f{FEAT}_p10raw", "randfeata_p10", "randfeatb_p10", "gauss_n10d")]
spec = {"run": "sot_countdown", "engine": "vllm", "seeds": SEEDS, "max_new_tokens": 32768, "window": 8192, "tasks_from": "sot_countdown",
        "decision": "D-67", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "sae": f"{REPO}/{SUB}", "feature": FEAT,
        "random_features": rand_feats, "units_primary": "sae_normalised", "raw_scale_per_unit_s": scale, "conditions": conds}
(rd / "run.json").write_text(json.dumps(spec, indent=2))
print("[sot] run.json ->", rd / "run.json", "conditions:", [c["name"] for c in conds])
