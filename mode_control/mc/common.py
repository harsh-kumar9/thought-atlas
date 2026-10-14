"""Shared paths, model loading, prompt rendering, and residual-stream hooks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import os

REPO = Path(__file__).resolve().parents[2]
MC = REPO / "mode_control"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Model registry: select with MC_MODEL=<key> (keys match configs/exp.yaml gen_models).
MODELS = {
    "qwen35_9b": {"hf_id": "Qwen/Qwen3.5-9B", "loader": "qwen35"},
    "qwen35_4b": {"hf_id": "Qwen/Qwen3.5-4B", "loader": "qwen35"},
    "reasoner": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "loader": "causal_lm"},
}
GEN_MODEL_KEY = os.environ.get("MC_MODEL", "qwen35_9b")
MODEL_ID = MODELS[GEN_MODEL_KEY]["hf_id"]
# qwen35_9b keeps the original layout (data/); other models live under data/<key>/
DATA = MC / "data" if GEN_MODEL_KEY == "qwen35_9b" else MC / "data" / GEN_MODEL_KEY
JUDGE_MODEL = "google/gemma-4-31B-it"

KIM = ["Question_and_Answering", "Perspective_Shift", "Conflict_of_Perspectives", "Reconciliation"]
GAN = ["verification", "backtracking", "subgoal", "backward_chaining"]
BEH = KIM + GAN
FAMILIES = {"conv": KIM, "cog": GAN, "all": BEH}
DOMAINS = ["math", "idea"]

# Shared decoding from configs/exp.yaml (generation.decode, sampling_policy=shared).
DECODE = dict(temperature=0.6, top_p=0.95, top_k=20)

TRACES_BASELINE = REPO / f"data/v2/traces/traces_{GEN_MODEL_KEY}.parquet"
TRACKB_BASELINE = REPO / "data/v2/judge/trackB_full__google_gemma-4-31B-it.parquet"
TRACKA_BASELINE = REPO / "data/v2/judge/trackA_counts__google_gemma-4-31B-it.parquet"
TASKS_DIR = REPO / "data/v2/tasks"


def stable_fold(value: str, *, n_folds: int = 5, seed: int = 42) -> int:
    """Byte-identical copy of src/analysis/paper/splits.py::stable_fold (prompt-disjoint folds)."""
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_strategy_prompts() -> dict:
    return json.loads((MC / "configs/strategy_prompts.json").read_text())


# ---------------------------------------------------------------- model / tokenizer
def load_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str = MODEL_ID, device: str = "cuda:0"):
    import torch
    if MODELS[GEN_MODEL_KEY]["loader"] == "qwen35":
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    model = Cls.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    model.eval()
    return model


def hidden_size(model) -> int:
    cfg = model.config
    return int(getattr(cfg, "text_config", cfg).hidden_size)


def decoder_layers(model):
    """The ModuleList of decoder layers (Qwen3_5DecoderLayer), independent of wrapper class."""
    import torch
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) and \
                mod[0].__class__.__name__.endswith("DecoderLayer"):
            return mod
    raise RuntimeError("decoder layer list not found")


def render_prompt(tok, user_prompt: str, system: str | None = None) -> str:
    """Chat-template rendering identical to the generation contract (thinking enabled), plus an
    optional system message carrying a strategy instruction. The template ends with '<think>\\n'."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_prompt}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert out.endswith("<think>\n"), "unexpected chat template: thinking prefix missing"
    return out


def _layer_output(output):
    import torch
    return output if torch.is_tensor(output) else output[0]


def _replace_output(output, h):
    import torch
    return h if torch.is_tensor(output) else (h,) + tuple(output[1:])


# ---------------------------------------------------------------- hooks
class Capture:
    """Forward hook that pools the residual stream at fixed positions.

    positions: dict name -> (start, stop) slice over the sequence axis (negative allowed,
    interpreted on the padded right-aligned batch). Pooling = mean over the slice.
    """

    def __init__(self, positions: dict[str, tuple[int, int]]):
        self.positions = positions
        self.out: dict[str, np.ndarray] = {}

    def __call__(self, module, args, output):
        import torch
        h = _layer_output(output)
        for name, (a, b) in self.positions.items():
            self.out[name] = h[:, a:b, :].float().mean(dim=1).cpu().numpy().astype(np.float16)
        return None


class Steerer:
    """Residual-stream intervention in a subspace U at one decoder layer.

    z = U^T (h - center)
    mode "gamma": h' = h + U[(gamma-1) z]         (1 = identity, 2 = amplify, 0 = neutralize, -1 = reflect)
    mode "add":   h' = h + U alpha                  (alpha: r-vector in raw projection units)
    mode "swap":  h' = h + U (target - z)           (target: r-vector, e.g. the other domain's centroid)

    Timing: call 0 is the prefill. With sustained=False the intervention covers the last
    `prompt_tail` prompt positions plus decode calls 1..pulse_k-prompt_tail, i.e. the residual
    positions that produce generated tokens 1..pulse_k. With sustained=True every position.
    """

    def __init__(self, U, center, mode: str, *, gamma: float | None = None, alpha=None,
                 target=None, pulse_k: int = 64, sustained: bool = False, prompt_tail: int = 1,
                 alpha_late=None, readout=None):
        import torch
        # mode "target" (signature controller, D-33): readout = dict(W [d,k], b [k], scale, phi_star [k], gain, lam, mask [k])
        # h' = h + gain * G (phi_star - phi_hat(h)), phi_hat = (h/scale) W + b, G = scale * W (W^T W + lam I)^-1 restricted to mask
        self.readout = None
        if readout is not None:
            W = np.asarray(readout["W"], np.float32); k = W.shape[1]
            mask = np.asarray(readout.get("mask", np.ones(k)), np.float32)
            mu = np.asarray(readout["mu"], np.float32); sd = np.asarray(readout["sd"], np.float32)
            # ridge relative to the readout scale (rel_lam x mean eigenvalue of W^T W); absolute "lam" kept for old specs
            lam = float(readout["rel_lam"]) * float(np.trace(W.T @ W) / k) if "rel_lam" in readout else float(readout.get("lam", 1e-3))
            G = (sd[:, None] * W) @ np.linalg.inv(W.T @ W + lam * np.eye(k, dtype=np.float32))   # [d, k], in raw units
            self.readout = {"W": torch.as_tensor(W), "b": torch.as_tensor(np.asarray(readout["b"], np.float32)),
                            "G": torch.as_tensor(G.astype(np.float32)), "mu": torch.as_tensor(mu), "sd": torch.as_tensor(sd),
                            "phi_star": torch.as_tensor(np.asarray(readout["phi_star"], np.float32)), "gain": float(readout.get("gain", 1.0)),
                            "mask": torch.as_tensor(mask), "phi_star_abs": np.asarray(readout["phi_star"], np.float32),
                            "cap": float(readout.get("cap", 0.0)), "open_loop": bool(readout.get("open_loop", False)),
                            "open_norm": float(readout.get("open_norm", 0.0))}
            if "domain_means" in readout: self.readout["domain_means"] = readout["domain_means"]
        self.U = torch.as_tensor(np.asarray(U, dtype=np.float32))          # [d, r]
        self.center = torch.as_tensor(np.asarray(center, dtype=np.float32))
        self.mode = mode
        self.gamma = gamma
        self.alpha = None if alpha is None else torch.as_tensor(np.asarray(alpha, dtype=np.float32))
        # scheduled additive steering (mode "add" only): alpha for generated tokens 1..switch, alpha_late after.
        # `switch` is set per batch (int, or a [B] tensor of per-row switch points) before generation.
        self.alpha_late = None if alpha_late is None else torch.as_tensor(np.asarray(alpha_late, dtype=np.float32))
        self.switch = None
        self.target = None if target is None else torch.as_tensor(np.asarray(target, dtype=np.float32))
        self.pulse_k = int(pulse_k)
        self.sustained = bool(sustained)
        self.prompt_tail = int(prompt_tail)
        self.calls = 0
        self.n_steered_positions = 0
        self.n_late_positions = 0
        self.device = None

    def reset(self):
        self.calls = 0
        self.n_steered_positions = 0
        self.n_late_positions = 0

    def _to(self, device):
        if self.device != device:
            self.U = self.U.to(device); self.center = self.center.to(device)
            if self.alpha is not None: self.alpha = self.alpha.to(device)
            if self.alpha_late is not None: self.alpha_late = self.alpha_late.to(device)
            if self.readout is not None: self.readout = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.readout.items()}
            if self.target is not None: self.target = self.target.to(device)
            self.device = device

    def __call__(self, module, args, output):
        import torch
        h = _layer_output(output)
        B, T, d = h.shape
        call = self.calls
        self.calls += 1
        if self.sustained:
            sl = slice(0, T)
        elif call == 0:
            if self.prompt_tail <= 0:
                return None
            sl = slice(max(0, T - self.prompt_tail), T)
        else:
            if call > self.pulse_k - self.prompt_tail:
                return None
            sl = slice(0, T)
        self._to(h.device)
        x = h[:, sl, :].float()
        if self.mode == "target":
            R = self.readout
            phi_hat = ((x - R["mu"]) / R["sd"]) @ R["W"] + R["b"]              # [B, t, k]
            ps = R["phi_star"]
            if ps.dim() == 2: ps = ps[:, None, :]                                  # per-row targets [B, 1, k]
            if R["open_loop"]:
                # constant direction per row: move the readout from the domain mean to the target, scaled to a fixed norm
                v = (ps * R["mask"]) @ R["G"].T                                    # [B, 1, d]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-6) * R["open_norm"]
                delta = R["gain"] * v.expand_as(x)
            else:
                err = (ps - phi_hat) * R["mask"]
                delta = R["gain"] * (err @ R["G"].T)                              # [B, t, d]
                if R["cap"] > 0:                                                    # per-token cap relative to the residual norm
                    lim = R["cap"] * x.norm(dim=-1, keepdim=True); dn = delta.norm(dim=-1, keepdim=True)
                    delta = delta * torch.clamp(lim / (dn + 1e-6), max=1.0)
            h2 = h.clone(); h2[:, sl, :] = (x + delta).to(h.dtype)
            self.n_steered_positions += (sl.stop - sl.start) if sl.stop else T
            return _replace_output(output, h2)
        z = (x - self.center) @ self.U                                   # [B, t, r]
        if self.mode == "gamma":
            delta = ((self.gamma - 1.0) * z) @ self.U.T
        elif self.mode == "add" and self.alpha_late is not None:
            # generated-token index produced by this call: prefill (call 0) -> token 1 (all-early);
            # decode call c -> token c + 1 (prompt_tail = 1)
            sw = self.switch if self.switch is not None else 0
            sw = torch.as_tensor(sw, device=h.device).reshape(-1)
            if sw.numel() == 1: sw = sw.expand(B)
            early = (call + 1) <= sw if call > 0 else torch.ones(B, dtype=torch.bool, device=h.device)
            a_row = torch.where(early[:, None], self.alpha[None, :], self.alpha_late[None, :])   # [B, r]
            delta = (a_row @ self.U.T)[:, None, :].expand_as(x)
            self.n_late_positions += int((~early).sum()) * T   # decode calls have T = 1; prefill is all-early
        elif self.mode == "add":
            delta = (self.alpha @ self.U.T).expand_as(x)
        elif self.mode == "swap":
            delta = (self.target - z) @ self.U.T
        else:
            raise ValueError(self.mode)
        h2 = h.clone()
        h2[:, sl, :] = (x + delta).to(h.dtype)
        self.n_steered_positions += (sl.stop - sl.start) if sl.stop else T
        return _replace_output(output, h2)


class HeadAblator:
    """D-63: silences chosen heads of one decoder layer by zeroing their slice of the attention output before o_proj
    (full attention, after the sigmoid gate) or out_proj (linear attention, after the gated norm): the gate-closed state
    of this architecture. Applied on every position, prompt and generation. attach(layer) registers the pre-hook."""

    def __init__(self, heads: list[int]):
        self.heads = sorted(set(int(h) for h in heads)); self.mode = "ablate"
        self.calls = 0; self.n_steered_positions = 0; self.n_late_positions = 0; self.nh = None; self.hd = None

    def reset(self):
        self.calls = 0; self.n_steered_positions = 0

    def attach(self, layer):
        if hasattr(layer, "self_attn"):
            proj = layer.self_attn.o_proj; nh = proj.in_features // int(layer.self_attn.head_dim)
        else:
            proj = layer.linear_attn.out_proj; nh = int(layer.linear_attn.num_v_heads)
        hd = proj.in_features // nh; self.nh, self.hd = nh, hd
        assert max(self.heads) < nh, (self.heads, nh)
        heads = self.heads

        def pre(module, args):
            x = args[0]; B, T = x.shape[:2]
            self.calls += 1; self.n_steered_positions += T
            x = x.reshape(B, T, nh, hd).clone(); x[:, :, heads, :] = 0
            return (x.reshape(B, T, nh * hd),) + tuple(args[1:])
        return proj.register_forward_pre_hook(pre)


def orthonormal(U: np.ndarray) -> np.ndarray:
    """QR orthonormal basis with each column's sign fixed so that q_j · U_j > 0 (QR's sign is arbitrary)."""
    U = np.asarray(U, dtype=np.float64)
    q, _ = np.linalg.qr(U)
    for j in range(q.shape[1]):
        if q[:, j] @ U[:, j] < 0:
            q[:, j] *= -1
    return q.astype(np.float32)
