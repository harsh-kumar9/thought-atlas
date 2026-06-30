#!/usr/bin/env bash
# env/blackwell_env_setup.sh — build the cu128 conda env for vega/mira (sm_120).
# Run ONCE from inside an interactive session on a Blackwell box:
#     srun --partition=ashton --qos=ashton -w vega --gres=gpu:1 \
#          --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash --login
#     bash env/blackwell_env_setup.sh
#
# Builds /ada1/u/harsh/miniconda3/envs/sote (shared NFS -> serves BOTH vega & mira,
# same sm_120 arch). The Grace cu121 `myproject`/`society` envs will NOT run here.
set -euo pipefail

ENV_NAME="${ENV_NAME:-sote}"          # "society-of-thought"
CONDA_ROOT="${CONDA_ROOT:-/ada1/u/harsh/miniconda3}"
PY="${PY:-3.11}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu128}"   # cu128 or cu129

eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda create -y -n "${ENV_NAME}" python="${PY}"
conda activate "${ENV_NAME}"

export PIP_CACHE_DIR=/ada1/u/harsh/.cache/pip   # avoid cross-device-link errors

# 1) torch FIRST, from the cu128/cu129 index (ships sm_120 kernels; stable 2.11+).
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"

# 2) Inference + analysis stack. vLLM recent (SM120 kernels land >=0.13; FP4 MoE newer).
#    NOTE: no flash-attn (no clean sm_120 path) — vLLM uses FlashInfer; HF uses sdpa.
pip install \
  "vllm>=0.13" \
  "transformers>=4.57" "accelerate>=1.2" "datasets>=3.2" \
  "polars>=1.0" "pyarrow>=18" "pandas>=2.2" "numpy>=2.0" \
  "hydra-core>=1.3" "omegaconf>=2.3" \
  "wandb" "rich" "tqdm" \
  "scikit-learn>=1.5" "scipy>=1.14" "statsmodels>=0.14" \
  "krippendorff" "pingouin"          # IRR: Krippendorff alpha, ICC
# Temporal-stats deps (functional ANOVA, HMM) — used by the analysis stage:
pip install "scikit-fda" "hmmlearn" "ruptures"

# 3) math-answer checking for the math performance metric
pip install "math-verify" "latex2sympy2" || echo "WARN: math-verify install failed; correctness falls back to normalize-compare"

echo "=== SANITY CHECKS (must all pass before trusting any run) ==="
python - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "device_capability", cap)
assert cap == (12, 0), f"expected sm_120 (12,0), got {cap} — wrong env/box"
# bf16 matmul on device
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); _ = (a @ b); torch.cuda.synchronize()
print("bf16 matmul ok")
import torch.nn.functional as F
q = torch.randn(1, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
F.scaled_dot_product_attention(q, q, q); print("sdpa ok")
PY

echo "=== vLLM import + tiny generate (proves SM120 kernels fire; watch for 'Marlin'/sm warnings) ==="
python - <<'PY'
from vllm import LLM, SamplingParams
# Tiny model just to prove the engine builds & generates on sm_120. Swap for a judge later.
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)
out = llm.generate(["2+2="], SamplingParams(max_tokens=8, temperature=0))
print("vllm generate ok:", repr(out[0].outputs[0].text))
PY

echo "DONE. Env '${ENV_NAME}' built at ${CONDA_ROOT}/envs/${ENV_NAME} (shared on vega+mira)."
