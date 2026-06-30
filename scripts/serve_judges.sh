#!/usr/bin/env bash
# scripts/serve_judges.sh — stand up N single-GPU OpenAI-compatible vLLM judge servers
# (one per GPU, ports 8001..800N), then point client code at http://localhost:800X/v1.
# Alternative to the offline batch path in run_judge.py; useful for the bake-off /
# interactive iteration. Run inside a Blackwell srun/sbatch allocation.
set -euo pipefail
MODEL="${1:?usage: serve_judges.sh <hf_model_id> [n_gpus] [quant]}"
N="${2:-4}"
QUANT="${3:-}"            # "" for bf16 (preferred, no quant risk) or "fp8"
EXTRA=""
[ -n "$QUANT" ] && EXTRA="--quantization $QUANT --kv-cache-dtype fp8_e4m3 --calculate-kv-scales"
for i in $(seq 0 $((N-1))); do
  PORT=$((8001+i))
  echo "GPU $i -> port $PORT"
  CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
    --served-model-name judge --port "$PORT" \
    --gpu-memory-utilization 0.92 --max-model-len 16384 $EXTRA \
    > "outputs/judge_gpu${i}.log" 2>&1 &
done
wait
