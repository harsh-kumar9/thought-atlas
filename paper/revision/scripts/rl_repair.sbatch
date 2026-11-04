#!/bin/bash
#SBATCH --job-name=atlas_rl_repair
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=mira
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --requeue
set -euo pipefail
# Native sampler avoids FlashInfer sampling JIT requiring unavailable nvcc.
export VLLM_USE_FLASHINFER_SAMPLER=0
REVISION_ROOT=$1
RUN_NAME=$2
shift 2
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache TMPDIR=/ada1/u/harsh/tmp
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
cd "$REVISION_ROOT"
/ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/rl_repair.py \
 --root /ada1/u/harsh/thought-atlas --out "results/rl/$RUN_NAME" "$@"
if /ada1/u/harsh/miniconda3/envs/sote/bin/python -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d["stage"]!="complete" else 1)' "results/rl/$RUN_NAME/status.json"; then
 /opt/slurm/bin/scontrol requeue "$SLURM_JOB_ID"
fi
