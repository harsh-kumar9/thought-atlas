#!/bin/bash
#SBATCH --job-name=atlas_behavior
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=vega
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
REVISION_ROOT=$1
TARGETS=(math gpqa planning idea moral)
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
cd "$REVISION_ROOT"
/ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/export_behavior.py \
 --root /ada1/u/harsh/thought-atlas --out results/behavior --target "${TARGETS[$SLURM_ARRAY_TASK_ID]}"
