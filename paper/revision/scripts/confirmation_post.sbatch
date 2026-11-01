#!/bin/bash
#SBATCH --job-name=atlas_confirm_cpu
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=vega
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
REVISION_ROOT=$1
STAGE=$2
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
cd "$REVISION_ROOT"
if [ "$STAGE" = prepare_labels ]; then
 /ada1/u/harsh/miniconda3/envs/sote/bin/python scripts/confirmation.py prepare_labels --root /ada1/u/harsh/thought-atlas --out results/confirmation
else
 /ada1/u/harsh/miniconda3/envs/sote/bin/python scripts/analyze_confirmation.py "$STAGE" --root /ada1/u/harsh/thought-atlas --out results/confirmation
fi
