#!/bin/bash
#SBATCH --job-name=atlas_readout_audit
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=vega
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 POLARS_MAX_THREADS=4
REVISION_ROOT=$1
STAGE=$2
cd "$REVISION_ROOT"
/ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/readout_validation.py "$STAGE" \
 --root /ada1/u/harsh/thought-atlas --out "$REVISION_ROOT/results/readout" --fold "${SLURM_ARRAY_TASK_ID:-0}"
