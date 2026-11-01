#!/bin/bash
# CPU-only; submit from an isolated source snapshot and precreate the log folder.
#SBATCH --job-name=atlas_random_timing
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=vega
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$2/mplcache"
export PYTHONPATH="$1"
mkdir -p "$MPLCONFIGDIR"
cd "$1"
/ada1/u/harsh/miniconda3/envs/sote/bin/python -u paper/scripts/random_timing_control.py \
  --out "$2" --workers 8 --repeats 100 --bootstrap 10000
