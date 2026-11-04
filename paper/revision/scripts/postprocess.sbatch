#!/bin/bash
#SBATCH --job-name=atlas_repair_analysis
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=vega
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
REVISION_ROOT=$1
STUDY=$2
ANALYSIS_ROOT=${3:-$REVISION_ROOT/results/$STUDY}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=4
cd "$REVISION_ROOT"
case "$STUDY" in
  multijudge) /ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/analyze_multijudge.py --out "$ANALYSIS_ROOT" ;;
  elicitation) /ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/analyze_elicitation.py --out results/elicitation ;;
  rl_evaluation) /ada1/u/harsh/miniconda3/envs/sote/bin/python -u scripts/analyze_rl_evaluation.py --out results/rl_evaluation ;;
  *) exit 2 ;;
esac
