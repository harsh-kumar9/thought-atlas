#!/bin/bash
# P2 step 1 (D-42): within-task success correlates per domain; CPU only, idempotent.
source /ada1/u/harsh/miniconda3/etc/profile.d/conda.sh
conda activate sote
cd /ada1/u/harsh/thought-atlas
export PYTHONPATH=/ada1/u/harsh/thought-atlas CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
python -m mode_control.mc.success_dir > /ada1/u/harsh/thought-atlas/mode_control/outputs/p2_success_dir.log 2>&1
echo "exit $?" >> /ada1/u/harsh/thought-atlas/mode_control/outputs/p2_success_dir.log
