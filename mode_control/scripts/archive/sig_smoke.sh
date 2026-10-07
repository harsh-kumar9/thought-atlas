#!/bin/bash
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib PYTHONPATH=/ada1/u/harsh/thought-atlas
rm -rf mode_control/data/runs/sigsmoke
python -m mode_control.mc.make_run --run sigsmoke --layer 16 --window 8192 --tag _v2w8192 --max-new-tokens 256 --n-per-domain 2 --task-seed 20260910 --domains math,idea --wave sigctrl --map map_corpus1_dm8_L16_r8.npz --targets math_A4,idea_A3 --gains 1,2 2>&1 | grep -vE "Warning|warnings" | tail -1
sbatch --exclude=grace --time=00:40:00 --job-name=sigsmoke mode_control/scripts/mode.sbatch steer sigsmoke
squeue -u harsh -h -o "%i %j %T %r"
