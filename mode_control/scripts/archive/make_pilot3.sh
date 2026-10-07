#!/bin/bash
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib PYTHONPATH=/ada1/u/harsh/thought-atlas
python -m mode_control.mc.make_run --run sigpilot3_math --layer 19 --window 8192 --tag _v2w8192 --wave sigctrl3 --map map_corpus1_dm8_L16_r8.npz --targets math_A2 --rel-lam 0.001 --cap 0.25 --open-norm 2.25 --max-new-tokens 12288 --n-per-domain 200 --task-seed 20260911 --domains math --exclude-runs corpus1_math_a,corpus1_math_b,sigpilot2_math 2>&1 | grep -vE "Warning|warnings" | tail -1
python - <<'PY'
import json
s = json.load(open("mode_control/data/runs/sigpilot3_math/run.json")); print("excluded", len(s["exclude_instances"]), "conds", [c["name"] for c in s["conditions"]])
PY
j0=$(sbatch --exclude=grace --time=20:00:00 --job-name=sigpilot3_math-a mode_control/scripts/mode.sbatch pilot sigpilot3_math sigctrl3 baseline,cl_math_A2_g2p0,cl_random_g2p0 | awk '{print $NF}')
j1=$(sbatch --exclude=grace --time=20:00:00 --job-name=sigpilot3_math-b mode_control/scripts/mode.sbatch pilot sigpilot3_math sigctrl3 ol_math_A2_g3p0,ol_random_g3p0,sus_math_L19 | awk '{print $NF}')
e=$(sbatch --exclude=grace --time=01:00:00 --dependency=afterok:$j0:$j1 --job-name=sigpilot3_math-eval mode_control/scripts/mode.sbatch sigeval sigpilot3_math | awk '{print $NF}')
echo "jobs: a=$j0 b=$j1 eval=$e"
squeue -u harsh -h -o "%i %j %T %r" | grep sig
