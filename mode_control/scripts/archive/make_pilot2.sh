#!/bin/bash
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib PYTHONPATH=/ada1/u/harsh/thought-atlas
MAP=map_corpus1_dm8_L16_r8.npz
COMMON="--layer 16 --window 8192 --tag _v2w8192 --wave sigctrl2 --map $MAP --gains 2,3 --rel-lam 0.001 --cap 0.25 --open-norm 2.25"
rm -rf mode_control/data/runs/sigsmoke2
python -m mode_control.mc.make_run --run sigsmoke2 $COMMON --max-new-tokens 256 --n-per-domain 2 --task-seed 20260910 --domains math,idea --targets math_A2,idea_A4 2>&1 | grep -vE "Warning|warnings" | tail -1
python -m mode_control.mc.make_run --run sigpilot2_math $COMMON --max-new-tokens 12288 --n-per-domain 40 --task-seed 20260910 --domains math --targets math_A4,math_A2 --exclude-runs corpus1_math_a,corpus1_math_b 2>&1 | grep -vE "Warning|warnings" | tail -1
python -m mode_control.mc.make_run --run sigpilot2_idea $COMMON --max-new-tokens 12288 --n-per-domain 40 --task-seed 20260910 --domains idea --targets idea_A3,idea_A4 --exclude-runs corpus1_idea_a,corpus1_idea_b 2>&1 | grep -vE "Warning|warnings" | tail -1
python - <<'PY'
import json
s = json.load(open("mode_control/data/runs/sigpilot2_math/run.json")); print("conds:", [c["name"] for c in s["conditions"]])
PY
sm=$(sbatch --exclude=grace --time=00:40:00 --job-name=sigsmoke2 mode_control/scripts/mode.sbatch steer sigsmoke2 | awk '{print $NF}'); echo "smoke $sm"
for r in sigpilot2_math sigpilot2_idea; do
  j=$(sbatch --exclude=grace --time=20:00:00 --dependency=afterok:$sm --job-name=$r mode_control/scripts/mode.sbatch pilot $r sigctrl2 | awk '{print $NF}')
  q=$(sbatch --exclude=grace --time=03:00:00 --dependency=afterok:$j --job-name=$r-quality mode_control/scripts/mode.sbatch quality $r | awk '{print $NF}')
  e=$(sbatch --exclude=grace --time=01:00:00 --dependency=afterany:$q --job-name=$r-eval mode_control/scripts/mode.sbatch sigeval $r | awk '{print $NF}')
  echo "$r: pilot=$j quality=$q eval=$e"
done
squeue -u harsh -h -o "%i %j %T %r"
