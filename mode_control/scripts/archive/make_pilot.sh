#!/bin/bash
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib PYTHONPATH=/ada1/u/harsh/thought-atlas
MAP=map_corpus1_dm8_L16_r8.npz
python -m mode_control.mc.make_run --run sigpilot_math --layer 16 --window 8192 --tag _v2w8192 --max-new-tokens 12288 --n-per-domain 40 --task-seed 20260910 --domains math --wave sigctrl --map $MAP --targets math_A4,math_A2 --gains 1,2 --exclude-runs corpus1_math_a,corpus1_math_b 2>&1 | grep -vE "Warning|warnings" | tail -1
python -m mode_control.mc.make_run --run sigpilot_idea --layer 16 --window 8192 --tag _v2w8192 --max-new-tokens 12288 --n-per-domain 40 --task-seed 20260910 --domains idea --wave sigctrl --map $MAP --targets idea_A3,idea_A4 --gains 1,2 --exclude-runs corpus1_idea_a,corpus1_idea_b 2>&1 | grep -vE "Warning|warnings" | tail -1
python - <<'PY'
import json
for r in ["sigpilot_math", "sigpilot_idea"]:
    s = json.load(open(f"mode_control/data/runs/{r}/run.json"))
    print(r, "excluded", len(s["exclude_instances"] or []), "conds:", [c["name"] for c in s["conditions"]])
PY
DEP=${1:?mapfit job id}
for r in sigpilot_math sigpilot_idea; do
  j=$(sbatch --exclude=grace --time=16:00:00 --dependency=afterok:$DEP --job-name=$r mode_control/scripts/mode.sbatch pilot $r sigctrl | awk '{print $NF}')
  q=$(sbatch --exclude=grace --time=03:00:00 --dependency=afterok:$j --job-name=$r-quality mode_control/scripts/mode.sbatch quality $r | awk '{print $NF}')
  e=$(sbatch --exclude=grace --time=01:00:00 --dependency=afterok:$q --job-name=$r-eval mode_control/scripts/mode.sbatch sigeval $r | awk '{print $NF}')
  echo "$r: pilot=$j quality=$q eval=$e"
done
squeue -u harsh -h -o "%i %j %T %r"
