#!/bin/bash
set -euo pipefail
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib PYTHONPATH=/ada1/u/harsh/thought-atlas
python -m mode_control.mc.make_run --run sigdiv_math --layer 19 --window 8192 --tag _v2w8192 --wave sigctrl3 --map map_corpus1_dm8_L16_r8.npz --targets math_A2 --rel-lam 0.001 --cap 0.25 --open-norm 2.25 --max-new-tokens 12288 --n-per-domain 40 --task-seed 20260906 --seeds 1,2,3,4,5 --domains math --only cl_math_A2_g2p0 2>&1 | grep -vE "Warning|warnings" | tail -1
python - <<'PY'
import polars as pl, json
a = set(pl.read_parquet("mode_control/data/runs/sched1_div/tasks.parquet").filter(pl.col("task_type") == "math")["instance_id"].to_list())
s = json.load(open("mode_control/data/runs/sigdiv_math/run.json")); print("conds", [c["name"] for c in s["conditions"]], "seeds", s["seeds"], "task_seed", s["task_seed"])
PY
j=$(sbatch --exclude=grace --time=12:00:00 --job-name=sigdiv_math mode_control/scripts/mode.sbatch pilot sigdiv_math sigctrl3 cl_math_A2_g2p0 | awk '{print $NF}')
echo "job $j"; squeue -u harsh -h -o "%i %j %T %r" | grep -E " sig"
