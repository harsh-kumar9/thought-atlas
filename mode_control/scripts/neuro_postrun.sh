#!/bin/bash
# D-59/D-62 post-run stages for the nm_math and nm_idea steering runs. Run from ada after both runs have finished
# (every arm has its steer_log_<cond>.json and no partial_<cond>.parquet). Submits, per run: Track B on seed 1 (judgeB),
# and for nm_math the free-text extraction (freexg, 2 shards), for nm_idea the quality judge. Read each job's output
# before the next stage (neuro_steer_eval, density).
set -u
source /ada1/u/harsh/miniconda3/etc/profile.d/conda.sh && conda activate sote
cd /ada1/u/harsh/thought-atlas
export PYTHONPATH=/ada1/u/harsh/thought-atlas HF_HUB_OFFLINE=1 TMPDIR=/ada1/u/harsh/tmp OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
DD=mode_control/data
for run in nm_math nm_idea; do
  if ls $DD/runs/$run/partial_*.parquet >/dev/null 2>&1; then echo "$run: generation still in progress"; continue; fi
  python -m mode_control.mc.tx_analyze stage_judge --run $run
  sbatch --exclude=grace --time=12:00:00 --job-name=tb_$run mode_control/scripts/mode.sbatch judgeB $run "$DD/runs/$run/judge_in_s1/traces_*.parquet" s1
done
if [ ! -f $DD/runs/nm_math/partial_line_s0p5.parquet ] && [ -f $DD/runs/nm_math/steer_log_rand_n1p0.json ]; then
  mkdir -p $DD/extract_in_nm
  for p in $DD/runs/nm_math/traces_*.parquet; do ln -sf "$PWD/$p" "$DD/extract_in_nm/traces_nm_math__$(basename ${p#*/traces_})"; done
  ls $DD/extract_in_nm | wc -l
  sbatch --exclude=grace --time=08:00:00 --job-name=freex_nm0 mode_control/scripts/mode.sbatch freexg mode_control/data/extract_in_nm mode_control/data/extract_free_nm 0 2
  sbatch --exclude=grace --time=08:00:00 --job-name=freex_nm1 mode_control/scripts/mode.sbatch freexg mode_control/data/extract_in_nm mode_control/data/extract_free_nm 1 2
fi
if [ -f $DD/runs/nm_idea/steer_log_rand_n1p0.json ] && [ -f $DD/runs/nm_idea/steer_log_curve_s0p5.json ]; then
  sbatch --exclude=grace --time=12:00:00 --job-name=q_nm_idea mode_control/scripts/mode.sbatch quality nm_idea
fi
squeue -u harsh -o "%.9i %.14j %.2t %.10M %R" | grep -v JOBID
