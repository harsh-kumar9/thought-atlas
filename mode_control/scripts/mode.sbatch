#!/bin/bash
# mode_control/scripts/mode.sbatch — single-GPU stages for the reasoning-mode control study.
# Submit from ada, repo root:  sbatch -w mira mode_control/scripts/mode.sbatch <stage> [args...]
#   stages: smoke | acts | pilot <run> <wave> [conds] | steer <run> [conds] | judge <run> [conds] | quality <run>
#SBATCH --job-name=mode
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=mode_control/outputs/%x_%j.log
set -euo pipefail
# On failure kill only processes on THIS job's GPU. (Until 2026-10-07 this ran `pkill -u $USER -f VLLM::EngineCore`, which
# killed every vLLM engine of ours on the node, including healthy jobs that shared it; SLURM's cgroup cleanup handles the rest.)
cleanup() { rc=$?; pkill -9 -P $$ 2>/dev/null || true; if [ "$rc" -ne 0 ]; then for p in $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; fi; }
trap cleanup EXIT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer
export TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export XDG_CONFIG_HOME=/ada1/u/harsh/.config MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib TMPDIR=/ada1/u/harsh/tmp
# CUDA's JIT cache and flashinfer's workspace default to the NFS home; jobs hung for hours in rpc_wait on
# /h/224/harsh/.nv/ComputeCache/index (2026-10-03, again 2026-10-07). Keep both on /ada1.
export CUDA_CACHE_PATH=/ada1/u/harsh/.cache/nv/ComputeCache FLASHINFER_WORKSPACE_BASE=/ada1/u/harsh/.cache
export PYTHONPATH=/ada1/u/harsh/thought-atlas:${PYTHONPATH:-}
export MC_MODEL=${MC_MODEL:-qwen35_9b}
if [ "$MC_MODEL" = qwen35_9b ]; then DD=mode_control/data; else DD=mode_control/data/$MC_MODEL; fi
mkdir -p /ada1/u/harsh/.config /ada1/u/harsh/.cache/matplotlib /ada1/u/harsh/tmp /ada1/u/harsh/.cache/nv/ComputeCache
cd /ada1/u/harsh/thought-atlas
pwd; hostname; date; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" | head -1)
if [ "${USED:-0}" -gt 8000 ]; then echo "GPU already has ${USED}MB used (leaked process?) — aborting"; exit 3; fi
set +u; eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate sote; set -u
STAGE=${1:?stage}; shift || true
case "$STAGE" in
  smoke) python -m mode_control.mc.activations smoke ;;
  acts)  python -m mode_control.mc.activations AB --batch 8 ;;
  acts_late) python -m mode_control.mc.activations AB --batch 8 --k "${K:-192}" ;;
  actsA) python -m mode_control.mc.activations A --batch 8 ;;
  actsB) python -m mode_control.mc.activations B --batch 8 ;;
  steer) RUN=${1:?run}; CONDS=${2:-}; python -m mode_control.mc.steer --run "$RUN" ${CONDS:+--conditions "$CONDS"} --batch "${BATCH:-32}" ;;
  judge) RUN=${1:?run}; CONDS=${2:-${CONDS:-}}; JD="${JDIR:-$DD/runs/$RUN/judge/job$SLURM_JOB_ID}"; mkdir -p "$JD"   # JDIR=<existing dir> resumes a timed-out judge
         GLOB="$DD/runs/$RUN/traces_*.parquet"
         if [ -n "${CONDS:-}" ]; then mkdir -p "$JD/in"; for c in ${CONDS//,/ }; do ln -sf "$PWD/$DD/runs/$RUN/traces_$c.parquet" "$JD/in/traces_$c.parquet"; done; GLOB="$JD/in/traces_*.parquet"; fi
         python -m src.judge.run_judge --config mode_control/configs/exp_mode.yaml --judge-model google/gemma-4-31B-it \
           --track "${TRACK:-both}" --traces-glob "$GLOB" --out-dir "$JD" --track-b-limit 0
         python -m mode_control.mc.analyze --run "$RUN" || echo "analyze failed" ;;
  quality) RUN=${1:?run}; python -m src.judge.run_quality --config mode_control/configs/exp_mode.yaml --judge-model google/gemma-4-31B-it \
           --traces-glob "$DD/runs/$RUN/traces_*.parquet" --out-dir "$DD/runs/$RUN/judge" ;;
  pilot) RUN=${1:?run}; WAVE=${2:-full}; CONDS=${3:-${CONDS:-}}; TAG=${TAG:-_v2}   # conds as 3rd positional (--export splits on commas)
         # Disjoint condition subsets (CONDS) may run in parallel jobs; each job judges into its own subdir.
         [ -f $DD/acts/stageA_k64.npz ] || python -m mode_control.mc.activations A --batch 8
         [ -f $DD/acts/stageB_k32.npz ] || python -m mode_control.mc.activations B --batch 8
         [ -f "$DD/directions/layer_phase_map$TAG.csv" ] || python -m mode_control.mc.directions --window "${WINDOW:-4096}" --tag "$TAG"
         python -m mode_control.mc.plots --window "${WINDOW:-4096}" --tag "$TAG" || echo "plots failed (non-fatal)"
         [ -f "$DD/runs/$RUN/run.json" ] || python -m mode_control.mc.make_run --run "$RUN" --layer auto --window "${WINDOW:-4096}" --n-per-domain "${NPD:-40}" --wave "$WAVE" --tag "$TAG"
         python -m mode_control.mc.steer --run "$RUN" --batch "${BATCH:-32}" ${CONDS:+--conditions "$CONDS"}
         JD="$DD/runs/$RUN/judge/job$SLURM_JOB_ID"; mkdir -p "$JD"
         GLOB="$DD/runs/$RUN/traces_*.parquet"
         if [ -n "${CONDS:-}" ]; then mkdir -p "$JD/in"; for c in ${CONDS//,/ }; do ln -sf "$PWD/$DD/runs/$RUN/traces_$c.parquet" "$JD/in/traces_$c.parquet"; done; GLOB="$JD/in/traces_*.parquet"; fi
         python -m src.judge.run_judge --config mode_control/configs/exp_mode.yaml --judge-model google/gemma-4-31B-it \
           --track both --traces-glob "$GLOB" --out-dir "$JD" --track-b-limit 0
         python -m mode_control.mc.analyze --run "$RUN" || echo "analyze failed" ;;
  diversity) RUN=${1:?run}; python -m mode_control.mc.method_judge --run "$RUN" || echo "method judge failed"
             python -m mode_control.mc.diversity --run "$RUN" ;;
  similarity) RUN=${1:?run}; python -m mode_control.mc.similarity --run "$RUN" ;;
  complementarity) RUN=${1:?run}; python -m mode_control.mc.complementarity --run "$RUN" ${ARMS:+--arms "$ARMS"} --no-embed ;;
  similarity_arms) RUN=${1:?run}; python -m mode_control.mc.similarity --run "$RUN" --arms "${ARMS:?arms}" ;;
  sigacts) RUNS=${1:?runs}; python -m mode_control.mc.signature_ctrl extract --runs "$RUNS" --batch "${BATCH:-2}" ;;
  sigfit) RUNS=${1:?runs}; TAG=${2:-v1}; python -m mode_control.mc.signature_ctrl fit --runs "$RUNS" --tag "$TAG" --lams "${LAMS:-10,100,1000,10000}" --ranks "${RANKS:-1,2,4,8}" ${DEMEAN:+--demean-domain} ;;
  sigeval) RUN=${1:?run}; python -m mode_control.mc.sigeval --run "$RUN" ;;
  mapfit) RUNS=${1:?runs}; TAG=${2:?tag}; python -m mode_control.mc.signature_ctrl mapfit --runs "$RUNS" --tag "$TAG" --layer "${LAYER:-16}" --rank "${RANK:-8}" --lam "${LAM:-100000}" ${DEMEAN:+--demean-domain} ;;
  vprep) python -m mode_control.mc.vsteer seal --layers 16,19,22,25
         for L in 16 19 22 25; do python -m mode_control.mc.vsteer sealvec --name seal_L$((L + 1)) --layer-out $L; done
         python -m mode_control.mc.vsteer sealvec --name sealdir_L17_n6p75 --layer-out 16 --norm 6.75
         python -m mode_control.mc.vsteer check --a2 ol_math_A2 --seal seal_L23 ;;
  vcheck) python -m mode_control.mc.vsteer check --a2 ol_math_A2 --seal "${1:-seal_L23}" ;;
  judgeB) RUN=${1:?run}; GLOB=${2:?glob}; JD="$DD/runs/$RUN/judge/trackB_${3:-sub}"; mkdir -p "$JD"   # Track B only, fixed dir so a requeue resumes
         python -m src.judge.run_judge --config "${JCFG:-mode_control/configs/exp_mode.yaml}" --judge-model google/gemma-4-31B-it --track B --traces-glob "$GLOB" --out-dir "$JD" --track-b-limit 0 ;;
  rltrain) RUN=${1:?run}; shift; python -m mode_control.mc.rlbias train --run "$RUN" "$@" ;;
  methods) RUN=${1:?run}; python -m mode_control.mc.method_judge --run "$RUN" ;;
  p3dir) python -m mode_control.mc.vsteer promptdir --run p3_strat --cond prompt_figure --name pdir_figure --seeds 1,2 --scales 1,2
         python -m mode_control.mc.vsteer promptdir --run p3_strat --cond prompt_cases --name pdir_cases --seeds 1,2 --scales 1,2 ;;
  mcextract) SH=${1:?shard}; NS=${2:?num_shards}; mkdir -p mode_control/data/extract
         python -m src.judge.run_answer_extraction --config configs/exp.yaml --judge-model google/gemma-4-31B-it            --traces-glob "mode_control/data/extract_in/traces_*.parquet" --out-dir mode_control/data/extract --shard "$SH" --num-shards "$NS" ;;
  xdiag) python /ada1/u/harsh/tmp/extract_diag.py ;;
  freex) python -m mode_control.mc.extract_free extract --shard "${1:?shard}" --num-shards "${2:?n}" ;;
  freex2) XF_IN=mode_control/data/extract_in2 XF_OUT=mode_control/data/extract_free2 python -m mode_control.mc.extract_free extract --shard "${1:?shard}" --num-shards "${2:?n}" ;;
  qualityc) RUN=${1:?run}; C=${2:?cond}   # quality judge for one condition file (smaller unit for ashton_low preemption)
         python -m src.judge.run_quality --config mode_control/configs/exp_mode.yaml --judge-model google/gemma-4-31B-it \
           --traces-glob "$DD/runs/$RUN/traces_$C.parquet" --out-dir "$DD/runs/$RUN/judge/quality_$C" ;;
  f1gen) M=${1:?model}; SH=${2:?shard}; NS=${3:?nshards}   # D-50: atlas generator on the minimal-pair tasks, one replica per job
         MASTER_ADDR=127.0.0.1 MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000)) VLLM_PORT=$((40000 + SLURM_JOB_ID % 10000)) \
         python -m src.generate.generate_traces --config mode_control/configs/exp_f1.yaml --gen-model "$M" --tasks-dir mode_control/data/f1/tasks \
           --out-dir mode_control/data/f1/traces --shard "$SH" --num-shards "$NS" ;;
  f1gen2) M=${1:?model}   # D-50 noise floor: second sample (seed 43) of the solve_free and generate prompts
         MASTER_ADDR=127.0.0.1 MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000)) VLLM_PORT=$((40000 + SLURM_JOB_ID % 10000)) \
         python -m src.generate.generate_traces --config mode_control/configs/exp_f1_s2.yaml --gen-model "$M" --tasks-dir mode_control/data/f1/tasks_s2 \
           --out-dir mode_control/data/f1/traces_s2 ;;
  f1acts) MC_MODEL=${1:?model} python -m mode_control.mc.f1_pairs acts ;;
  freexg) XF_IN=${1:?in_dir} XF_OUT=${2:?out_dir} python -m mode_control.mc.extract_free extract --shard "${3:?shard}" --num-shards "${4:?n}" ;;
  freexa) XF_IN=data/v2/extract_free_atlas/in XF_OUT=data/v2/extract_free_atlas/out python -m mode_control.mc.extract_free extract --shard "${1:?shard}" --num-shards "${2:?n}" ;;
  vgen) RUN=${1:?run}; CONDS=${2:-}; python -m mode_control.mc.vsteer gen --run "$RUN" ${CONDS:+--conditions "$CONDS"} ;;
  budget) RUN=${1:?run}; python -m mode_control.mc.budget --run "$RUN" ;;
  readiness) python -m mode_control.mc.readiness --arms "${1:?arms}" ${CUTS:+--cuts "$CUTS"} ${SMOKE:+--smoke "$SMOKE"} ;;   # D-62 forced-answer probe
  neuro) MODE=${1:?mode}; shift; python -m mode_control.mc.neuro_extract "$MODE" "$@" ;;   # D-55..D-58: smoke | corpus | prompts | taskset
  neuro_an) python -m mode_control.mc.neuro_analyze "${1:-all}" ;;
  neuro_sae) CMD=${1:?cmd}; shift; python -m mode_control.mc.neuro_sae "$CMD" "$@" ;;   # D-60 (fetch runs on ada, not in a job)
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
date
