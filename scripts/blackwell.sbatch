#!/bin/bash
# scripts/blackwell.sbatch — SLURM template for vega/mira (Blackwell, sm_120).
# Submit from ada:  sbatch scripts/blackwell.sbatch
# Override per-stage: sbatch --job-name=judge --export=ALL,STAGE=judge scripts/blackwell.sbatch
#
# Confirmed cluster facts (May 2026): ashton partition spans grace,vega,mira.
#   - gres is PLAIN: --gres=gpu:N  (NOT typed gpu:rtx_pro_6000:N)
#   - --time REQUIRED (DefaultTime=NONE) ; PreemptMode=REQUEUE -> add --requeue + checkpoint
#   - -w vega|mira to PIN to a Blackwell node (else a small job may land on grace=A6000=wrong arch)
#   - --gres=gpu:4 packs within one NUMA island (0-3); whole box gpu:8 spans both islands
#   - local NVMe is /vega (or /mira): 14TB, fast, NOT backed up. Stage hot data here.
#SBATCH --job-name=sote
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH -w vega                       # pin to Blackwell; change to mira to use the other box
#SBATCH --gres=gpu:4                  # 4 = one NUMA island, no manual pinning needed
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00               # REQUIRED on this partition
#SBATCH --requeue                     # PreemptMode=REQUEUE -> make the job resumable
#SBATCH --output=outputs/%x_%j.log
#SBATCH --error=outputs/%x_%j.err

set -euo pipefail

cleanup() {
  rc=$?
  pkill -9 -P $$ 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    pkill -9 -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Per-rank thread budget = cpus_per_task / num_gpus (avoid oversubscription)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/ada1/u/harsh/.cache/huggingface          # writable HF home (Xet derives its dirs from this)
export HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface     # personal blob cache (writable)
export HF_HUB_DISABLE_XET=1                               # Xet tries to write logs under HF_HOME -> skip it
# FlashInfer JIT-caches .so files compiled against a CUDA version; a stale cu13 cache fails to
# load against cu129 (libcudart.so.13 missing). Use vLLM's native sampler (plenty fast) and keep
# any JIT cache on writable /ada1, not the tiny home quota.
export VLLM_USE_FLASHINFER_SAMPLER=0
# ALL cache dirs default to $HOME/.cache (= /h/<user>, tiny student quota). HF_HOME only moves
# weights; vLLM/FlashInfer/Triton/Inductor each have separate cache envs -> redirect them to /ada1
# or compiled artifacts (incl. the cu13 JIT .so that bit us) pile up in home and break.
export XDG_CACHE_HOME=/ada1/u/harsh/.cache                # catch-all base for ~/.cache fallbacks
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm          # vLLM torch.compile / artifact cache
export FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer
export TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
# Shared lab cache stays available read-only; vLLM/HF will fall back to downloading any model
# not already in the personal cache. (Pre-stage big models with: hf download <id> to this HF_HOME.)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export NCCL_P2P_DISABLE=1   # uncomment if PCIe P2P misbehaves on TP
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export XDG_CONFIG_HOME=/ada1/u/harsh/.config
export MPLCONFIGDIR=/ada1/u/harsh/.cache/matplotlib
export TMPDIR=/ada1/u/harsh/tmp
mkdir -p /ada1/u/harsh/.config /ada1/u/harsh/.cache/matplotlib /ada1/u/harsh/tmp

pwd; hostname; date
nvidia-smi
nvidia-smi topo -m || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Stage hot data onto local NVMe before the dataloader iterates (NFS is slow for random reads).
# Pick the local NVMe for whichever box we landed on; fall back to NFS scratch if the local
# per-user dir isn't writable yet (e.g. /vega/u/$USER not provisioned — ask Joseph to create it).
NODE="$(hostname -s)"
LOCAL="/${NODE}/u/${USER}/sote"                 # -> /vega/u/<you>/sote or /mira/u/<you>/sote
if ! mkdir -p "$LOCAL"/{data,runs,scratch} 2>/dev/null; then
  echo "WARN: local NVMe scratch $LOCAL not writable; falling back to NFS scratch." >&2
  LOCAL="/ada1/u/${USER}/society-task-exp2/.scratch"
  mkdir -p "$LOCAL"/{data,runs,scratch}
fi
echo "scratch: $LOCAL"
# rsync -a /ada1/u/${USER}/society-task-exp2/data/ "$LOCAL/data/"   # enable per-stage when staging helps

set +u   # conda hook/activate references unbound vars; re-enable -u after
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"; conda activate sote
set -u

# Args: positional first (sbatch blackwell.sbatch generate anchor for Llama-3.1-8B-Instruct),
# env-var override second.
STAGE="${1:-${STAGE:-help}}"
MODEL="${2:-${MODEL:-}}"        # for generate: config model key; for judge: the judge model id
echo "STAGE=$STAGE MODEL=${MODEL:-<none>}"
case "$STAGE" in
  generate)
    [ -n "$MODEL" ] || { echo "generate needs a model key, e.g. anchor=Llama-3.1-8B-Instruct or reasoner=DeepSeek-R1-Distill-Llama-8B"; exit 2; }
    # True data-parallel: one single-GPU vLLM replica per allocated GPU, each over a dataset shard.
    # (vLLM 0.22 dropped in-process data_parallel_size; replicate-don't-shard is the right pattern
    #  for an 8B model that fits one card anyway.)
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
    NG=${#GPUS[@]}
    echo "data-parallel: $NG replicas of $MODEL, one per GPU [${CUDA_VISIBLE_DEVICES}]"
    pids=()
    for i in $(seq 0 $((NG-1))); do
      # Each replica = independent single-GPU engine. Give each its own distributed-init port
      # so the 4 engine-cores don't collide on the rendezvous endpoint.
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + i * 13)) VLLM_PORT=$((29500 + i * 13 + 1)) \
      python -m src.generate.generate_traces \
        --config configs/exp.yaml --gen-model "$MODEL" --shard "$i" --num-shards "$NG" \
        > "outputs/${SLURM_JOB_ID}_${MODEL}_shard${i}.log" 2>&1 &
      pids+=($!)
      sleep 3   # small stagger so the 4 cores don't race on init
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    # concat shard parquets -> canonical traces_<model>.parquet (downstream also accepts the glob)
    python -m src.generate.merge_shards --model "$MODEL" --out-dir data/traces || true
    exit $rc ;;
  segment)  python -m src.segment.segmenter --in data/traces/traces.parquet --out data/segments/segments.parquet ;;
  judge)
    [ -n "$MODEL" ] || { echo "judge needs a judge-model id: sbatch blackwell.sbatch judge <hf_id> [track]"; exit 2; }
    TRACK="${3:-A}"   # default Track A (cheap); pass B or both for the per-sentence pass
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
    NG=${#GPUS[@]}
    echo "judge data-parallel: $NG replicas, model=$MODEL track=$TRACK"
    pids=()
    for i in $(seq 0 $((NG-1))); do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29600 + i * 13)) VLLM_PORT=$((29600 + i * 13 + 1)) \
      python -m src.judge.run_judge --config configs/exp.yaml --judge-model "$MODEL" \
        --track "$TRACK" --traces-glob "data/traces/traces_*.parquet" --out-dir data/judge/prod \
        --shard "$i" --num-shards "$NG" \
        > "outputs/${SLURM_JOB_ID}_judge_shard${i}.log" 2>&1 &
      pids+=($!); sleep 3
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    exit $rc ;;
  quality)
    # moral/idea SUCCESS scoring (rubric quality of the ANSWER ONLY, not reasoning).
    # Separate from `judge` (behavior labels). Same DP fan-out; distinct port base (29700).
    [ -n "$MODEL" ] || { echo "quality needs a judge-model id: sbatch blackwell.sbatch quality <hf_id>"; exit 2; }
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
    NG=${#GPUS[@]}
    echo "quality data-parallel: $NG replicas, model=$MODEL (moral+idea, answer-only)"
    pids=()
    for i in $(seq 0 $((NG-1))); do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29700 + i * 13)) VLLM_PORT=$((29700 + i * 13 + 1)) \
      python -m src.judge.run_quality --config configs/exp.yaml --judge-model "$MODEL" \
        --traces-glob "data/traces/traces_*.parquet" --out-dir data/judge/prod \
        --shard "$i" --num-shards "$NG" \
        > "outputs/${SLURM_JOB_ID}_quality_shard${i}.log" 2>&1 &
      pids+=($!); sleep 3
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    exit $rc ;;
  analyze)
    [ -n "$MODEL" ] || { echo "analyze needs the judge tag: sbatch blackwell.sbatch analyze <judge_tag>"; exit 2; }
    python scripts/run_analysis.py --config configs/exp.yaml --judge-tag "$MODEL" ;;
  *) echo "usage: sbatch blackwell.sbatch {generate <model>|segment|judge <id> [A|B|both]|quality <id>|analyze <tag>} (got '$STAGE')"; exit 2 ;;
esac
