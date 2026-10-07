#!/bin/bash
cd /ada1/u/harsh/thought-atlas
source /ada1/u/harsh/miniconda3/bin/activate sote
export HF_HUB_OFFLINE=1 HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface OMP_NUM_THREADS=24
MC_MODEL=reasoner python -m mode_control.mc.directions --window 4096 --tag _k192 --stageA mode_control/data/reasoner/acts/stageA_k192.npz --stageB mode_control/data/reasoner/acts/stageB_k192.npz > /ada1/u/harsh/tmp/r1_directions.log 2>&1
MC_MODEL=qwen35_9b python -m mode_control.mc.directions --window 4096 --tag _k192 --stageA mode_control/data/acts/stageA_k192.npz --stageB mode_control/data/acts/stageB_k192.npz > /ada1/u/harsh/tmp/qwen_directions_k192.log 2>&1
