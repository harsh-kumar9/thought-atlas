#!/bin/bash
#SBATCH --job-name=atlas_gpu_probe
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --nodelist=mira
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --requeue
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf 'Allocated GPU: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,uuid,driver_version --format=csv
/ada1/u/harsh/miniconda3/envs/sote/bin/python - <<'PY'
import os,json,torch
from vllm.platforms import current_platform
assert torch.cuda.is_available()
assert torch.cuda.device_count()==1
x=torch.ones((8,8),device='cuda');assert float((x@x).sum())==512
cap=current_platform.get_device_capability(0)
assert cap is not None
print(json.dumps({'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':torch.cuda.get_device_name(0),'cuda_compute_verified':True,'vllm_nvml_capability':str(cap),'scope':'Assigned GPU only; no model or scientific run.'}))
PY
