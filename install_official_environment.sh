#!/usr/bin/env bash
# Run via srun after reviewing the bootstrap result and pinned requirements.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${ROUTEB_STORAGE:?Set personal scratch}"
: "${ROUTEB_CODE:?Set code directory}"
umask 077
export PIP_CACHE_DIR="$ROUTEB_STORAGE/pip-cache"
export TMPDIR="$ROUTEB_STORAGE/tmp"
export HF_HOME="$ROUTEB_STORAGE/hf-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-2}"
python="$ROUTEB_STORAGE/envs/official/bin/python"
vendor="$ROUTEB_STORAGE/vendor/DeepGaze3.5-VL"
test "$(git -C "$vendor" rev-parse HEAD)" = 5cd84d2225beb92e77617b1ee5c5480cae948ab7
test -z "$(git -C "$vendor" status --porcelain)"
cmp "$vendor/requirements.txt" "$ROUTEB_CODE/evidence/official-requirements.txt"
"$python" -c 'import sys; assert sys.version_info[:2] == (3,12)'
cuda_packages=(torch==2.9.0+cu128 torchvision==0.24.0+cu128)
if [[ "${ROUTEB_INSTALL_PROFILE:-official}" == official ]]; then
  cuda_packages+=(torchaudio==2.9.0+cu128)
fi
"$python" -m pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$ROUTEB_CODE/constraints-cu128.txt" \
  --report "$ROUTEB_CODE/evidence/torch-install-$SLURM_JOB_ID.json" \
  "${cuda_packages[@]}"
requirements=( -r "$vendor/requirements.txt" )
if [[ "${ROUTEB_INSTALL_PROFILE:-official}" == hf ]]; then
  # The project's exact HF scoring/training path does not import vLLM.
  requirements=( -r "$ROUTEB_CODE/requirements.txt" )
fi
"$python" -m pip install --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c "$ROUTEB_CODE/constraints-cu128.txt" "${requirements[@]}" \
  --report "$ROUTEB_CODE/evidence/official-install-$SLURM_JOB_ID.json"
"$python" -m pip check
"$python" -m pip freeze > "$ROUTEB_CODE/evidence/official-environment-$SLURM_JOB_ID.txt"
if [[ "${ROUTEB_CPU_INSTALL:-0}" == 1 ]]; then
  "$python" -c 'import torch, transformers, peft, av; print(torch.__version__, transformers.__version__, peft.__version__, av.__version__); print("GPU validation pending")'
else
  "$python" "$ROUTEB_CODE/cuda_probe.py"
fi
