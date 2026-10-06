#!/usr/bin/env bash
# Called via srun; all downloads and installs stay inside the allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${ROUTEB_STORAGE:?Set the personal scratch directory}"
: "${ROUTEB_CODE:?Set the code directory}"
umask 077
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PIP_CACHE_DIR="$ROUTEB_STORAGE/pip-cache"
export UV_CACHE_DIR="$ROUTEB_STORAGE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROUTEB_STORAGE/python"
export UV_PYTHON_BIN_DIR="$ROUTEB_STORAGE/bin"
export HF_HOME="$ROUTEB_STORAGE/hf-cache"
export TMPDIR="$ROUTEB_STORAGE/tmp"
mkdir -p "$TMPDIR" "$ROUTEB_STORAGE/envs" "$ROUTEB_STORAGE/vendor" "$UV_PYTHON_BIN_DIR"
bootstrap="$ROUTEB_STORAGE/envs/bootstrap"
official="$ROUTEB_STORAGE/envs/official"
if [[ ! -x "$bootstrap/bin/uv" ]]; then
  # LRZ system Python lacks ensurepip; bootstrap uv without a system venv.
  python3 -m pip install --disable-pip-version-check --upgrade --target "$bootstrap" \
    --only-binary=:all: --index-url https://pypi.org/simple uv==0.8.22
fi
uv="$bootstrap/bin/uv"
"$uv" --version
"$uv" python install 3.12.11
if [[ ! -e "$official" ]]; then
  "$uv" venv --python 3.12.11 "$official"
fi
"$official/bin/python" -c 'import sys; assert sys.version_info[:3] == (3,12,11); print(sys.version)'
"$uv" pip install --python "$official/bin/python" --index-url https://pypi.org/simple pip==25.2

commit=5cd84d2225beb92e77617b1ee5c5480cae948ab7
vendor="$ROUTEB_STORAGE/vendor/DeepGaze3.5-VL"
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -e "$vendor" ]]; then
  git init "$vendor"
  git -C "$vendor" remote add origin https://github.com/Susmit-A/DeepGaze3.5-VL.git
  git -C "$vendor" fetch --depth=1 origin "$commit"
  git -C "$vendor" checkout --detach "$commit"
fi
test "$(git -C "$vendor" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$vendor" status --porcelain)"
test -f "$vendor/requirements.txt"
git -C "$vendor" rev-parse HEAD > "$ROUTEB_CODE/evidence/deepgaze-commit.txt"
cp "$vendor/requirements.txt" "$ROUTEB_CODE/evidence/official-requirements.txt"
{
  printf 'Job=%s\nCompleted=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)"
  "$uv" --version
  "$official/bin/python" --version
  "$official/bin/python" -m pip --version
  printf 'DeepGaze commit=%s\n' "$commit"
  printf 'CUDA and model dependencies not installed or verified.\n'
} > "$ROUTEB_CODE/evidence/bootstrap-$SLURM_JOB_ID.txt"
cat "$ROUTEB_CODE/evidence/bootstrap-$SLURM_JOB_ID.txt"
