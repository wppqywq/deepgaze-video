#!/usr/bin/env bash
#SBATCH --job-name=fovlanding
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?}"
storage="${ROUTEB_STORAGE:?Set the project storage directory}"
initialization="${ROUTEB_INITIALIZATION:-$storage/models/gaze-initialization}"
mode="${1:?Pass setup, audit, smoke, overfit, build, export, check, train, baseline, ar, ar-weights, compare, aggregate, deduplicate, resume, or replicate}"
shift
umask 077
run="$storage/runs/$mode-$SLURM_JOB_ID"
mkdir -p "$run/code/evidence" "$storage/tmp"
if [[ "$mode" == resume || "$mode" == replicate ]]; then
  original="${1:?Pass the original training directory}"
  inputs="${2:?Pass its original input export directory}"
  shift 2
  cp -r "$(dirname "$original")/code/." "$run/code/"
  printf '%s\n' "$original" > "$run/source-training.txt"
else
  cp ./*.py ./*.sh constraints-cu128.txt requirements.txt "$run/code/"
  cp -r routeb scripts tests configs "$run/code/"
fi
cp "$0" "$run/submission.sbatch"
git rev-parse HEAD > "$run/parent-commit.txt"
git diff --binary > "$run/worktree.patch"
git status --short > "$run/worktree-status.txt"
export ROUTEB_STORAGE="$storage" ROUTEB_CODE="$run/code"
export TMPDIR="$storage/tmp" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$storage/hf-cache" HF_HUB_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python="$storage/envs/official/bin/python"
cd "$run/code"
case "$mode" in
  setup)
    export ROUTEB_INSTALL_PROFILE=hf ROUTEB_CPU_INSTALL=1
    srun bash bootstrap_environment.sh
    srun bash install_official_environment.sh
    ;;
  audit)
    srun "$python" -m unittest tests.test_fovlanding
    srun "$python" scripts/check_fovlanding.py audit --storage "$storage" \
      --recovered "${ROUTEB_RECOVERED:?Set the recovered provenance directory}" --output "$run/audit" "$@"
    ;;
  smoke)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    srun "$python" cuda_probe.py
    srun "$python" scripts/check_fovlanding.py smoke --storage "$storage" --initialization "$initialization" --output "$run/smoke" "$@"
    ;;
  overfit)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    srun "$python" scripts/check_fovlanding.py overfit --storage "$storage" --initialization "$initialization" --output "$run/overfit" "$@"
    ;;
  build)
    audit_path="${1:?Pass the completed development audit directory}"
    srun "$python" fovlanding.py --config "$audit_path/development-config.json" \
      --trials "$audit_path/development-pilot-trials.jsonl" --output "$run/targets"
    ;;
  export)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    srun "$python" -m unittest tests.test_fovlanding
    srun "$python" scripts/check_fovlanding.py export --storage "$storage" --initialization "$initialization" --output "$run/inputs" "$@"
    ;;
  train)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    inputs="${1:?Pass the completed input export directory}"
    variant="${2:?Pass d1, r4, or d4}"
    shift 2
    srun "$python" -m routeb.train --input-protocol fovlanding-v1 \
      --initialization "$initialization" \
      --base "$storage/models/InternVL3_5-8B-HF" \
      --outputs "$inputs" --scan-report "$inputs/scan.json" \
      --config configs/fovlanding_train.json --variant "$variant" \
      --run "$run/training" --segment-updates 1000 --max-seconds 42000 "$@"
    ;;
  baseline)
    srun "$python" scripts/compare_video.py --input-protocol fovlanding-v1 \
      --center-bias --output "$run/comparison" "$@"
    ;;
  ar-weights)
    srun "$python" - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
from torchvision.models import ResNet18_Weights
root = Path(os.environ['ROUTEB_STORAGE']) / 'models'
path = root / 'resnet18-f37072fd.pth'
url = ResNet18_Weights.IMAGENET1K_V1.url
if not path.exists():
    temporary = path.with_suffix('.incomplete')
    torch.hub.download_url_to_file(url, temporary, hash_prefix='f37072fd')
    os.replace(temporary, path)
with path.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
if not digest.startswith('f37072fd'):
    raise ValueError('ResNet18 weight hash mismatch')
record = {'url': url, 'sha256': digest, 'weights': 'ResNet18_Weights.IMAGENET1K_V1'}
(root / 'resnet18-source.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
PY
    ;;
  deduplicate)
    srun "$python" scripts/deduplicate_static.py --report "$run/deduplication.json" "$@"
    ;;
  ar)
    inputs="${1:?Pass the completed input export directory}"
    shift
    srun "$python" -m routeb.baseline --input-protocol fovlanding-v1 \
      --outputs "$inputs" --scan-report "$inputs/scan.json" \
      --weights "$storage/models/resnet18-f37072fd.pth" \
      --cache "$storage/cache/ar-fovlanding-v1" --run "$run/training" "$@"
    ;;
  compare)
    for selection in best final; do
      options=()
      if [[ "$selection" == best ]]; then options+=(--best); fi
      srun "$python" scripts/compare_video.py --input-protocol fovlanding-v1 \
        --output "$run/$selection" "${options[@]}" "$@"
    done
    ;;
  aggregate)
    inputs="${1:?Pass the common input export directory}"
    shift
    for selection in best final; do
      options=()
      for directory in "$@"; do options+=(--seed-comparison "$directory/$selection"); done
      srun "$python" scripts/compare_video.py --input-protocol fovlanding-v1 \
        --outputs "$inputs" --output "$run/$selection" "${options[@]}"
    done
    ;;
  resume)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python" -c 'import json,sys; p=json.load(open(sys.argv[1])); open(sys.argv[2], "w").write(json.dumps(p["config"], indent=2)+"\n")' \
      "$original/protocol.json" "$run/training-config.json"
    mapfile -t saved < <("$python" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["initialization"]); print(p["variant"])' "$original/protocol.json")
    srun "$python" -m routeb.train --input-protocol fovlanding-v1 \
      --initialization "${saved[0]}" --variant "${saved[1]}" \
      --base "$storage/models/InternVL3_5-8B-HF" \
      --outputs "$inputs" --scan-report "$inputs/scan.json" \
      --config "$run/training-config.json" --run "$original" \
      --segment-updates 1000 --max-seconds 19800 "$@"
    ;;
  replicate)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    seed="${1:?Pass a new nonnegative training seed}"
    shift
    "$python" - "$original" "$inputs" "$run" "$seed" <<'PY'
import hashlib, json, sys
from pathlib import Path
original, inputs, run = map(Path, sys.argv[1:4])
seed = int(sys.argv[4])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
protocol = json.loads((original / 'protocol.json').read_text())
result = json.loads((original / 'result.json').read_text())
assert result['status'] == 'complete' and result['protocol_sha256'] == sha(original / 'protocol.json')
assert protocol['architecture'] == 'single-gaze-adapter-fovlanding-v1' and protocol['action'] == 'train'
assert seed >= 0 and seed != protocol['config']['seed']
assert sha(inputs / 'scan.json') == protocol['inputs']['scan_sha256']
for name, expected in protocol['source_sha256'].items():
    assert sha(run / 'code' / name) == expected, 'Original training source changed: ' + name
config = dict(protocol['config'], seed=seed)
(run / 'training-config.json').write_text(json.dumps(config, indent=2) + '\n')
(run / 'replication.json').write_text(json.dumps({'source_protocol_sha256': sha(original / 'protocol.json'),
    'source_run': str(original), 'variant': protocol['variant'], 'seed': seed,
    'config_change': {'seed': [protocol['config']['seed'], seed]},
    'training_source_sha256': protocol['source_sha256']}, indent=2) + '\n')
PY
    if [[ "${1:-}" == --prepare-only ]]; then
      cat "$run/replication.json"
      exit 0
    fi
    mapfile -t saved < <("$python" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["initialization"]); print(p["variant"])' "$original/protocol.json")
    srun "$python" -m routeb.train --input-protocol fovlanding-v1 \
      --initialization "${saved[0]}" --variant "${saved[1]}" \
      --base "$storage/models/InternVL3_5-8B-HF" \
      --outputs "$inputs" --scan-report "$inputs/scan.json" \
      --config "$run/training-config.json" --run "$run/training" \
      --segment-updates 1000 --max-seconds 19800 "$@"
    ;;
  check)
    srun "$python" -m unittest discover -s tests
    ;;
  *) echo 'Pass setup, audit, smoke, overfit, build, export, check, train, baseline, ar, ar-weights, compare, aggregate, deduplicate, resume, or replicate' >&2; exit 2 ;;
esac
