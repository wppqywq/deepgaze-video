"""Fetch the pinned base and official free-viewing assets inside Slurm."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = "OpenGVLab/InternVL3_5-8B-HF"
REVISION = "741a7d03020411e666c6109218ab71e08151ef86"
COMMIT = "5cd84d2225beb92e77617b1ee5c5480cae948ab7"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run in Slurm')
    root = Path(os.environ['ROUTEB_STORAGE']).resolve()
    evidence = Path(os.environ['ROUTEB_CODE']) / 'evidence'
    vendor = root / 'vendor/DeepGaze3.5-VL'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(vendor), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != COMMIT or git('status', '--porcelain'):
        raise RuntimeError('Official checkout differs from the pinned version')
    if shutil.disk_usage(root).free < 100 * 1024**3:
        raise RuntimeError('Less than the planned 100 GiB free storage headroom')
    lfs_expected = {}
    for line in git('lfs', 'ls-files', '--long').splitlines():
        oid, _, name = line.split(maxsplit=2)
        if name.startswith(('model/combined_adapter/', 'data/')):
            lfs_expected[name] = oid
    if 'model/combined_adapter/adapter_model.safetensors' not in lfs_expected:
        raise RuntimeError('Combined adapter missing from LFS manifest')
    subprocess.run(['git', '-C', str(vendor), 'lfs', 'pull',
                    '--include=model/combined_adapter/**,data/**', '--exclude='], check=True)
    for name, expected in lfs_expected.items():
        if digest(vendor / name) != expected:
            raise RuntimeError('LFS content mismatch: ' + name)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(BASE, revision=REVISION)
    if info.sha != REVISION:
        raise RuntimeError('Base revision mismatch')
    model = Path(snapshot_download(repo_id=BASE, revision=REVISION,
                 local_dir=root / 'models/InternVL3_5-8B-HF',
                 max_workers=min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))))
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards:
        raise RuntimeError('No model shards in index')
    hashes = {}
    for name in ['config.json', 'model.safetensors.index.json', *shards]:
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()) or not path.is_file():
            raise RuntimeError('Invalid model shard path: ' + name)
        hashes[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'base_repo': BASE,
              'base_revision': info.sha, 'base_directory': str(model),
              'official_commit': COMMIT, 'verified_lfs_sha256': lfs_expected,
              'base_files': hashes, 'status': 'Pinned weights downloaded; inference untested'}
    (evidence / ('weights-' + result['job_id'] + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
