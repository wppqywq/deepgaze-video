"""Deduplicate immutable adapter snapshots without changing paths or bytes.

Run only after training has stopped. Default is a read-only plan. Apply uses
hash-verified same-filesystem hard links and preserves checkpoint manifests.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deduplicate(campaign=None, apply=False, training_runs=None):
    if (campaign is None) == (not training_runs):
        raise ValueError('Provide a legacy campaign or explicit completed training runs')
    variants = Path(campaign).resolve() / 'variants' if campaign is not None else None
    folders = sorted(variants.iterdir()) if variants else sorted(Path(p).absolute() for p in training_runs)
    if len(set(folders)) != len(folders):
        raise ValueError('Duplicate training run')
    report = {'campaign': str(variants.parent) if variants else None,
              'training_runs': [str(p) for p in folders] if training_runs else [],
              'applied': apply, 'files': [], 'reclaimed_bytes': 0}
    with ExitStack() as stack:
        protocols = {}
        for folder in folders:
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError('Unexpected variant directory')
            lock = stack.enter_context((folder / '.writer.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if training_runs:
                protocol = json.loads((folder / 'protocol.json').read_text())
                result = json.loads((folder / 'result.json').read_text())
                if (protocol.get('architecture') != 'single-gaze-adapter-fovlanding-v1' or
                        result.get('status') != 'complete' or
                        result.get('protocol_sha256') != digest(folder / 'protocol.json') or
                        result.get('update') != protocol['config']['max_optimizer_steps'] or
                        result['progress']['validated_epochs'] != result['update'] // protocol['config']['validation_every_updates'] or
                        result.get('frozen_weights_unchanged') is not True):
                    raise ValueError('Only fully validated completed fovlanding runs may be deduplicated')
                protocols[folder] = protocol
        canonical = {}
        manifests = [(folder, path) for folder in folders
                     for path in sorted((folder / 'checkpoints').glob('checkpoint-*/manifest.json'))]
        for folder, manifest_path in manifests:
            if '.incomplete' in str(manifest_path):
                continue
            name = 'adapter/adapter_model.safetensors' if training_runs else 'adapter/static/adapter_model.safetensors'
            manifest = json.loads(manifest_path.read_text())
            if training_runs and (manifest['metadata']['protocol'] != protocols[folder] or name not in manifest['sha256']):
                raise ValueError('Checkpoint protocol or adapter inventory differs')
            if name not in manifest['sha256']:
                continue
            path = manifest_path.parent / name
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Adapter path escapes its run')
            expected = manifest['sha256'][name]
            if digest(path) != expected:
                raise ValueError('Adapter hash mismatch')
            info = path.stat()
            key = (expected, info.st_size, info.st_dev, info.st_uid, info.st_gid, info.st_mode)
            if key not in canonical:
                canonical[key] = path
                continue
            source = canonical[key]
            if path.samefile(source):
                continue
            # Count actual blocks released only if this was the inode's last link.
            released = info.st_blocks * 512 if info.st_nlink == 1 else 0
            entry = {'path': str(path), 'canonical': str(source), 'sha256': expected,
                     'reclaimable_bytes': released}
            report['files'].append(entry)
            if apply:
                temporary = path.with_name(path.name + '.deduplicate')
                if temporary.exists():
                    raise FileExistsError('Unfinished deduplication exists')
                os.link(source, temporary)
                try:
                    if path.stat() != info:
                        raise RuntimeError('Adapter changed during deduplication')
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                report['reclaimed_bytes'] += released
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--campaign', type=Path)
    sources.add_argument('--training-run', type=Path, action='append')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run hashing/file processing inside Slurm')
    if args.report.exists():
        parser.error('Use a new report path')
    result = deduplicate(args.campaign, args.apply, args.training_run)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))
