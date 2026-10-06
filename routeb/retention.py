"""Bound storage only for new single-adapter runs, never legacy checkpoints."""
import json
from pathlib import Path
import re
import shutil


def prune_intermediate(folder, protocol, steps_per_epoch, keep_recent=2):
    if protocol.get('architecture') not in ('single-gaze-adapter-v2', 'single-gaze-adapter-fovlanding-v1') or keep_recent < 2:
        raise ValueError('Retention requires an identified single-adapter run and at least two recovery states')
    folder = Path(folder)
    candidates = []
    for path in folder.glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d{6})-v(\d{3})', path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Checkpoint must be a real directory')
        manifest = json.loads((path / 'manifest.json').read_text())
        update, validated = map(int, match.groups())
        if manifest['metadata']['protocol'] != protocol or manifest['update'] != update:
            raise ValueError('Refusing retention across protocols or corrupt counters')
        if manifest['metadata']['progress']['validated_epochs'] != validated:
            raise ValueError('Invalid validation counter')
        candidates.append((update, validated, path, manifest))
    candidates.sort()
    keep = {path.name for _, _, path, _ in candidates[-keep_recent:]}
    for update, validated, path, manifest in candidates:
        if update == 0 or (validated > 0 and update == validated * steps_per_epoch):
            keep.add(path.name)
        best = manifest['metadata']['progress'].get('best_checkpoint')
        if best:
            keep.add(Path(best).name)
    retired = folder / 'retired-manifests'
    removed = []
    for _, _, path, manifest in candidates:
        if path.name in keep:
            continue
        retired.mkdir(exist_ok=True)
        # Preserve the exact original manifest, before deleting only this state.
        destination = retired / (path.name + '.json')
        if destination.exists():
            raise FileExistsError('Retention record already exists')
        destination.write_bytes((path / 'manifest.json').read_bytes())
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
