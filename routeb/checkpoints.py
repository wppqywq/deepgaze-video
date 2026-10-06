"""Pilot checkpoint discovery: only atomically published complete saves count."""
import json
from pathlib import Path
import re


def quarantine_incomplete(folder, job_id):
    """Preserve interrupted writes so replay can publish the same update safely."""
    folder = Path(folder)
    moved = []
    for path in sorted(folder.glob('checkpoint-*.incomplete')):
        if not re.fullmatch(r'checkpoint-\d{6}-v\d{3}\.incomplete', path.name):
            raise RuntimeError('Unexpected incomplete checkpoint name')
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError('Unexpected incomplete checkpoint object')
        destination = folder / 'interrupted' / (path.name + '-found-by-' + str(job_id))
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError('Interrupted checkpoint archive already exists')
        path.rename(destination)
        moved.append(str(destination.relative_to(folder)))
    return moved


def checkpoint_name(update, validated_epochs):
    return 'checkpoint-{:06d}-v{:03d}'.format(update, validated_epochs)


def latest_checkpoint(folder, protocol):
    candidates = []
    for path in Path(folder).glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d{6})-v(\d{3})', path.name)
        if match is None:
            continue  # e.g. a killed writer's .incomplete directory
        if not (path / 'manifest.json').is_file():
            raise RuntimeError('Published checkpoint lacks manifest: ' + str(path))
        manifest = json.loads((path / 'manifest.json').read_text())
        update, validated = map(int, match.groups())
        if manifest['metadata']['protocol'] != protocol:
            raise RuntimeError('Checkpoint protocol/source/input mismatch: ' + str(path))
        progress = manifest['metadata']['progress']
        if manifest['update'] != update or progress['validated_epochs'] != validated:
            raise RuntimeError('Checkpoint directory/state counters differ')
        candidates.append((update, validated, path, manifest))
    if not candidates:
        return None
    _, _, path, manifest = max(candidates, key=lambda item: item[:2])
    return path, manifest
