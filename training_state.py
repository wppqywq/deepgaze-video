"""Optimizer-boundary checkpoints and epoch-local sample batches.

Checkpoints are trusted local artifacts: torch.load restores Python/NumPy RNG
objects. Never pass a checkpoint from an untrusted source to restore_checkpoint.
No mid-accumulation or Slurm signal recovery is claimed here.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


class SampleOrder:
    """Own RNG for shuffling; retain a short final batch within each epoch."""

    def __init__(self, indices, seed):
        self.indices = list(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError('Sample indices must be nonempty and unique')
        self.rng = random.Random(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = self.indices.copy()
        self.rng.shuffle(self.order)

    def next_batch(self, batch_size):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = self.indices.copy()
            self.rng.shuffle(self.order)
        end = min(self.cursor + batch_size, len(self.order))
        result = self.order[self.cursor:end]
        self.cursor = end
        return result

    def state_dict(self):
        return {'indices': self.indices.copy(), 'order': self.order.copy(),
                'epoch': self.epoch, 'cursor': self.cursor, 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['indices'] != self.indices or sorted(state['order']) != sorted(self.indices):
            raise ValueError('Checkpoint sample population differs')
        if not 0 <= state['cursor'] <= len(self.indices) or state['epoch'] < 0:
            raise ValueError('Invalid epoch/cursor')
        self.order = state['order'].copy()
        self.epoch = state['epoch']
        self.cursor = state['cursor']
        self.rng.setstate(state['rng'])


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def trainable_fingerprint(model):
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            result.update(name.encode())
            result.update(str(parameter.dtype).encode())
            result.update(str(tuple(parameter.shape)).encode())
            result.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def save_checkpoint(path, model, optimizer, scheduler, order, update, metadata):
    """Publish a complete new directory only at a completed update boundary."""
    path = Path(path)
    temporary = path.with_name(path.name + '.incomplete')
    if path.exists() or temporary.exists():
        raise FileExistsError('Refusing to overwrite checkpoint or incomplete save')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Clear gradients after optimizer.step before saving')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    state = {'schema': 1, 'update': update, 'metadata': metadata,
             'trainable': trainable_state(model), 'optimizer': cpu_copy(optimizer.state_dict()),
             'scheduler': copy.deepcopy(scheduler.state_dict()),
             'sample_order': order.state_dict(), 'rng': rng_state()}
    if not state['trainable'] or update < 0:
        raise ValueError('Invalid trainable state/update')
    torch.save(state, temporary / 'training-state.pt')
    # PEFT export is for evaluation; training-state.pt carries exact resume data.
    model.save_pretrained(temporary / 'adapter', safe_serialization=True)
    hashes = {str(p.relative_to(temporary)): digest(p)
              for p in sorted(temporary.rglob('*')) if p.is_file()}
    manifest = {'schema': 1, 'update': update, 'metadata': metadata, 'sha256': hashes,
                'boundary': 'after optimizer/scheduler step and zero_grad(set_to_none=True)'}
    (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    os.rename(temporary, path)
    return manifest


def restore_checkpoint(path, model, optimizer, scheduler, order, metadata):
    """Load into a freshly constructed matching model; restore RNG last."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema'] != 1 or manifest['metadata'] != metadata:
        raise ValueError('Checkpoint configuration/provenance differs')
    if 'training-state.pt' not in manifest['sha256']:
        raise ValueError('Checkpoint lacks training state')
    for name, expected in manifest['sha256'].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or digest(target) != expected:
            raise ValueError('Checkpoint file hash/path mismatch: ' + name)
    state = torch.load(path / 'training-state.pt', map_location='cpu', weights_only=False)
    if state['schema'] != 1 or state['metadata'] != metadata or state['update'] != manifest['update']:
        raise ValueError('Checkpoint manifest/state mismatch')
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state['trainable'].keys():
        raise ValueError('Trainable parameter inventory differs')
    for name, param in parameters.items():
        saved = state['trainable'][name]
        if param.shape != saved.shape or param.dtype != saved.dtype:
            raise ValueError('Trainable parameter shape/dtype differs: ' + name)
    with torch.no_grad():
        for name, param in parameters.items():
            param.copy_(state['trainable'][name])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    order.load_state_dict(state['sample_order'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state['rng'])
    return state['update']


def mean_backward(loss_for_index, indices):
    """Microbatch=1; normalize by the actual batch count, including epoch tail."""
    if not indices:
        raise ValueError('Empty effective batch')
    total = 0.0
    for index in indices:
        loss = loss_for_index(index)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite or nonscalar loss')
        (loss / len(indices)).backward()
        total += float(loss.detach())
    return total / len(indices)
