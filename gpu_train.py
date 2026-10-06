"""Shared model loading, exact scoring, and training checks."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoModelForImageTextToText
from vlm_check import constrained_log_mass, prepare

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def load_base(path):
    model = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    require(all(p.device.type == 'cuda' for p in model.parameters()), 'CPU/offloaded parameters')
    return model


def packed(processor, row, max_length):
    tensors, positions, digits = prepare(processor, row)
    require(tensors['input_ids'].shape[1] <= max_length, 'Context exceeds frozen limit')
    require(min(positions) >= 1 and len(positions) == 4, 'Invalid causal digit positions')
    tensors = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
               for k, v in tensors.items()}
    return tensors, positions, digits


def predictions(model, tensors, positions):
    # Full-vocabulary teacher forcing; no labels/default SFT loss.
    outputs = model(**tensors, use_cache=False, return_dict=True)
    return outputs.logits[0, torch.tensor(positions, device='cuda') - 1, :]


def log_mass(model, item):
    tensors, positions, digits = item
    logits = predictions(model, tensors, positions)
    target = tensors['input_ids'][0, positions]
    mass = constrained_log_mass(logits, target, digits)
    require(bool(torch.isfinite(mass)), 'Non-finite target mass')
    return mass


@torch.no_grad()
def eval_mass(model, item):
    model.eval()
    return float(log_mass(model, item))


def frozen_digest(model):
    """Exact full frozen-parameter check, chunked to avoid extra model copies."""
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        digest.update(name.encode())
        raw = param.detach().contiguous().reshape(-1).view(torch.uint8)
        for part in raw.split(8 * 1024 * 1024):
            digest.update(part.cpu().numpy().tobytes())
    return digest.hexdigest()
