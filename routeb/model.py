"""One trainable copy of the official gaze adapter; no merging or extra LoRA."""
import json
from pathlib import Path

import torch
from peft import PeftConfig, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

from gpu_train import load_base, require
from vlm_check import constrained_log_mass


def load_model(initialization, adapter=None, trainable=True, base=None):
    initialization = Path(initialization)
    manifest = json.loads((initialization / 'initialization.json').read_text())
    require(manifest['representation'] == 'frozen_static_lora', 'Use the preserved official initialization')
    source = Path(adapter) if adapter is not None else initialization / 'static'
    config = PeftConfig.from_pretrained(source, local_files_only=True)
    require(config.r == 32 and config.lora_alpha == 64 and config.bias == 'none', 'Unexpected gaze adapter')
    config.inference_mode = False
    model = get_peft_model(load_base(Path(base if base is not None else manifest['source_base'])), config)
    # Allocate FP32 before loading, avoiding BF16 rounding of adapter tensors.
    names = []
    for name, parameter in model.named_parameters():
        is_adapter = '.lora_A.default.' in name or '.lora_B.default.' in name
        require(not is_adapter or 'language_model.layers.' in name, 'Adapter outside the language model')
        parameter.requires_grad_(trainable and is_adapter)
        if is_adapter:
            parameter.data = parameter.data.float()
            names.append(name)
    require(len(names) == model.config.text_config.num_hidden_layers * 7 * 2, 'Unexpected adapter inventory')
    weights = load_peft_weights(str(source), device='cpu', local_files_only=True)
    result = set_peft_model_state_dict(model, weights, adapter_name='default')
    require(not result.unexpected_keys and not any('.lora_' in name for name in result.missing_keys),
            'Incomplete adapter load')
    require(list(model.peft_config) == ['default'], 'Expected exactly one adapter')
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.eval()
    return model


def log_mass(model, item, full_logits=False):
    tensors, positions, digit_ids = item
    indices = torch.tensor(positions, device=tensors['input_ids'].device) - 1
    if full_logits:
        logits = model(**tensors, use_cache=False, return_dict=True).logits[0, indices]
    else:
        # Official Transformers interface avoids all-position vocabulary projection.
        logits = model(**tensors, use_cache=False, return_dict=True, logits_to_keep=indices).logits[0]
    mass = constrained_log_mass(logits, tensors['input_ids'][0, positions], digit_ids)
    require(bool(torch.isfinite(mass)), 'Nonfinite cell probability')
    return mass


@torch.no_grad()
def eval_mass(model, item):
    model.eval()
    return float(log_mass(model, item))
