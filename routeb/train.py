"""Single-adapter video training or unadapted evaluation; one bounded job, no chaining."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
from importlib.metadata import version

import torch
from transformers import AutoProcessor

from gpu_train import file_hash, frozen_digest, packed, require, seed_all, write_json
from routeb.checkpoints import checkpoint_name, latest_checkpoint, quarantine_incomplete
from routeb.data import PROMPT_VERSION, FovlandingInputs, VideoInputs
from routeb.evaluation import evaluate
from routeb.model import eval_mass, load_model, log_mass
from routeb.retention import prune_intermediate
from training_state import SampleOrder, mean_backward, restore_checkpoint, save_checkpoint, trainable_fingerprint


def learning_rate_multiplier(update, config):
    if config.get('scheduler', 'constant') == 'constant':
        return 1.
    total = config['max_optimizer_steps']
    warmup = int(total * config['warmup_fraction'])
    if update < warmup:
        return (update + 1) / max(1, warmup)
    fraction = min(1., (update - warmup) / max(1, total - warmup))
    return .5 * (1 + math.cos(math.pi * fraction))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--scan-report', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=['d1', 'd4', 'r4'], required=True)
    parser.add_argument('--action', choices=['train', 'static'], default='train')
    parser.add_argument('--segment-updates', type=int, default=200)
    parser.add_argument('--max-seconds', type=int, default=10000)
    parser.add_argument('--input-protocol', choices=['video-v2', 'fovlanding-v1'], default='video-v2')
    parser.add_argument('--base', type=Path, help='Relocated, hash-verified base model')
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Run inside Slurm')
    # Allow full fixed-budget runs within the site's 48-hour allocation limit.
    require(args.segment_updates > 0 and 300 <= args.max_seconds <= 171600, 'Invalid segment budget')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUDA required')
    require(torch.cuda.device_count() == 1, 'One allocated CUDA device required')
    config = json.loads(args.config.read_text())
    fovlanding = args.input_protocol == 'fovlanding-v1'
    require(config['schema'] == (3 if fovlanding else 2) and config['deterministic_algorithms'], 'Invalid configuration')
    budget = config['max_optimizer_steps'] if fovlanding else config['maximum_epochs']
    require(budget > 0 and config['effective_batch'] > 0 and
            config['checkpoint_every_updates'] > 0, 'Invalid training configuration')
    if fovlanding:
        require(config['validation_every_updates'] > 0 and
                budget % config['validation_every_updates'] == 0 and
                config['checkpoint_selection'] == 'validation_macro_film_log2p' and
                config['scheduler'] == 'cosine' and 0 <= config['warmup_fraction'] < 1,
                'Invalid fixed-budget validation/scheduler settings')
    inputs = (FovlandingInputs if fovlanding else VideoInputs)(args.outputs, args.scan_report)
    initialization = json.loads((args.initialization / 'initialization.json').read_text())
    for name, expected in initialization['weights_sha256'].items():
        path = args.base / Path(name).name if args.base and Path(name).is_absolute() else args.initialization / name
        require(file_hash(path) == expected, 'Initialization artifact changed: ' + name)
    require(config['max_length'] == inputs.scan['max_length'], 'Context limit differs from input audit')
    root = Path(__file__).resolve().parent.parent
    sources = ['routeb/train.py', 'routeb/data.py', 'routeb/model.py', 'routeb/evaluation.py',
               'routeb/checkpoints.py', 'routeb/retention.py', 'gpu_train.py', 'training_state.py',
               'vlm_check.py', 'stage0.py']
    if fovlanding:
        sources += ['fovlanding.py', 'scripts/check_fovlanding.py']
    protocol = {'architecture': 'single-gaze-adapter-fovlanding-v1' if fovlanding else 'single-gaze-adapter-v2',
                'prompt': 'fovlanding-v1' if fovlanding else PROMPT_VERSION,
                'variant': args.variant, 'action': args.action, 'config': config,
                'inputs': inputs.provenance, 'initialization': str(args.initialization.resolve()),
                'initialization_sha256': file_hash(args.initialization / 'initialization.json'),
                'source_sha256': {name: file_hash(root / name) for name in sources},
                'runtime': {name: version(name) for name in ['torch', 'transformers', 'peft']}}
    args.run.mkdir(parents=True, exist_ok=True)
    lock = (args.run / '.writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.run / 'protocol.json'
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol, 'Existing run protocol differs')
    else:
        require(not any(args.run.glob('checkpoints/*')), 'Unidentified existing checkpoints')
        write_json(protocol_path, protocol)
    job = args.run / 'jobs' / os.environ['SLURM_JOB_ID']
    job.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed_all(config['seed'], deterministic=True)
    torch.set_num_threads(1 if fovlanding else min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    processor = AutoProcessor.from_pretrained(args.initialization, local_files_only=True)
    model = load_model(args.initialization, trainable=args.action == 'train', base=args.base)
    frozen_before = frozen_digest(model) if fovlanding and args.action == 'train' else None
    write_json(job / 'resources.json', {'device': torch.cuda.get_device_name(0),
        'precision': 'BF16 base, FP32 LoRA', 'micro_batch': 1, 'effective_batch': config['effective_batch'],
        'cpu_torch_threads': torch.get_num_threads(),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_tokens': inputs.scan.get('input_tokens'), 'digit_ids': inputs.scan.get('digit_ids'),
        'peak_allocated_bytes_after_loading': torch.cuda.max_memory_allocated()})
    validation = inputs.rows(args.variant, 'validation')

    def score(epoch):
        output = job / f'validation-epoch-{epoch:03d}.jsonl'
        report = evaluate(model, processor, inputs, args.variant, range(len(validation)),
                          output, rows=validation, score_fn=eval_mass)
        report.update(epoch=epoch, predictions=str(output.relative_to(args.run)))
        if fovlanding:
            report.update(validation_index=epoch, optimizer_update=epoch * config['validation_every_updates'],
                          seed=config['seed'], run_id=args.run.name, protocol_version='fovlanding-v1',
                          adapter_fingerprint=trainable_fingerprint(model),
                          scientific_calibration_verified=False)
        path = output.with_suffix('.json')
        write_json(path, report)
        return {'epoch': epoch, 'mean_bits': report['mean_bits'],
                'selection_bits': report['macro_film_bits'] if fovlanding else report['mean_bits'],
                'macro_film_bits': report['macro_film_bits'],
                'report': str(path.relative_to(args.run)), 'sha256': file_hash(path)}

    if args.action == 'static':
        require(not (args.run / 'result.json').exists(), 'Static evaluation already complete')
        evaluation = score(0)
        write_json(args.run / 'result.json', {'status': 'complete', 'evaluation': evaluation,
                                            'protocol_sha256': file_hash(protocol_path)})
        return

    rows = inputs.rows(args.variant, 'train')
    steps = math.ceil(len(rows) / config['effective_batch'])
    interval = config['validation_every_updates'] if fovlanding else steps
    maximum_updates = budget if fovlanding else budget * steps
    maximum_validations = maximum_updates // interval
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_multiplier(step, config))
    order = SampleOrder(range(len(rows)), config['seed'])
    checkpoints = args.run / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    quarantine_incomplete(checkpoints, os.environ['SLURM_JOB_ID'])
    latest = latest_checkpoint(checkpoints, protocol)
    progress = {'validated_epochs': 0, 'best_epoch': None, 'best_bits': None,
                'best_checkpoint': None, 'evaluations': [], 'seen_targets': 0,
                'epoch_loss_sum': 0., 'epoch_targets': 0, 'initial_evaluation': None}
    update, last_saved = 0, None
    if latest:
        last_saved, manifest = latest
        update = restore_checkpoint(last_saved, model, optimizer, scheduler, order, manifest['metadata'])
        progress = copy.deepcopy(manifest['metadata']['progress'])

    def persist():
        nonlocal last_saved
        path = checkpoints / checkpoint_name(update, progress['validated_epochs'])
        if path != last_saved:
            save_checkpoint(path, model, optimizer, scheduler, order, update,
                            {'protocol': protocol, 'progress': copy.deepcopy(progress)})
            last_saved = path
        write_json(args.run / 'latest.json', {'update': update, 'checkpoint': str(path.relative_to(args.run)),
                                            'progress': progress})
        prune_intermediate(checkpoints, protocol, interval)

    persist()
    if fovlanding and update == 0 and progress['initial_evaluation'] is None:
        initial_path = args.run / 'initial-evaluation.json'
        if initial_path.exists():
            initial = json.loads(initial_path.read_text())
            require(file_hash(args.run / initial['report']) == initial['sha256'], 'Initial evaluation changed')
            progress['initial_evaluation'] = initial
        else:
            progress['initial_evaluation'] = score(0)
            write_json(initial_path, progress['initial_evaluation'])
    start_update = update
    with (job / 'updates.jsonl').open('x') as stream:
        while True:
            completed = update // interval
            if completed > progress['validated_epochs']:
                require(completed == progress['validated_epochs'] + 1 and (fovlanding or order.cursor == len(rows)),
                        'Epoch boundary mismatch')
                # Conservative reserve based on the old full-validation measurements.
                reserve = 1500 if args.variant == 'd1' else 3000
                if args.max_seconds - (time.monotonic() - started) < reserve:
                    break
                evaluation = score(completed)
                require(fovlanding or progress['epoch_targets'] == len(rows), 'Incomplete training epoch')
                progress['evaluations'].append(evaluation)
                progress['validated_epochs'] = completed
                if progress['best_bits'] is None or evaluation['selection_bits'] > progress['best_bits']:
                    progress.update(best_epoch=completed, best_bits=evaluation['selection_bits'],
                                    best_checkpoint='checkpoints/' + checkpoint_name(update, completed))
                progress.update(epoch_targets=0, epoch_loss_sum=0.)
                persist()
            if update >= maximum_updates or update - start_update >= args.segment_updates:
                break
            if args.max_seconds - (time.monotonic() - started) < 180:
                break
            model.train()
            # Frozen visual features must not acquire train-mode randomness.
            model.get_base_model().model.vision_tower.eval()
            model.get_base_model().model.multi_modal_projector.eval()
            batch = order.next_batch(config['effective_batch'])
            optimizer.zero_grad(set_to_none=True)
            before = time.monotonic()
            loss = mean_backward(lambda i: -log_mass(model, packed(processor, rows[i], config['max_length'])), batch)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['clip_norm'], error_if_nonfinite=True)
            require(float(norm) > 0, 'No adapter gradient')
            if fovlanding:
                require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen weight acquired a gradient')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            progress['seen_targets'] += len(batch)
            progress['epoch_targets'] += len(batch)
            progress['epoch_loss_sum'] += loss * len(batch)
            entry = {'update': update, 'indices': batch, 'loss_nats': loss,
                     'seconds': time.monotonic() - before, 'learning_rate_next_update': optimizer.param_groups[0]['lr'],
                     'gradient_norm': float(norm)}
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            print(json.dumps(entry), flush=True)
            if update % config['checkpoint_every_updates'] == 0 or order.cursor == len(rows):
                persist()
    persist()
    if frozen_before is not None:
        require(frozen_digest(model) == frozen_before, 'Frozen weights changed')
    result = {'status': 'complete' if progress['validated_epochs'] == maximum_validations
                                   else 'segment_complete', 'update': update, 'progress': progress,
              'elapsed_seconds': time.monotonic() - started,
              'protocol_sha256': file_hash(protocol_path),
              'device': torch.cuda.get_device_name(0),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'trainable_parameters': sum(p.numel() for p in parameters)}
    if fovlanding:
        result.update(scope='development_assumptions', scientific_calibration_verified=False,
                      frozen_weights_unchanged=True, validation_unit='100-update checks, not epochs')
    write_json(job / 'result.json', result)
    if result['status'] == 'complete':
        write_json(args.run / 'result.json', result)


if __name__ == '__main__':
    main()
