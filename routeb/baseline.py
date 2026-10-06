"""B1 on server PNGs with cached ResNet18 features and complete epoch exports.

This is a new preprocessing protocol, not reproduction of the Mac video decoder.
The architecture and cell likelihood remain those of the original baseline.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18

from baseline import Population, cell_log_mass, tensors, parameters
from gpu_train import file_hash, require, write_json
from routeb.data import VideoInputs, FovlandingInputs
from routeb.evaluation import macro_film_bits
from stage0 import grid_log_mass, gmm_grid

TRANSFORM = 'materialized-480x270-PNG; PIL-bilinear224x126; black-pad224x224-y49; ImageNet; layer4-pool2x2-v2'
FOV_TRANSFORM = 'native-video-PNG; PIL-bilinear-aspect-fit224x224-black-pad; ImageNet; layer4-pool2x2-fovlanding-v1'


def features(paths, weights, cache_root, device, fovlanding=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    weight_hash = file_hash(weights)
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    encoder = nn.Sequential(*list(net.children())[:-2]).requires_grad_(False).eval().to(device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    result, hashes = {}, {}
    for video in sorted({video for video, _ in paths}):
        indices = sorted(index for clip, index in paths if clip == video)
        provenance = {'transform': FOV_TRANSFORM if fovlanding else TRANSFORM, 'weights': weight_hash,
                      'images': {str(index): file_hash(paths[video, index]) for index in indices}}
        file = cache_root / (video + '.npz')
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                require(json.loads(str(saved['provenance'])) == provenance and
                        saved['indices'].tolist() == indices, 'Feature cache provenance differs')
                values = saved['features']
        else:
            values = []
            with torch.inference_mode():
                for start in range(0, len(indices), 32):
                    batch = []
                    for index in indices[start:start + 32]:
                        with Image.open(paths[video, index]) as source:
                            if fovlanding:
                                frame = source.convert('RGB')
                                canvas = ImageOps.pad(frame, (224, 224), method=Image.Resampling.BILINEAR, color='black')
                            else:
                                require(source.size == (480, 270), 'Unexpected materialized image geometry')
                                frame = source.convert('RGB').resize((224, 126), Image.Resampling.BILINEAR)
                                canvas = Image.new('RGB', (224, 224))
                                canvas.paste(frame, (0, 49))
                        batch.append(np.asarray(canvas, dtype=np.float32).copy())
                        frame.close()
                        canvas.close()
                    pixels = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device) / 255
                    encoded = encoder((pixels - mean) / std)
                    values.append(nn.functional.adaptive_avg_pool2d(encoded, (2, 2)).flatten(1).cpu().numpy())
            values = np.concatenate(values).astype(np.float32)
            temporary = file.with_suffix('.incomplete')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, features=values, indices=indices, provenance=json.dumps(provenance))
            os.replace(temporary, file)
        require(values.shape == (len(indices), 2048) and np.isfinite(values).all(), 'Invalid feature cache')
        result.update({(video, index): value for index, value in zip(indices, values)})
        hashes[file.name] = file_hash(file)
        print(json.dumps({'stage': 'features', 'video': video, 'frames': len(indices)}), flush=True)
    return result, hashes


def fovlanding_tensors(inputs, variant, cache=None):
    """Approved gaze fields and identical causal frame plans; no observer input."""
    from scripts.compare_video import history_features
    from fovlanding import serialize_target
    targets = {row['target_id']: row for row in inputs.targets}
    ordered = [targets[key] for split in ('train', 'validation') for key in inputs.ids[split]]
    packed = history_features(ordered, 4, True)
    # GRU consumes events oldest first; no last-sample time or event-type input.
    h = packed[:, :24].reshape(-1, 4, 6)[:, ::-1].astype(np.float32)
    hm = packed[:, 24:][:, ::-1].astype(np.float32)
    v = np.zeros((len(ordered), 4, 2048), np.float32)
    vm = np.zeros_like(hm)
    vt = np.zeros_like(hm)
    rows, paths = [], {}
    for n, target in enumerate(ordered):
        audit = target['audit']
        video = audit['video_id']
        serialize_target(target, variant.upper(), inputs.paths[video])
        plans = target['model_input']['frame_plans'][variant.upper()]
        for i, frame in enumerate(plans, 4 - len(plans)):
            key = (video, frame['source_frame_index'])
            paths[key] = Path(inputs.paths[video][key[1]])
            if cache is not None:
                v[n, i] = cache[key]
            vm[n, i] = 1
            vt[n, i] = float(f"{frame['relative_ms']:.3f}") / 1000
        rows.append({'id': target['target_id'], 'video': video, 'source_group': audit['source_film_id'],
                     'split': audit['split'], 'target': {'cell': target['target']['xy_grid']}})
    if cache is None:
        return rows, paths
    train = np.array([row['split'] == 'train' for row in rows])
    history, visual = h[train][hm[train].astype(bool)], v[train][vm[train].astype(bool)]
    norm = {'hm': history.mean(0), 'hs': np.maximum(history.std(0), 1e-3),
            'vm': visual.mean(0), 'vs': np.maximum(visual.std(0), 1e-3)}
    h = ((h - norm['hm']) / norm['hs']) * hm[..., None]
    v = ((v - norm['vm']) / norm['vs']) * vm[..., None]
    arrays = [h, hm, v, vm, vt, np.array([row['target']['cell'] for row in rows], np.float32)]
    return [torch.from_numpy(a) for a in arrays], norm, train


def export_heatmaps(model, data, validation_ids, rows, run, device):
    """Selected-model cell masses, aligned by ID; float64 likelihood stays separate."""
    maps, mixtures, effective = [], [], []
    examples = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation_ids), 128):
            ids = validation_ids[start:start + 128]
            output = model(*[array[ids].to(device) for array in data[:5]])
            weights, means, scales = [value.cpu().numpy() for value in parameters(output.double())]
            scores = cell_log_mass(output, data[5][ids].to(device)).cpu().numpy()
            for index, w, m, s, score in zip(ids, weights, means, scales, scores):
                grid = gmm_grid(w, m, s)
                x, y = rows[index]['target']['cell']
                require(np.isfinite(grid).all() and abs(grid.sum()-1) < 1e-10, 'Invalid AR heatmap')
                if score > -80:
                    require(abs(math.log(grid[int(y), int(x)])-score) < 1e-6, 'Heatmap/likelihood mismatch')
                maps.append(grid.astype(np.float32))
                mixtures.append(np.c_[w, m, s])
                effective.append(float(np.exp(-np.sum(w * np.log(np.maximum(w, 1e-300))))))
                film = rows[index]['source_group']
                if film not in examples:
                    name = f'heatmap-example-{len(examples)+1}.png'
                    # Linear display scale; raw normalized probabilities remain in NPZ.
                    gray = np.rint(255 * grid / grid.max()).astype(np.uint8)
                    preview = Image.fromarray(gray).resize((400, 400), Image.Resampling.NEAREST)
                    preview.save(run / name)
                    examples[film] = {'target_id': rows[index]['id'], 'path': name,
                                      'display_scale': 'grayscale; 0 black, sample maximum white'}
    path = run / 'best-heatmaps.npz'
    np.savez_compressed(path, target_ids=np.array([rows[i]['id'] for i in validation_ids]),
        probabilities=np.stack(maps), mixture_parameters=np.stack(mixtures))
    return {'file': path.name, 'sha256': file_hash(path), 'shape': [len(maps), 100, 100],
            'axes': 'target,y,x; each grid sums to one; float32 may underflow in far tails',
            'mixture_columns': 'weight,mean_x,mean_y,scale_x,scale_y; before rectangle conditioning',
            'effective_components': {'definition': 'exp(entropy of input-dependent mixture weights), before truncation',
                'mean': float(np.mean(effective)), 'min': min(effective), 'max': max(effective)},
            'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--input-protocol', choices=['video-v2', 'fovlanding-v1'], default='video-v2')
    parser.add_argument('--scan-report', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True, help='ResNet18 IMAGENET1K_V1 state dict')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--variant', choices=['d1', 'd4', 'r4'], default='d4')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Run inside Slurm')
    require(args.epochs > 0 and args.weights.is_file(), 'Positive epochs and local ResNet18 weights required')
    require(not args.run.exists(), 'Use a new B1 run directory')
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    torch.manual_seed(args.seed)
    fovlanding = args.input_protocol == 'fovlanding-v1'
    inputs = (FovlandingInputs if fovlanding else VideoInputs)(args.outputs, args.scan_report)
    rows, paths = fovlanding_tensors(inputs, args.variant) if fovlanding else inputs.baseline_rows(args.variant)
    started = time.monotonic()
    cache, feature_hashes = features(paths, args.weights, args.cache, args.device, fovlanding)
    data, normalization, train = fovlanding_tensors(inputs, args.variant, cache) if fovlanding else tensors(rows, cache)
    del cache
    args.run.mkdir(parents=True)
    model = Population(history_dimensions=6 if fovlanding else 5).to(args.device)
    root = Path(__file__).resolve().parent.parent
    protocol = {'architecture': 'resnet18-gru-mdn-fovlanding-v1' if fovlanding else 'resnet18-gru-mdn-v2', 'variant': args.variant,
                'inputs': inputs.provenance, 'transform': FOV_TRANSFORM if fovlanding else TRANSFORM,
                'epochs': args.epochs, 'seed': args.seed, 'batch_size': 128, 'learning_rate': .001,
                'feature_sha256': feature_hashes, 'weights_sha256': file_hash(args.weights),
                'trainable_parameters': sum(p.numel() for p in model.parameters()),
                'source_sha256': {name: file_hash(root / name) for name in
                                  ['routeb/baseline.py', 'routeb/data.py', 'baseline.py', 'stage0.py',
                                   'fovlanding.py', 'scripts/compare_video.py', 'routeb/evaluation.py']}}
    if fovlanding:
        protocol.update(scope='development_assumptions', scientific_calibration_verified=False,
            selection='validation film-macro; earliest epoch on ties',
            reference='Youyou_proposal140926.pdf, section 3.2, pages 3-4; population spatial subset',
            components={'count': 5, 'adaptive_count': False, 'input_dependent': ['weights', 'means', 'diagonal scales']},
            limitations=['No subject embedding, event-type/duration heads, optical flow or scene-cut input.',
                         'Four-event bounded history; five diagonal Gaussian components; pooled 2x2 frozen ResNet18 map.',
                         'Different visual representation/capacity and training budget from DeepGaze; not an image ablation.'])
    write_json(args.run / 'protocol.json', protocol)
    np.savez(args.run / 'normalization.npz', **normalization)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(args.seed)
    train_ids, validation_ids = np.flatnonzero(train), np.flatnonzero(~train)
    curves, best_bits, best_epoch = [], -math.inf, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.
        order = rng.permutation(train_ids)
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            h, hm, v, vm, vt, y = [array[ids].to(args.device) for array in data]
            optimizer.zero_grad(set_to_none=True)
            loss = -cell_log_mass(model(h, hm, v, vm, vt), y).mean()
            require(bool(torch.isfinite(loss)), 'Nonfinite B1 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(validation_ids), 128):
                ids = validation_ids[start:start + 128]
                h, hm, v, vm, vt, y = [array[ids].to(args.device) for array in data]
                scores.extend((cell_log_mass(model(h, hm, v, vm, vt), y) / math.log(2)).cpu().tolist())
        output = args.run / f'validation-epoch-{epoch:03d}.jsonl'
        with output.open('x') as stream:
            for index, bits in zip(validation_ids, scores):
                sample = rows[index]
                prior = float(grid_log_mass(inputs.prior, sample['target']['cell']) / math.log(2))
                row = {key: sample[key] for key in ['id', 'video', 'source_group']}
                row.update(bits=bits, prior_bits=prior, ig_bits=bits - prior)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        mean_bits = math.fsum(scores) / len(scores)
        entry = {'epoch': epoch, 'mean_bits': mean_bits, 'train_nll_nats': loss_sum / len(train_ids),
                 'predictions': output.name, 'predictions_sha256': file_hash(output)}
        if fovlanding:
            films = {}
            for index, bits in zip(validation_ids, scores):
                films.setdefault(rows[index]['source_group'], []).append(bits)
            entry['by_film'] = {film: {'targets': len(values), 'mean_bits': math.fsum(values) / len(values)}
                                for film, values in films.items()}
            entry['selection_bits'] = macro_film_bits(entry['by_film'])
        curves.append(entry)
        # B1 states are small: keep every epoch for complete learning-curve comparisons.
        torch.save(model.state_dict(), args.run / f'epoch-{epoch:03d}.pt')
        entry['checkpoint_sha256'] = file_hash(args.run / f'epoch-{epoch:03d}.pt')
        selection_bits = entry.get('selection_bits', mean_bits)
        if selection_bits > best_bits:
            best_bits, best_epoch = selection_bits, epoch
        write_json(args.run / 'progress.json', {'epochs': curves, 'best_epoch': best_epoch, 'best_bits': best_bits})
        print(json.dumps(entry), flush=True)
    heatmaps = None
    if fovlanding:
        model.load_state_dict(torch.load(args.run / f'epoch-{best_epoch:03d}.pt', map_location=args.device, weights_only=True))
        heatmaps = export_heatmaps(model, data, validation_ids, rows, args.run, args.device)
    write_json(args.run / 'result.json', {'status': 'complete', 'epochs': curves, 'best_epoch': best_epoch,
                                        'heatmaps': heatmaps,
                                        'best_bits': best_bits, 'elapsed_seconds': time.monotonic() - started,
                                        'protocol_sha256': file_hash(args.run / 'protocol.json')})


if __name__ == '__main__':
    main()
