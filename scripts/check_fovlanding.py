"""Bounded development audit and static-model checks for fovlanding-v1."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fovlanding import build_trial_targets, load_config, read_jsonl, serialize_target


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit(args):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from scipy.io import loadmat

    root, output = args.storage, args.output
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads(args.preparation.read_text())
    trials_path = Path(preparation['standardized_trials']['path'])
    if digest(trials_path) != preparation['standardized_trials']['sha256']:
        raise ValueError('Standardized trial content changed')
    dataset_audit = json.loads(args.dataset_audit.read_text())
    raw_manifest = args.raw_manifest
    if digest(raw_manifest) != dataset_audit['trial_manifest']['sha256']:
        raise ValueError('Audited raw manifest changed')
    raw_hashes = {row['source_file']: row['raw_sha256'] for row in read_jsonl(raw_manifest)}
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/fovlanding_v1.json').read_text())
    split = json.loads((args.recovered / 'mapping_handoff/original/fixed-splits.json').read_text())
    alignment = json.loads((args.recovered / 'mapping_handoff/original/alignment-check.json').read_text())
    event_path = args.recovered / 'confirmation_events/development-events.json'
    saved_events = json.loads(event_path.read_text())
    expected = dataset_audit['recovered_record_hashes'][str(event_path)]
    if digest(event_path) != expected:
        raise ValueError('Recovered detector events changed')
    event_trials = {row['source_file']: row for row in saved_events['trials']}
    assumptions = {
        'clock': '(raw timestamp - first untrimmed raw timestamp) / 1000; unverified playback origin',
        'screen': [2560, 1440], 'video_rectangle': 'Centered aspect fit using decoded dimensions and SAR',
        'detector_px2deg': saved_events['px2deg'], 'detector_processing_hz': saved_events['processing_rate_hz'],
        'max_raw_gap_ms': 10, 'scientific_calibration_verified': False,
        'authorization': 'User requested development experiments after empirical sampling checks on 2026-09-30',
    }
    if args.development:
        config['input']['max_raw_gap_ms'] = 10
    video_root = root / 'source/revision-1/Clips-small'
    raw_root = root / 'source/revision-1/GazeData'
    # Decode only development videos, before accessing any model scores.
    development = {key: value for key, value in alignment['videos'].items()}
    video_records = {}
    for video_id, old in sorted(development.items()):
        path = video_root / split['videos'][video_id]['filename']
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            pts = []
            motion = []
            previous = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Decoded frame lacks PTS')
                pts.append(float(frame.pts * frame.time_base) * 1000)
                gray = np.log1p(np.asarray(frame.to_image().convert('L').resize((64, 36)), dtype=float))
                motion.append(0.0 if previous is None else float(np.abs(gray - previous).mean()))
                previous = gray
            info = {'width': stream.width, 'height': stream.height,
                    'sample_aspect_ratio': str(stream.sample_aspect_ratio),
                    'time_base': str(stream.time_base), 'codec': stream.codec_context.name}
            info['duration_ms'] = float(stream.duration * stream.time_base) * 1000 if stream.duration else pts[-1] + np.median(np.diff(pts))
            info['sar'] = float(stream.sample_aspect_ratio) if stream.sample_aspect_ratio else 1.0
        if not pts or not all(math.isfinite(t) for t in pts) or any(a >= b for a, b in zip(pts, pts[1:])):
            raise ValueError('PTS must be finite and strictly increasing')
        recovered = np.array(old['frame_pts_seconds'], dtype=float) * 1000
        error = float(np.max(np.abs(recovered - pts))) if len(recovered) == len(pts) else None
        video_records[video_id] = {'path': str(path), 'sha256': digest(path), 'stream': info,
            'frame_pts_ms': pts, 'recovered_max_error_ms': error,
            'recovered_pts_match': error is not None and error <= .001,
            'motion': motion, 'cut_candidate_indices': np.flatnonzero(np.asarray(motion) > np.mean(motion) + np.std(motion)).tolist()}
    write(output / 'development-video-pts.json', video_records)

    exclusions, coverage = Counter(), Counter()
    metadata = []
    candidates = defaultdict(list)
    identifiers = set()
    split_films = defaultdict(set)
    pilot_trials = output / 'development-pilot-trials.jsonl'
    pilot_stream = pilot_trials.open('x') if args.development else None
    for trial in read_jsonl(trials_path):
        if trial['split'] not in ('train', 'validation'):
            raise ValueError('Audit is development-only')
        video_id = trial['video_id']
        trial['frame_pts_ms'] = video_records[video_id]['frame_pts_ms']
        raw_path = raw_root / trial['source_file'].split('/raw/gaze/', 1)[1]
        raw_hash = digest(raw_path)
        if raw_hash != raw_hashes[trial['source_file']]:
            raise ValueError('Raw trial content changed')
        mat = loadmat(raw_path, simplify_cells=True)
        raw = mat['eyetrackRecord']
        times = np.asarray(raw['t'], dtype=float).ravel()
        standardized = np.asarray([sample['time_ms'] for sample in trial['samples']])
        if len(times) != len(standardized) or not np.allclose(times - times[0], standardized, rtol=0, atol=1e-6):
            raise ValueError('Standardized samples differ from the full raw timestamp sequence')
        steps = np.diff(times)
        usable = (np.asarray(raw['missing']).ravel() == 0) & np.isfinite(raw['x']) & np.isfinite(raw['y'])
        x, y = np.asarray(raw['x']).ravel(), np.asarray(raw['y']).ravel()
        video = video_records[video_id]
        width, height = video['stream']['width'] * video['stream']['sar'], video['stream']['height']
        scale = min(2560 / width, 1440 / height)
        rect = [(2560 - width * scale) / 2, (1440 - height * scale) / 2, width * scale, height * scale]
        if args.development:
            trial['display_rect'] = rect
            trial['video_end_ms'] = video['stream']['duration_ms']
            for event, saved in zip(trial['events'], event_trials[trial['source_file']]['events'], strict=True):
                if (event['label'], event['start_sample'], event['stop_sample']) != (saved['label'], saved['raw_start_index'], saved['raw_stop_index']):
                    raise ValueError('Standardized event differs from its recovered source')
                event['usable_samples'] = saved['usable_samples']
            pilot_stream.write(json.dumps(trial, allow_nan=False, separators=(',', ':')) + '\n')
        fields = {}
        for key, value in raw.items():
            array = np.asarray(value)
            record = {'shape': list(array.shape), 'dtype': str(array.dtype)}
            # Record scalars that could establish synchronization/geometry; avoid guessing.
            if array.size <= 16 and any(word in key.lower() for word in
                    ('screen', 'display', 'resolution', 'start', 'offset', 'origin', 'rate', 'video', 'sample')):
                record['value'] = str(value)
            fields[key] = record
        metadata.append({'trial_id': trial['trial_id'], 'source_file': trial['source_file'],
            'raw_sha256': raw_hash, 'fields': fields,
            'mat_top_level_keys': sorted(key for key in mat if not key.startswith('__')),
            'sample_step_histogram_ms': {str(step): int(count) for step, count in
                zip(*np.unique(steps, return_counts=True))},
            'median_sample_step_ms': float(np.median(steps)),
            'raw_gaps_over_10_ms': int(np.sum(steps > 10)),
            'valid_fraction': float(usable.mean()),
            'coordinate_quantiles_valid': {name: np.quantile(values[usable], [.01, .5, .99]).tolist() for name, values in [('x', x), ('y', y)]},
            'outside_1920x1080_valid_fraction': float(np.mean((x[usable] >= 1920) | (y[usable] >= 1080) | (x[usable] < 0) | (y[usable] < 0))),
            'assumed_video_rectangle': rect, 'raw_end_ms': float(times[-1] - times[0]),
            'decoded_video_end_ms': video['stream']['duration_ms'],
            'min_sample_step_ms': float(np.min(steps)), 'max_sample_step_ms': float(np.max(steps)),
            'gaps_over_1_5_median_period': int(np.sum(steps > 1.5 * np.median(steps))),
            'sample_period_status': 'Measured diagnostic, not acquisition documentation'})
        targets, rejected = build_trial_targets(trial, config)
        exclusions.update(rejected)
        coverage['trials_' + trial['split']] += 1
        coverage['targets_' + trial['split']] += len(targets)
        split_films[trial['split']].add(trial['source_film_id'])
        for target in targets:
            if target['target_id'] in identifiers:
                raise ValueError('Duplicate target ID')
            identifiers.add(target['target_id'])
            plans = target['model_input']['frame_plans']
            if plans['D1'][0] != plans['D4'][-1]:
                raise ValueError('D1/D4 current frame differs')
            if [p['relative_ms'] for p in plans['D4']] != [p['relative_ms'] for p in plans['R4']]:
                raise ValueError('D4/R4 time slots differ')
        # Spread the audit across source films and fixation/pursuit history.
        for target in targets:
            pursuit = any('PURS' in event['source_labels'] for event in target['audit']['history_sources'])
            from bisect import bisect_right
            cutoff = target['model_input']['cutoff_ms']
            current = target['model_input']['frame_plans']['D1'][0]['source_frame_index']
            cut = any(cutoff - 1000 <= video['frame_pts_ms'][i] <= target['audit']['target_start_ms'] for i in video['cut_candidate_indices'])
            high_motion = bool(video['motion'][current] > np.median(video['motion']))
            key = (trial['source_film_id'], float(np.median(steps)), pursuit, cut, high_motion, bool(np.mean(~usable) > .02))
            target['audit']['sampling_stratum'] = {'median_raw_step_ms': key[1], 'pursuit': pursuit,
                'scene_cut_candidate': cut, 'high_motion': high_motion, 'high_missing_trial': key[5]}
            if len(candidates[key]) < 50:
                candidates[key].append((target, trial['video_path'], trial['samples']))
    if pilot_stream:
        pilot_stream.close()
    if split_films['train'] & split_films['validation']:
        raise ValueError('Source film leakage between development splits')
    write(output / 'development-raw-metadata.json', metadata)
    selected, used, feature_counts = [], Counter(), Counter()
    films = sorted({key[0] for key in candidates})
    for _ in range(50):
        for film in films:
            available = [key for key in candidates if key[0] == film and used[key] < len(candidates[key])]
            if available and len(selected) < 50:
                key = max(sorted(available), key=lambda key: sum(1 / (1 + feature_counts[axis, key[axis]]) for axis in range(1, 6)))
                selected.append(candidates[key][used[key]])
                used[key] += 1
                feature_counts.update((axis, key[axis]) for axis in range(1, 6))
    frame_requests = defaultdict(set)
    audit_records = []
    for index, (target, path, samples) in enumerate(selected):
        video_id = target['audit']['video_id']
        pts = video_records[video_id]['frame_pts_ms']
        from bisect import bisect_right
        target_frame = bisect_right(pts, target['audit']['target_start_ms']) - 1
        d4_indices = [row['source_frame_index'] for row in target['model_input']['frame_plans']['D4']]
        frame_requests[path].update(d4_indices + [target_frame])
        audit_records.append({'index': index, 'target': target, 'path': path,
            'd4_indices': d4_indices, 'target_frame': target_frame})
    images = {}
    input_paths = defaultdict(dict)
    if args.development:
        (output / 'frames').mkdir()
    for path, indices in frame_requests.items():
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in indices:
                    if args.development:
                        frame_path = output / 'frames' / (Path(path).stem + f'-{index:06d}.jpg')
                        frame.to_image().save(frame_path, quality=95)
                        input_paths[path][index] = str(frame_path)
                    # Explicit historical reference-canvas assumption, not verified display geometry.
                    images[path, index] = ImageOps.pad(frame.to_image(), (384, 216), color='black')
    audit_dir = output / 'visual-audit'
    audit_dir.mkdir()
    html = ['<!doctype html><meta charset="utf-8"><title>Development audit</title>',
        '<h1>Development diagnostics: synchronization and geometry remain unverified</h1>',
        '<p>50 transitions; assumed 2560x1440 reference screen, centered aspect-preserving video. '
        'Fifth panel is future target audit ONLY, excluded from model inputs. Green: past trajectory; red: target.</p>']
    for record, (_, _, samples) in zip(audit_records, selected):
        target, path = record['target'], record['path']
        canvas = Image.new('RGB', (1920, 280), 'white')
        draw = ImageDraw.Draw(canvas)
        plans = target['model_input']['frame_plans']['D4']
        for slot, frame_index in enumerate(record['d4_indices'] + [record['target_frame']]):
            image = images[path, frame_index].copy()
            overlay = ImageDraw.Draw(image)
            for event in target['audit']['history_sources']:
                points = [(sample['x'] / 2560 * 384, sample['y'] / 1440 * 216)
                    for sample in samples[event['sample_start']:event['sample_stop']] if sample['valid']]
                if len(points) >= 2:
                    overlay.line(points, fill='lime', width=2)
            if slot == 4:
                sample = samples[target['audit']['target_sample_index']]
                x, y = sample['x'] / 2560 * 384, sample['y'] / 1440 * 216
                overlay.ellipse((x-5, y-5, x+5, y+5), outline='red', width=3)
            canvas.paste(image, (slot * 384, 35))
            label = f"Input {slot+1}: {plans[slot]['relative_ms']:.3f} ms" if slot < 4 else 'FUTURE TARGET AUDIT ONLY'
            draw.text((slot * 384 + 5, 5), label, fill='black')
        draw.text((5, 255), target['target_id'] + ' | ASSUMED geometry and playback origin', fill='black')
        name = f'transition-{record["index"]:02d}.jpg'
        canvas.save(audit_dir / name, quality=90)
        html.append(f'<p>{target["target_id"]}</p><img style="max-width:100%" src="{name}">')
    (audit_dir / 'index.html').write_text('\n'.join(html))
    write(output / 'visual-audit-records.json', audit_records)
    if args.development:
        rows = [{'target': record['target'], 'variants': {condition: serialize_target(record['target'], condition, input_paths[record['path']])
                 for condition in ('D1', 'R4', 'D4')}} for record in audit_records]
        train = [row for row in rows if row['target']['audit']['split'] == 'train'][:16]
        if len(train) != 16 or not all(row['recovered_pts_match'] for row in video_records.values()):
            raise ValueError('Insufficient development coverage or invalid video PTS')
        write(output / 'pilot-bundle.json', {'scope': 'development_assumptions', 'assumptions': assumptions,
              'overfit_target_ids': [row['target']['target_id'] for row in train], 'rows': rows})
    report = {'status': 'development_diagnostics_complete_formal_build_blocked',
        'job_id': os.environ['SLURM_JOB_ID'], 'pyav': av.__version__,
        'ffmpeg_libraries': {key: list(value) for key, value in av.library_versions.items()},
        'standardized_trials_sha256': digest(trials_path),
        'dataset_manifest_sha256': config['provenance']['dataset_manifest_sha256'],
        'split_manifest_sha256': digest(args.recovered / 'mapping_handoff/original/fixed-splits.json'),
        'coverage': dict(coverage), 'exclusions': dict(exclusions),
        'paired_target_ids': True, 'decoded_development_videos': len(video_records),
        'recovered_pts_all_match': all(row['recovered_pts_match'] for row in video_records.values()),
        'visual_audit_transitions': len(selected), 'visual_audit_path': str(audit_dir / 'index.html'),
        'sampled_stratum_counts': {field: dict(Counter(str(row['target']['audit']['sampling_stratum'][field]) for row in audit_records))
            for field in ('median_raw_step_ms', 'pursuit', 'scene_cut_candidate', 'high_motion', 'high_missing_trial')},
        'observed_median_period_ms_trial_counts': dict(Counter(str(row['median_sample_step_ms']) for row in metadata)),
        'raw_fields_all_trials': sorted(set.intersection(*(set(row['fields']) for row in metadata))),
        'raw_timestamp_sequence_preserved': True,
        'builder_sha256': digest(Path(__file__).resolve().parents[1] / 'fovlanding.py'),
        'assumptions': assumptions if args.development else None,
        'development_pilot': {'eligible': bool(args.development), 'scope': 'development_assumptions',
            'source_standardized_trials_sha256': preparation['standardized_trials']['sha256'],
            'bundle_sha256': digest(output / 'pilot-bundle.json') if args.development else None},
        'formal_target_build': {'valid_now': False, 'blockers': [
            'Eye-clock to playback-onset mapping lacks independent evidence',
            'Per-trial video display rectangle and px2deg lack independent evidence',
            'Irregular raw sample intervals lack acquisition/export provenance; detector input rate remains unvalidated',
            'Development detector events and visual audit await scientific validation']}}
    if args.development:
        report['status'] = 'empirical_development_checks_passed_under_recorded_assumptions'
        report['standardized_trials_sha256'] = digest(pilot_trials)
        config['provenance'].update(project_git_commit=(output.parent / 'parent-commit.txt').read_text().strip(),
            project_worktree_patch_sha256=digest(output.parent / 'worktree.patch'),
            data_validation_file=str(output / 'summary.json'), audited_trial_metadata_file=str(raw_manifest), validation_scope='development_assumptions', assumptions=assumptions)
        write(output / 'development-config.json', config)
    write(output / 'summary.json', report)
    print(json.dumps(report), flush=True)


def smoke(args):
    import torch
    from transformers import AutoProcessor
    from gpu_train import packed, require, seed_all
    from routeb.model import load_model, log_mass

    output, root = args.output, args.storage
    output.mkdir(parents=True, exist_ok=False)
    seed_all(0, deterministic=True)
    initialization = args.initialization
    processor = AutoProcessor.from_pretrained(initialization, local_files_only=True)
    image = root / 'vendor/DeepGaze3.5-VL/data/images/MIT_0985.jpg'
    row = {'images': [str(image)], 'conversations': [
        {'from': 'human', 'value': '<image>Predict the next gaze coordinate on a 100 x 100 grid. '
         'Past coordinates: (50, 50). Output only [(XX, YY)] with two digits per axis.'},
        {'from': 'gpt', 'value': '[(70, 44)]'}]}
    item = packed(processor, row, 4096)
    model = load_model(initialization, trainable=False, base=root / 'models/InternVL3_5-8B-HF')
    tensors, positions, digit_ids = item
    with torch.no_grad():
        selected = float(log_mass(model, item))
        full = float(log_mass(model, item, full_logits=True))
        changed = {key: value.clone() for key, value in tensors.items()}
        changed['input_ids'][0, positions[2:]] = digit_ids[0]
        indices = torch.tensor(positions, device=tensors['input_ids'].device) - 1
        original_logits = model(**tensors, use_cache=False, logits_to_keep=indices, return_dict=True).logits[0]
        changed_logits = model(**changed, use_cache=False, logits_to_keep=indices, return_dict=True).logits[0]
        error = float((original_logits[0] - changed_logits[0]).abs().max())
    require(math.isfinite(selected) and selected <= 0, 'Invalid cell log probability')
    require(abs(selected - full) < .01, 'Selected/full vocabulary score differs')
    require(error <= 1e-5, 'Future target digits affect the first prediction')
    write(output / 'result.json', {'status': 'static_hf_smoke_passed', 'job_id': os.environ['SLURM_JOB_ID'],
        'scope': 'Static image with diagnostic target; not a video-data or training acceptance',
        'device': torch.cuda.get_device_name(0), 'sequence_length': int(tensors['input_ids'].shape[1]),
        'digit_ids': digit_ids, 'target_positions': positions, 'target_token_ids': tensors['input_ids'][0, positions].tolist(),
        'four_digit_logits': original_logits[:, digit_ids].float().cpu().tolist(),
        'log2p': selected / math.log(2), 'selected_full_error_nats': abs(selected-full),
        'future_digit_first_logit_max_error': error, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)})
    print((output / 'result.json').read_text(), flush=True)


def export(args):
    """Materialize each causal frame once, then scan every model input on CPU."""
    import shutil
    import time
    import av
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from gpu_train import require
    from vlm_check import prepare

    output, root = args.output, args.storage
    started = time.monotonic()
    # Small per-image tensor operations suffer from thread-pool overhead.
    torch.set_num_threads(1)
    config = load_config(args.audit / 'development-config.json', args.audit / 'development-pilot-trials.jsonl')
    audit = json.loads((args.audit / 'summary.json').read_text())
    require(digest(Path(__file__).resolve().parents[1] / 'fovlanding.py') == audit['builder_sha256'], 'Builder differs from audit')
    summary = json.loads((args.targets / 'build-summary.json').read_text())
    require(summary['protocol'] == config['protocol'] and summary['provenance'] == config['provenance'], 'Build protocol differs')
    targets = list(read_jsonl(args.targets / 'targets.jsonl'))
    # Reconstruct every target, binding the build to the audited raw/event input.
    reconstructed = []
    for trial in read_jsonl(args.audit / 'development-pilot-trials.jsonl'):
        reconstructed.extend(build_trial_targets(trial, config)[0])
    require(targets == reconstructed, 'Full target build differs from the audited trials')
    require(dict(Counter(target['audit']['split'] for target in targets)) ==
            {split: audit['coverage']['targets_' + split] for split in ('train', 'validation')}, 'Coverage differs')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.targets / 'targets.jsonl', output / 'targets.jsonl')
    write(output / 'source-config.json', config)
    videos = json.loads((args.audit / 'development-video-pts.json').read_text())
    wanted = defaultdict(set)
    for target in targets:
        wanted[target['audit']['video_id']].update(frame['source_frame_index'] for frame in target['model_input']['frame_plans']['D4'])
    paths, manifest = {}, {}
    reused = None
    if args.reuse is not None:
        require(digest(args.reuse / 'targets.jsonl') == digest(output / 'targets.jsonl') and
                json.loads((args.reuse / 'source-config.json').read_text()) == config, 'Reusable export has different targets/config')
        reused = json.loads((args.reuse / 'frames.json').read_text())
    for video, indices in sorted(wanted.items()):
        source = videos[video]
        require(digest(Path(source['path'])) == source['sha256'], 'Video changed')
        directory = output / 'frames' / video
        directory.mkdir(parents=True)
        paths[video], manifest[video] = {}, []
        if reused is not None:
            require({frame['index'] for frame in reused[video]} == indices, 'Reusable frame population differs')
            for frame in reused[video]:
                path = output / frame['path']
                require(abs(frame['pts_ms'] - source['frame_pts_ms'][frame['index']]) < .001 and
                        digest(args.reuse / frame['path']) == frame['sha256'], 'Reusable frame PTS/hash differs')
                os.link(args.reuse / frame['path'], path)
                paths[video][frame['index']] = str(path)
            manifest[video] = reused[video]
            print(json.dumps({'video': video, 'reused_verified_frames': len(indices)}), flush=True)
            continue
        with av.open(source['path']) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for index, frame in enumerate(container.decode(stream)):
                require(abs(float(frame.pts * frame.time_base) * 1000 - source['frame_pts_ms'][index]) < .001, 'Frame PTS changed')
                if index in indices:
                    path = directory / f'{index:06d}.png'
                    frame.to_image().convert('RGB').save(path, compress_level=1)
                    paths[video][index] = str(path)
                    manifest[video].append({'index': index, 'pts_ms': source['frame_pts_ms'][index],
                        'path': str(path.relative_to(output)), 'sha256': digest(path)})
        require(set(paths[video]) == indices, 'Missing causal frames')
        print(json.dumps({'video': video, 'unique_frames': len(indices)}), flush=True)
    write(output / 'frames.json', manifest)
    prior = np.ones((100, 100), dtype=np.float64)
    for target in targets:
        if target['audit']['split'] == 'train':
            x, y = target['target']['xy_grid']
            prior[y, x] += 1
    np.save(output / 'training-prior.npy', prior / prior.sum(), allow_pickle=False)
    initialization = args.initialization
    processor = AutoProcessor.from_pretrained(initialization, local_files_only=True)
    # Verify transferred base and official adapter against the original manifest.
    original = json.loads((initialization / 'initialization.json').read_text())
    for name, expected in original['weights_sha256'].items():
        path = root / 'models/InternVL3_5-8B-HF' / Path(name).name if Path(name).is_absolute() else initialization / name
        require(digest(path) == expected, 'Initialization content changed: ' + name)
    lengths, visual_counts = defaultdict(list), defaultdict(set)
    digits = None
    for index, target in enumerate(targets):
        rows = {variant: serialize_target(target, variant, paths[target['audit']['video_id']]) for variant in ('D1', 'D4', 'R4')}
        require(rows['D4']['conversations'] == rows['R4']['conversations'] and
                rows['R4']['images'] == rows['D1']['images'] * 4, 'Matched repeat control differs')
        # One fixed patch per image: R4 has the identical token sequence and image shapes as D4.
        for variant in ('D1', 'D4'):
            tensors, positions, legal = prepare(processor, rows[variant])
            tokens = tensors['input_ids'][0]
            require(len(tokens) <= 4096 and min(positions) >= 1, 'Context overflow or invalid response')
            visual = int((tokens == processor.image_token_id).sum())
            require(visual == 256 * len(rows[variant]['images']) and
                    tensors['pixel_values'].shape[0] == len(rows[variant]['images']), 'Image-token inventory differs')
            require(digits is None or legal == digits, 'Digit token IDs differ')
            digits = legal
            lengths[variant].append(len(tokens))
            visual_counts[variant].add(visual)
        if (index + 1) % 500 == 0:
            print(json.dumps({'processor_targets_scanned': index + 1, 'total': len(targets)}), flush=True)
    write(output / 'scan.json', {'status': 'Full fovlanding processor scan passed',
        'scope': 'development_assumptions', 'assumptions': config['provenance']['assumptions'],
        'audit_sha256': digest(args.audit / 'summary.json'), 'build_sha256': digest(args.targets / 'build-summary.json'),
        'split_counts': dict(Counter(target['audit']['split'] for target in targets)),
        'max_length': 4096, 'digit_ids': digits, 'patch_policy': 'One 448x448 patch per image, no crop or truncation',
        'frame_encoding': 'Lossless native-resolution RGB PNG, shared across all conditions',
        'r4_scan': 'D4-identical text and response tokens; four repeated D1 images with identical fixed patch shapes',
        'input_tokens': {variant: {'min': min(values), 'max': max(values), 'mean': float(np.mean(values)),
            'visual_tokens': sorted(visual_counts[variant])} for variant, values in lengths.items()},
        'unique_frames': sum(len(value) for value in manifest.values()),
        'cpu_torch_threads': 1,
        'reused_frame_source': str(args.reuse) if args.reuse else None,
        'prior': 'Training target histogram plus one pseudocount per grid cell; no validation fitting',
        'data_sha256': {name: digest(output / name) for name in ('targets.jsonl', 'frames.json', 'training-prior.npy', 'source-config.json')},
        'seconds': time.monotonic() - started})
    print((output / 'scan.json').read_text(), flush=True)


def overfit(args):
    import gc
    import time
    import torch
    from transformers import AutoProcessor
    from gpu_train import frozen_digest, packed, require, seed_all
    from routeb.model import eval_mass, load_model, log_mass
    from training_state import SampleOrder, mean_backward, save_checkpoint

    root, output = args.storage, args.output
    report = json.loads((args.audit / 'summary.json').read_text())
    config = load_config(args.audit / 'development-config.json', args.audit / 'development-pilot-trials.jsonl')
    require(digest(Path(__file__).resolve().parents[1] / 'fovlanding.py') == report['builder_sha256'], 'Builder changed after data audit')
    require(digest(args.audit / 'pilot-bundle.json') == report['development_pilot']['bundle_sha256'], 'Pilot bundle changed')
    bundle = json.loads((args.audit / 'pilot-bundle.json').read_text())
    selected = {row['target']['target_id']: row for row in bundle['rows']}
    rows = [selected[key] for key in bundle['overfit_target_ids']]
    require(len(rows) == 16 and all(row['target']['audit']['split'] == 'train' for row in rows), 'Overfit must use 16 shared training targets')
    require(torch.cuda.device_count() == 1 and 1 <= args.updates <= 100, 'Use one GPU and a bounded check')
    output.mkdir(parents=True, exist_ok=False)
    initialization = args.initialization
    processor = AutoProcessor.from_pretrained(initialization, local_files_only=True)
    results = {}
    for condition in ('D1', 'R4', 'D4'):
        seed_all(0, deterministic=True)
        model = load_model(initialization, base=root / 'models/InternVL3_5-8B-HF')
        items = [packed(processor, row['variants'][condition], 4096) for row in rows]
        parameters = [p for p in model.parameters() if p.requires_grad]
        require(parameters and all('.lora_' in name for name, p in model.named_parameters() if p.requires_grad), 'Unexpected trainable weights')
        frozen_before = frozen_digest(model)
        before = [eval_mass(model, item) for item in items]
        with torch.no_grad():
            full_error = abs(before[0] - float(log_mass(model, items[0], full_logits=True)))
        require(full_error < .01, 'Selected/full digit likelihood differs')
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        order = SampleOrder(range(16), 0)
        started = time.monotonic()
        directory = output / condition
        directory.mkdir()
        with (directory / 'updates.jsonl').open('x') as stream:
            for update in range(1, args.updates + 1):
                model.train()
                model.get_base_model().model.vision_tower.eval()
                model.get_base_model().model.multi_modal_projector.eval()
                optimizer.zero_grad(set_to_none=True)
                loss = mean_backward(lambda index: -log_mass(model, items[index]), order.next_batch(16))
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                require(float(norm) > 0, 'No finite adapter gradient')
                require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen weight acquired a gradient')
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                entry = {'condition': condition, 'update': update, 'mean_loss_nats': loss, 'gradient_norm': float(norm), 'elapsed_seconds': time.monotonic() - started}
                stream.write(json.dumps(entry) + '\n')
                stream.flush()
                print(json.dumps(entry), flush=True)
        after = [eval_mass(model, item) for item in items]
        require(frozen_digest(model) == frozen_before, 'Frozen model weights changed')
        mean_before, mean_after = -sum(before) / 16, -sum(after) / 16
        require(mean_after < mean_before, '16-target training loss did not decrease')
        metadata = {'scope': 'development_assumptions_training_check', 'condition': condition,
                    'bundle_sha256': report['development_pilot']['bundle_sha256'], 'assumptions': config['provenance']['assumptions']}
        save_checkpoint(directory / 'checkpoint', model, optimizer, scheduler, order, args.updates, metadata)
        results[condition] = {'initial_loss_nats': mean_before, 'final_loss_nats': mean_after,
            'per_target_initial_log2p': [value / math.log(2) for value in before],
            'per_target_final_log2p': [value / math.log(2) for value in after],
            'selected_full_error_nats': full_error, 'frozen_weights_unchanged': True,
            'seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        write(directory / 'result.json', results[condition])
        del parameters, optimizer, scheduler, model, items
        gc.collect()
        torch.cuda.empty_cache()
    write(output / 'result.json', {'status': 'bounded_training_check_passed', 'scope': 'development_assumptions',
        'updates_per_condition': args.updates, 'effective_batch': 16, 'learning_rate': 1e-4,
        'seed': 0, 'target_ids': bundle['overfit_target_ids'], 'results': results,
        'scientific_calibration_verified': False, 'validation_performance_evaluated': False})


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside Slurm')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'smoke', 'overfit', 'export'))
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovered', type=Path)
    parser.add_argument('--preparation', type=Path, help='Hash-pinned standardized-trial preflight record')
    parser.add_argument('--dataset-audit', type=Path, help='Raw-dataset audit record')
    parser.add_argument('--raw-manifest', type=Path, help='Audited raw-trial JSONL manifest')
    parser.add_argument('--initialization', type=Path, help='Preserved official gaze initialization')
    parser.add_argument('--development', action='store_true', help='Prepare development inputs under recorded acquisition assumptions')
    parser.add_argument('--audit', type=Path, help='Completed development audit directory for the training check')
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--targets', type=Path, help='Completed full target-build directory')
    parser.add_argument('--reuse', type=Path, help='Reuse hash-verified PNGs from an earlier export, without decoding again')
    args = parser.parse_args()
    if args.action == 'audit' and args.recovered is None:
        parser.error('--recovered is required for the development audit')
    if args.action == 'audit' and any(value is None for value in (args.preparation, args.dataset_audit, args.raw_manifest)):
        parser.error('audit requires --preparation, --dataset-audit, and --raw-manifest')
    if args.action != 'audit' and args.initialization is None:
        parser.error('--initialization is required for model checks and export')
    if args.action in ('overfit', 'export') and args.audit is None:
        parser.error('--audit is required')
    if args.action == 'export' and args.targets is None:
        parser.error('--targets is required for the full input export')
    {'audit': audit, 'smoke': smoke, 'overfit': overfit, 'export': export}[args.action](args)


if __name__ == '__main__':
    main()
