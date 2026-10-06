"""Compare v2 video, static and B1 runs by exact validation ID and cell mass."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from routeb.comparison import checked_scores, mean, select_evaluation, summarize_seeds


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_file(root, name, expected):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError('Artifact path/hash differs')
    return path


def read_scores(path, metadata):
    return checked_scores([json.loads(line) for line in path.read_text().splitlines() if line], metadata)


def load_run(root, epoch, best, metadata):
    protocol = read(root / 'protocol.json')
    architecture = protocol['architecture']
    if architecture == 'single-gaze-adapter-v2' and protocol['action'] == 'static':
        result = read(root / 'result.json')
        if result['status'] != 'complete' or result['protocol_sha256'] != digest(root / 'protocol.json'):
            raise ValueError('Static evaluation incomplete or changed')
        chosen = result['evaluation']
    elif architecture == 'single-gaze-adapter-v2':
        latest = read(root / 'latest.json')['progress']
        if latest['validated_epochs'] < epoch:
            raise ValueError('Requested common budget is incomplete')
        chosen = select_evaluation(latest['evaluations'], epoch, best)
    elif architecture == 'resnet18-gru-mdn-v2':
        result = read(root / 'result.json')
        if result['status'] != 'complete' or result['protocol_sha256'] != digest(root / 'protocol.json'):
            raise ValueError('B1 run incomplete or changed')
        chosen = select_evaluation(result['epochs'], epoch, best)
    else:
        raise ValueError('Expected a v2 model; legacy scores belong to the legacy comparison')
    if 'report' in chosen:
        report = read(checked_file(root, chosen['report'], chosen['sha256']))
    else:
        report = chosen
    path = checked_file(root, report['predictions'], report['predictions_sha256'])
    scores = read_scores(path, metadata)
    if not math.isclose(mean(row['bits'] for row in scores.values()), chosen['mean_bits'], abs_tol=1e-8):
        raise ValueError('Selected mean differs from predictions')
    return protocol, chosen['epoch'], scores


def compare_fovlanding(args):
    """Matched development comparison, including StaticD1 from D1 step zero."""
    from collections import defaultdict
    import numpy as np
    from routeb.data import FovlandingInputs

    inputs = FovlandingInputs(args.outputs, args.outputs / 'scan.json')
    metadata = {key: inputs.samples[key] for key in inputs.ids['validation']}
    specs = dict(spec.split('=', 1) for spec in args.run)
    if len(specs) != len(args.run) or set(specs) != {'d1', 'r4', 'd4'}:
        raise ValueError('Provide exactly one d1, r4 and d4 run')
    models, selections, common = {}, {}, None

    def scores(root, chosen):
        report = read(checked_file(root, chosen['report'], chosen['sha256']))
        records = read_scores(checked_file(root, report['predictions'], report['predictions_sha256']), metadata)
        grouped = defaultdict(list)
        for key, record in records.items():
            grouped[metadata[key]['source_group']].append(record['bits'])
        macro = mean(mean(values) for values in grouped.values())
        if not math.isclose(macro, report['macro_film_bits'], abs_tol=1e-8) or not math.isclose(macro, chosen['selection_bits'], abs_tol=1e-8):
            raise ValueError('Film-macro score differs from paired predictions')
        return records

    for variant, path in specs.items():
        root = Path(path)
        protocol = read(root / 'protocol.json')
        result = read(root / 'result.json')
        if (protocol['architecture'] != 'single-gaze-adapter-fovlanding-v1' or protocol['variant'] != variant
                or result['status'] != 'complete' or result['protocol_sha256'] != digest(root / 'protocol.json')
                or protocol['inputs'] != inputs.provenance):
            raise ValueError('Incomplete or mismatched fovlanding run')
        matched = {key: protocol[key] for key in ('prompt', 'config', 'inputs', 'initialization_sha256', 'source_sha256', 'runtime')}
        if common is not None and matched != common:
            raise ValueError('Protocols differ beyond visual condition')
        common = matched
        config = protocol['config']
        if args.updates != config['max_optimizer_steps'] or result['update'] != args.updates:
            raise ValueError('Different or incomplete optimization budgets')
        progress = result['progress']
        selected = select_evaluation(progress['evaluations'], args.updates // config['validation_every_updates'],
                                     args.best, metric='selection_bits')
        models[variant] = scores(root, selected)
        update = selected['epoch'] * config['validation_every_updates']
        checkpoint = root / 'checkpoints' / f"checkpoint-{update:06d}-v{selected['epoch']:03d}"
        manifest = read(checkpoint / 'manifest.json')
        if manifest['metadata']['protocol'] != protocol or manifest['update'] != update:
            raise ValueError('Selected checkpoint protocol differs')
        for name, expected in manifest['sha256'].items():
            checked_file(checkpoint, name, expected)
        selections[variant] = {'run': str(root), 'optimizer_update': update,
            'budget_updates': args.updates, 'selection_macro_film_bits': selected['selection_bits'],
            'checkpoint_manifest_sha256': digest(checkpoint / 'manifest.json'),
            'protocol_sha256': digest(root / 'protocol.json')}
        if variant == 'd1':
            models['static_d1'] = scores(root, progress['initial_evaluation'])
    groups = defaultdict(list)
    for key, record in metadata.items():
        groups[record['source_group']].append(key)
    by_film = {film: {name: mean(rows[key]['bits'] for key in keys) for name, rows in models.items()}
               for film, keys in sorted(groups.items())}
    contrasts = {}
    rng = np.random.default_rng(20260929)
    draws = rng.integers(len(groups), size=(5000, len(groups)))
    for left, right in (('d4', 'r4'), ('d4', 'd1'), ('r4', 'd1'), ('d1', 'static_d1')):
        differences = np.array([values[left] - values[right] for values in by_film.values()])
        contrasts[left + '-' + right] = {'macro_bits': float(differences.mean()),
            'by_film_bits': dict(zip(by_film, differences.tolist())),
            'film_bootstrap_95_percentile': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
    result = {'scope': 'development_assumptions', 'scientific_calibration_verified': False,
        'selection': 'validation_macro_film_best' if args.best else 'same_optimizer_update',
        'seed': common['config']['seed'], 'selections': selections, 'validation_targets': len(metadata),
        'films': len(groups), 'by_film_bits': by_film,
        'macro_film_bits': {name: mean(values[name] for values in by_film.values()) for name in models},
        'micro_target_bits': {name: mean(row['bits'] for row in rows.values()) for name, rows in models.items()},
        'paired_contrasts': contrasts, 'bootstrap_repetitions': 5000, 'bootstrap_seed': 20260929,
        'limitations': ['Single training seed; validation used for selection.',
            'Intervals resample these validation films only and are descriptive, not independent test evidence.',
            'Playback origin, per-trial display geometry and acquisition/export provenance remain assumed.']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'paired-scores.jsonl').open('x') as stream:
        for key in metadata:
            stream.write(json.dumps({'target_id': key, 'source_film_id': metadata[key]['source_group'],
                'observer_id': metadata[key]['observer'], 'log2p': {name: rows[key]['bits'] for name, rows in models.items()}}) + '\n')
    print(json.dumps(result), flush=True)


def history_features(targets, lags, full_history=False):
    """Newest completed events first; optionally include all prompt gaze fields."""
    import numpy as np
    rows = []
    for target in targets:
        history = target['model_input']['history']
        if not history or (full_history and len(history) > lags):
            raise ValueError('AR requires completed, strictly past foveations')
        for event in history:
            start, last, end = (event[key] for key in
                ('start_relative_ms', 'last_sample_relative_ms', 'end_boundary_relative_ms'))
            coordinates = np.asarray([event['start_xy_grid'], event['end_xy_grid']], dtype=float)
            if (not np.isfinite([start, last, end]).all() or not start <= last < 0 or not last <= end <= 0 or
                    coordinates.shape != (2, 2) or not np.isfinite(coordinates).all() or
                    (coordinates < 0).any() or (coordinates > 99).any() or
                    (coordinates != np.floor(coordinates)).any()):
                raise ValueError('Invalid or noncausal history')
        points, present = np.full((lags, 6 if full_history else 2), .5), np.zeros(lags)
        if full_history:
            points[:, 4:] = 0
        for index, event in enumerate(reversed(history[-lags:])):
            points[index, :2] = (np.asarray(event['end_xy_grid'], dtype=float) + .5) / 100
            if full_history:
                points[index, 2:4] = (np.asarray(event['start_xy_grid'], dtype=float) + .5) / 100
                # Match the prompt's three-decimal millisecond precision; use seconds.
                points[index, 4:] = [float(f"{event[key]:.3f}") / 1000 for key in
                                    ('start_relative_ms', 'end_boundary_relative_ms')]
            present[index] = 1
        rows.append(np.r_[points.ravel(), present])
    return np.asarray(rows)


def fit_history_gaussian(features, cells, ridge=None):
    import numpy as np
    y = (np.asarray(cells) + .5) / 100
    if ridge is None:
        return {'kind': 'persistence', 'scale': np.maximum(np.sqrt(np.mean((y - features[:, :2]) ** 2, axis=0)), .005)}
    center, spread = features.mean(axis=0), features.std(axis=0)
    spread[spread < 1e-8] = 1
    design = np.c_[np.ones(len(features)), (features - center) / spread]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    scale = np.maximum(np.sqrt(np.mean((y - design @ coefficients) ** 2, axis=0)), .005)
    return {'kind': 'ridge_coordinate_ar', 'center': center, 'spread': spread,
            'coefficients': coefficients, 'scale': scale, 'ridge': ridge}


def predict_history_gaussian(fit, features):
    import numpy as np
    return (features[:, :2] if fit['kind'] == 'persistence' else
            np.c_[np.ones(len(features)), (features - fit['center']) / fit['spread']] @ fit['coefficients'])


def history_baselines(inputs):
    """Select regularization/uncertainty using training-film folds only."""
    import numpy as np
    from stage0 import gaussian_cell_log_mass

    targets = {row['target_id']: row for row in inputs.targets}
    training = [targets[key] for key in inputs.ids['train']]
    validation = [targets[key] for key in inputs.ids['validation']]
    cells = np.asarray([row['target']['xy_grid'] for row in training])
    heldout_cells = np.asarray([row['target']['xy_grid'] for row in validation])
    films = np.asarray([row['audit']['source_film_id'] for row in training])
    if len(set(films)) < 2:
        raise ValueError('Training-film cross-validation needs at least two films')
    models, fits = {}, {}
    for name, lags, ridges in (('persistence', 1, [None]), ('ar1', 1, [.1, 10., 1000.]),
                              ('ar4', 4, [.1, 10., 1000.]), ('linear_history4', 4, [.1, 10., 1000.])):
        full_history = name == 'linear_history4'
        features = history_features(training, lags, full_history)
        candidates = []
        for ridge in ridges:
            fold_scores = {multiplier: [] for multiplier in (.5, 1., 2.)}
            for film in sorted(set(films)):
                held = films == film
                fit = fit_history_gaussian(features[~held], cells[~held], ridge)
                means = predict_history_gaussian(fit, features[held])
                for multiplier in fold_scores:
                    score = gaussian_cell_log_mass(means, fit['scale'] * multiplier, cells[held]) / math.log(2)
                    fold_scores[multiplier].append(float(score.mean()))
            candidates.extend({'ridge': ridge, 'scale_multiplier': multiplier,
                'training_film_cv_bits': mean(values)} for multiplier, values in fold_scores.items())
        chosen = max(candidates, key=lambda row: row['training_film_cv_bits'])
        fit = fit_history_gaussian(features, cells, chosen['ridge'])
        means = predict_history_gaussian(fit, history_features(validation, lags, full_history))
        scores = gaussian_cell_log_mass(means, fit['scale'] * chosen['scale_multiplier'], heldout_cells) / math.log(2)
        models[name] = dict(zip(inputs.ids['validation'], scores.tolist()))
        fits[name] = {'lags': lags, 'selected': chosen, 'candidates': candidates,
            'parameters': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fit.items()},
            'training_films': sorted(set(films)), 'fitting_split': 'train only; leave-one-training-film-out selection',
            'features': ('Per lag: end XY, start XY, start/end relative seconds at prompt precision; then presence masks'
                         if full_history else 'Per lag: completed-foveation end XY; then presence masks'),
            'feature_order': 'Newest first; XY are normalized cell centers; missing XY=.5, times=0, presence=0',
            'feature_dimensions': features.shape[1],
            'nominal_distribution_parameters': 2 if chosen['ridge'] is None else 2 * (features.shape[1] + 1) + 2,
            'scale_fit': 'RMS residual about the predicted mean, half-cell floor, train-film CV multiplier',
            'objective': 'Fixed last endpoint' if chosen['ridge'] is None else 'Ridge least squares on target cell centers; not digit-token likelihood training',
            'likelihood': 'Diagonal Gaussian integrated over the same 100x100 cells and conditioned on the video rectangle'}
    return models, fits


def center_bias(args):
    """Training-only spatial baselines, evaluated on the frozen validation IDs."""
    from collections import defaultdict
    import numpy as np
    from routeb.data import FovlandingInputs
    from stage0 import gmm_grid

    started = time.perf_counter()
    inputs = FovlandingInputs(args.outputs, args.outputs / 'scan.json')
    loaded = time.perf_counter()
    train = np.array([inputs.samples[key]['target']['cell'] for key in inputs.ids['train']], dtype=float)
    centers = (train + .5) / 100
    scale = centers.std(axis=0, ddof=1)
    bandwidth = scale * len(centers) ** (-1 / 6)
    grids = {'existing_histogram': inputs.prior,
        'train_gaussian': gmm_grid(np.ones(1), centers.mean(axis=0)[None], scale[None]),
        'train_kde_scott': gmm_grid(np.full(len(centers), 1 / len(centers)), centers,
                                  np.broadcast_to(bandwidth, centers.shape))}
    metadata = {key: inputs.samples[key] for key in inputs.ids['validation']}
    models = {}
    for name, grid in grids.items():
        if not np.isfinite(grid).all() or (grid <= 0).any() or abs(grid.sum() - 1) > 1e-10:
            raise ValueError('Invalid spatial baseline grid')
        models[name] = {key: float(np.log2(grid[sample['target']['cell'][1], sample['target']['cell'][0]]))
                        for key, sample in metadata.items()}
    spatial_done = time.perf_counter()
    fits = {}
    if args.history_baselines:
        history_models, fits = history_baselines(inputs)
        models.update(history_models)
    history_done = time.perf_counter()
    selections = {}
    common = None
    for spec in args.run or []:
        label, path = spec.split('=', 1)
        root = Path(path)
        protocol, result = read(root / 'protocol.json'), read(root / 'result.json')
        if (protocol['inputs'] != inputs.provenance or protocol['variant'] not in ('d1', 'r4', 'd4') or
                result['status'] != 'complete' or result['protocol_sha256'] != digest(root / 'protocol.json')):
            raise ValueError('Expected a completed visual condition on the same inputs')
        matched = {key: protocol[key] for key in ('config', 'initialization_sha256', 'source_sha256', 'runtime')}
        if common is not None and common != matched:
            raise ValueError('Visual training conditions have different protocols')
        common = matched
        progress = result['progress']
        chosen = select_evaluation(progress['evaluations'], progress['validated_epochs'], True, 'selection_bits')
        evaluations = [(label + '_best', chosen)]
        if protocol['variant'] == 'd1':
            evaluations.insert(0, (label + '_static', progress['initial_evaluation']))
        for name, selection in evaluations:
            if name in models:
                raise ValueError('Duplicate model label')
            report = read(checked_file(root, selection['report'], selection['sha256']))
            rows = read_scores(checked_file(root, report['predictions'], report['predictions_sha256']), metadata)
            models[name] = {key: row['bits'] for key, row in rows.items()}
            selections[name] = selection
    for spec in args.neural_ar or []:
        label, path = spec.split('=', 1)
        root = Path(path)
        protocol, result = read(root / 'protocol.json'), read(root / 'result.json')
        if (label in models or protocol['architecture'] != 'resnet18-gru-mdn-fovlanding-v1' or
                protocol['inputs'] != inputs.provenance or result['status'] != 'complete' or
                result['protocol_sha256'] != digest(root / 'protocol.json')):
            raise ValueError('Expected a distinct completed neural AR on the same inputs')
        chosen = select_evaluation(result['epochs'], protocol['epochs'], True, 'selection_bits')
        checked_file(root, f"epoch-{chosen['epoch']:03d}.pt", chosen['checkpoint_sha256'])
        rows = read_scores(checked_file(root, chosen['predictions'], chosen['predictions_sha256']), metadata)
        models[label] = {key: row['bits'] for key, row in rows.items()}
        selections[label] = {'run': str(root), 'protocol_sha256': digest(root / 'protocol.json'), **chosen}
    visual_done = time.perf_counter()
    films = defaultdict(list)
    for key, sample in metadata.items():
        films[sample['source_group']].append(key)
    by_film = {film: {name: mean(scores[key] for key in keys) for name, scores in models.items()}
               for film, keys in films.items()}
    contrasts = {}
    rng = np.random.default_rng(20260929)
    draws = rng.integers(len(films), size=(5000, len(films)))
    for left, right in [('d4_best', 'r4_best'), ('d4_best', 'd1_best')] + [
            (name, baseline) for name in ('d1_best', 'd4_best')
            for baseline in ['train_kde_scott', 'persistence', 'ar1', 'ar4', 'linear_history4'] +
                            [spec.split('=', 1)[0] for spec in args.neural_ar or []]]:
        if left in models and right in models:
            values = np.asarray([row[left] - row[right] for row in by_film.values()])
            contrasts[left + '-' + right] = {'macro_bits': float(values.mean()),
                'by_film_bits': dict(zip(by_film, values.tolist())),
                'descriptive_film_bootstrap_95': np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}
    report = {'scope': 'development_assumptions', 'train_targets': len(train), 'validation_targets': len(metadata),
        'inputs': inputs.provenance, 'fitting_split': 'train only; no validation bandwidth/parameter selection',
        'coordinate_fit': 'Quantized target cell centers in normalized video coordinates',
        'gaussian': 'Axis-aligned Gaussian integrated over cells and conditioned on the video rectangle',
        'kde': 'Equal-weight Gaussian mixture; diagonal Scott bandwidth; integrated over cells',
        'training_mean_xy': centers.mean(axis=0).tolist(), 'training_std_xy': scale.tolist(),
        'kde_bandwidth_xy': bandwidth.tolist(), 'selections': selections, 'by_film_bits': by_film,
        'history_fits': fits, 'paired_contrasts': contrasts, 'baseline_version': 2,
        'alignment': {
            'matched': ['Target IDs/split', 'Causal gaze cutoff', 'floor100 grid and exact normalized cell log2 mass', 'Film-macro aggregation'],
            'endpoint_ar': 'AR(1)/AR(4) omit gaze start coordinates and times available to the visual model.',
            'linear_history4': 'Same serialized gaze fields; no images or frame timestamps. Different model capacity, initialization, objective and tuning budget.',
            'interpretation': 'Simple coordinate autoregression, not a reproduction of DeepGaze3.5 digit-token autoregression. Gains cannot isolate the contribution of images.'},
        'timing_seconds': {'input_verification': loaded - started, 'spatial_fit_and_score': spatial_done - loaded,
            'history_cv_fit_and_score': history_done - spatial_done, 'load_saved_visual_scores': visual_done - history_done},
        'visual_score_source': 'Previously saved predictions; no VLM loading, training or inference in this job',
        'limitations': ['Single-seed development comparison; visual checkpoints selected on these validation films.',
                       'Film bootstrap is descriptive; acquisition/calibration assumptions remain unverified.',
                       'Linear single-Gaussian history models do not match VLM capacity or objective; this is not an image ablation.',
                       'Residual RMS is an untruncated continuous-Gaussian approximation; only CV scale selection uses discrete cell likelihood.'],
        'macro_film_bits': {name: mean(values[name] for values in by_film.values()) for name in models},
        'micro_target_bits': {name: mean(scores.values()) for name, scores in models.items()},
        'comparison_code_sha256': digest(Path(__file__)),
        'gaussian_code_sha256': digest(Path(__file__).resolve().parent.parent / 'stage0.py'),
        'scientific_calibration_verified': False}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / 'spatial-priors.npz', **grids)
    report['spatial_priors_sha256'] = digest(args.output / 'spatial-priors.npz')
    with (args.output / 'paired-scores.jsonl').open('x') as stream:
        for key, sample in metadata.items():
            stream.write(json.dumps({'target_id': key, 'source_film_id': sample['source_group'],
                'log2p': {name: scores[key] for name, scores in models.items()}}) + '\n')
    report['paired_scores_sha256'] = digest(args.output / 'paired-scores.jsonl')
    report['timing_seconds']['total_before_report_write'] = time.perf_counter() - started
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


def compare_seeds(args):
    from collections import defaultdict
    from routeb.data import FovlandingInputs

    inputs = FovlandingInputs(args.outputs, args.outputs / 'scan.json')
    expected = {key: inputs.samples[key]['source_group'] for key in inputs.ids['validation']}
    reports, evidence, common = [], [], None
    for directory in args.seed_comparison:
        report = read(directory / 'comparison.json')
        if set(report['selections']) != {'d1', 'r4', 'd4'}:
            raise ValueError('Missing visual condition in seed comparison')
        for variant, selection in report['selections'].items():
            run = Path(selection['run'])
            protocol = read(checked_file(run, 'protocol.json', selection['protocol_sha256']))
            if protocol['variant'] != variant or protocol['config']['seed'] != report['seed'] or protocol['inputs'] != inputs.provenance:
                raise ValueError('Seed, variant or paired input provenance differs')
            matched = {key: protocol[key] for key in ('prompt', 'initialization_sha256', 'source_sha256', 'runtime')}
            matched['config'] = {key: value for key, value in protocol['config'].items() if key != 'seed'}
            if common is not None and common != matched:
                raise ValueError('Training protocols differ beyond seed and condition')
            common = matched
            update = selection['optimizer_update']
            checkpoint = run / 'checkpoints' / f"checkpoint-{update:06d}-v{update // protocol['config']['validation_every_updates']:03d}"
            checked_file(checkpoint, 'manifest.json', selection['checkpoint_manifest_sha256'])
        pairs = [json.loads(line) for line in (directory / 'paired-scores.jsonl').read_text().splitlines() if line]
        if len(pairs) != len(expected) or {row['target_id'] for row in pairs} != set(expected):
            raise ValueError('Seed target population differs')
        grouped = defaultdict(list)
        for row in pairs:
            if row['source_film_id'] != expected[row['target_id']]:
                raise ValueError('Seed source-film pairing differs')
            grouped[row['source_film_id']].append(row['log2p'])
        if set(grouped) != set(report['by_film_bits']):
            raise ValueError('Seed film population differs')
        for film, scores in report['by_film_bits'].items():
            for model, score in scores.items():
                if not math.isclose(score, mean(row[model] for row in grouped[film]), abs_tol=1e-10):
                    raise ValueError('Seed film summary differs from paired predictions')
        reports.append(report)
        evidence.append({'directory': str(directory), 'comparison_sha256': digest(directory / 'comparison.json'),
                         'paired_scores_sha256': digest(directory / 'paired-scores.jsonl')})
    result = summarize_seeds(reports)
    result.update(comparisons=evidence, matched_protocol=common, inputs=inputs.provenance,
                  comparison_code_sha256=digest(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--run', action='append', help='Unique label=/path/to/run; repeat for each model')
    parser.add_argument('--epoch', type=int, choices=[1, 2, 3])
    parser.add_argument('--input-protocol', choices=['video-v2', 'fovlanding-v1'], default='video-v2')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--b1-epochs', type=int, default=20)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--center-bias', action='store_true')
    parser.add_argument('--history-baselines', action='store_true', help='Add training-fitted persistence, AR(1) and AR(4) to --center-bias')
    parser.add_argument('--neural-ar', action='append', default=[], help='LABEL=completed fovlanding neural AR run; used with --center-bias')
    parser.add_argument('--seed-comparison', type=Path, action='append', help='Completed per-seed comparison directory; repeat for at least three seeds')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run in Slurm')
    if args.output.exists() or args.b1_epochs < 1:
        parser.error('Use a new output directory and positive B1 budget')
    if args.seed_comparison:
        if args.input_protocol != 'fovlanding-v1' or args.run or args.center_bias:
            parser.error('--seed-comparison requires fovlanding-v1 and no --run/--center-bias')
        compare_seeds(args)
        return
    if args.history_baselines and not args.center_bias:
        parser.error('--history-baselines requires --center-bias')
    if args.center_bias:
        if args.input_protocol != 'fovlanding-v1':
            parser.error('--center-bias requires fovlanding-v1')
        center_bias(args)
        return
    if not args.run:
        parser.error('--run is required for model comparisons')
    if args.input_protocol == 'fovlanding-v1':
        compare_fovlanding(args)
        return
    if args.epoch is None:
        parser.error('--epoch is required for the historical video-v2 comparison')
    ids = read(args.outputs / 'vlm/sample_ids.json')['validation']
    samples = [json.loads(line) for line in (args.outputs / 'samples.jsonl').read_text().splitlines()]
    indexed = {row['id']: row for row in samples}
    if len(indexed) != len(samples) or len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')
    metadata = {sample_id: indexed[sample_id] for sample_id in ids}
    prior_rows = [json.loads(line) for line in (args.outputs / 'scores.jsonl').read_text().splitlines()]
    prior = checked_scores([{**row, 'bits': row['prior_bits'], 'ig_bits': 0.} for row in prior_rows], metadata)
    models, selections, common_inputs, vlm_protocol = {'b0': prior}, {}, None, None
    for spec in args.run:
        label, path = spec.split('=', 1)
        if not label or label in models:
            raise ValueError('Duplicate or empty model label')
        root = Path(path)
        is_b1 = read(root / 'protocol.json')['architecture'] == 'resnet18-gru-mdn-v2'
        budget = args.b1_epochs if is_b1 else args.epoch
        protocol, chosen_epoch, scores = load_run(root, budget, args.best, metadata)
        data = protocol['inputs']
        if data['samples_sha256'] != digest(args.outputs / 'samples.jsonl') or \
                data['pairing_sha256'] != digest(args.outputs / 'vlm/sample_ids.json') or \
                data['prior_sha256'] != digest(args.outputs / 'training-prior.npy'):
            raise ValueError('Run input provenance differs from comparison inputs')
        if common_inputs is not None and data != common_inputs:
            raise ValueError('Models used different inputs')
        common_inputs = data
        if not is_b1:
            common = {key: protocol[key] for key in ['prompt', 'config', 'initialization_sha256', 'source_sha256', 'runtime']}
            if vlm_protocol is not None and common != vlm_protocol:
                raise ValueError('VLM protocols differ beyond variant/action')
            vlm_protocol = common
        for sample_id, row in scores.items():
            if not math.isclose(row['prior_bits'], prior[sample_id]['prior_bits'], abs_tol=1e-8, rel_tol=0):
                raise ValueError('Prior differs across models')
        models[label] = scores
        selections[label] = {'run': str(root), 'epoch': chosen_epoch, 'budget': 0 if chosen_epoch == 0 else budget,
                             'architecture': protocol['architecture'], 'protocol_sha256': digest(root / 'protocol.json')}
    groups = {'all': ids}
    for sample_id in ids:
        groups.setdefault(metadata[sample_id]['source_group'], []).append(sample_id)
    summary = {}
    for group, targets in groups.items():
        summary[group] = {'targets': len(targets),
                         'models': {label: {'bits': mean(rows[i]['bits'] for i in targets),
                                            'ig_bits': mean(rows[i]['ig_bits'] for i in targets)}
                                    for label, rows in models.items()},
                         'paired_differences_bits': {left + '-' + right:
                             mean(models[left][i]['bits'] - models[right][i]['bits'] for i in targets)
                             for left, right in itertools.combinations(models, 2)}}
    args.output.mkdir(parents=True)
    result = {'selection': 'best' if args.best else 'same_epoch', 'selections': selections,
              'summary': summary, 'limitations': ['Development only; no final test.',
                  'B1 and VLM capacity, preprocessing and optimization differ.',
                  'No target-level significance claims; targets are dependent.']}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with (args.output / 'paired-scores.jsonl').open('x') as stream:
        for sample_id in ids:
            stream.write(json.dumps({'id': sample_id, 'source_group': metadata[sample_id]['source_group'],
                                     'bits': {label: rows[sample_id]['bits'] for label, rows in models.items()}}) + '\n')
    lines = ['# Paired video comparison', '', '| Scope | Targets | ' + ' | '.join(models) + ' |',
             '|---|---:|' + '---:|' * len(models)]
    for group, values in summary.items():
        lines.append('| ' + group + ' | ' + str(values['targets']) + ' | ' +
                     ' | '.join(f"{values['models'][label]['bits']:.6f}" for label in models) + ' |')
    lines += ['', 'Higher bits/target is better. Selection budgets:', '']
    lines += [f"- {label}: selected epoch {item['epoch']} within {item['budget']} epochs." for label, item in selections.items()]
    lines += ['', *result['limitations']]
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
