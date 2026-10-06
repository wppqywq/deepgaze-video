"""Strict ID-paired development summaries; no target-level significance tests."""
from collections import defaultdict
import math


def checked_scores(records, expected_metadata, require_prior=True):
    result = {}
    for record in records:
        sample_id = record['id']
        if sample_id in result or sample_id not in expected_metadata:
            raise ValueError('Duplicate or unexpected target ID')
        metadata = expected_metadata[sample_id]
        if any(record[key] != metadata[key] for key in ('video', 'source_group')):
            raise ValueError('Target metadata mismatch')
        if not math.isfinite(record['bits']) or record['bits'] > 1e-6:
            raise ValueError('Invalid log2 probability')
        if require_prior:
            if not math.isfinite(record['prior_bits']) or record['prior_bits'] > 1e-6:
                raise ValueError('Invalid prior log2 probability')
            if not math.isclose(record['ig_bits'], record['bits']-record['prior_bits'], abs_tol=1e-8, rel_tol=0):
                raise ValueError('Information gain mismatch')
        result[sample_id] = record
    if set(result) != set(expected_metadata):
        raise ValueError('Incomplete paired target population')
    return result


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values)


def summarize(models, metadata):
    """Equal-target means plus source-film means, always joined by ID."""
    ids = list(metadata)
    if not ids or any(set(rows) != set(ids) for rows in models.values()):
        raise ValueError('Models must cover the identical nonempty target population')
    groups = defaultdict(list)
    for sample_id in ids:
        groups[metadata[sample_id]['source_group']].append(sample_id)

    def scope(targets):
        scores = {name: {'bits': mean(rows[i]['bits'] for i in targets),
                         'ig_bits': mean(rows[i]['ig_bits'] for i in targets)}
                  for name, rows in models.items()}
        differences = {left+'-'+right: mean(models[left][i]['bits']-models[right][i]['bits'] for i in targets)
                       for left, right in [('d4', 'd1'), ('d4', 'r4'), ('d1', 'b1'), ('d4', 'b1'), ('r4', 'b1')]}
        return {'targets': len(targets), 'models': scores, 'paired_differences_bits': differences}

    return {'all_targets': scope(ids), 'by_source_group': {key: scope(value) for key, value in sorted(groups.items())}}


def select_evaluation(evaluations, epoch, best=False, metric='mean_bits'):
    selected = [row for row in evaluations if row['epoch'] <= epoch]
    if sorted(row['epoch'] for row in selected) != list(range(1, epoch+1)):
        raise ValueError('Missing or duplicate full-epoch evaluation')
    if any(not math.isfinite(row[metric]) for row in selected):
        raise ValueError('Nonfinite selection score')
    if best:
        return max(selected, key=lambda row: (row[metric], -row['epoch']))
    return next(row for row in selected if row['epoch'] == epoch)


def summarize_seeds(reports):
    """Average paired differences across seeds before resampling source films."""
    import numpy as np
    if len(reports) < 3 or len({row['seed'] for row in reports}) != len(reports):
        raise ValueError('Need at least three distinct training seeds')
    films = sorted(reports[0]['by_film_bits'])
    models = {'d1', 'r4', 'd4', 'static_d1'}
    if not films or any(sorted(row['by_film_bits']) != films or
            row['selection'] != reports[0]['selection'] for row in reports):
        raise ValueError('Seed comparisons must use the same films and selection rule')
    for report in reports:
        for scores in report['by_film_bits'].values():
            if set(scores) != models or not all(math.isfinite(value) for value in scores.values()):
                raise ValueError('Missing or nonfinite model scores')
    rng = np.random.default_rng(20260929)
    draws = rng.integers(len(films), size=(5000, len(films)))
    contrasts = {}
    for left, right in (('d4', 'r4'), ('d4', 'd1'), ('r4', 'd1'), ('d1', 'static_d1')):
        differences = np.array([[row['by_film_bits'][film][left] - row['by_film_bits'][film][right]
                                 for film in films] for row in reports])
        by_film, by_seed = differences.mean(axis=0), differences.mean(axis=1)
        contrasts[left + '-' + right] = {'macro_bits': float(by_film.mean()),
            'by_film_seed_mean_bits': dict(zip(films, by_film.tolist())),
            'by_seed_macro_bits': {str(row['seed']): float(value) for row, value in zip(reports, by_seed)},
            'seed_std_bits': float(by_seed.std(ddof=1)),
            'descriptive_film_bootstrap_95': np.quantile(by_film[draws].mean(axis=1), [.025, .975]).tolist()}
    return {'seeds': [row['seed'] for row in reports], 'films': len(films),
        'selection': reports[0]['selection'], 'paired_contrasts': contrasts,
        'macro_film_bits': {model: mean(row['by_film_bits'][film][model] for row in reports for film in films)
                            for model in sorted(models)},
        'aggregation': 'Average seeds within each source film, then bootstrap films; seed variability reported separately',
        'bootstrap_repetitions': 5000, 'bootstrap_seed': 20260929,
        'scope': 'development_assumptions', 'scientific_calibration_verified': False,
        'limitations': ['Repeated seeds do not increase the number of independent films.',
            'Development comparisons; validation selection and unverified acquisition assumptions remain limitations.']}
