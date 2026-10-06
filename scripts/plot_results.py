"""Export public aggregate results and plot the shared center-bias benchmark."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def export_results(storage):
    sources = {
        'selected': storage / 'runs/aggregate-5824755/best/comparison.json',
        'final': storage / 'runs/aggregate-5824755/final/comparison.json',
        'baselines': storage / 'runs/baseline-5823952/comparison/comparison.json',
    }
    records = {name: read(path) for name, path in sources.items()}
    selected, final, baselines = (records[name] for name in sources)
    if any(record['inputs'] != selected['inputs'] for record in records.values()):
        raise ValueError('Results have different input provenance')
    if selected['seeds'] != [0, 1, 2] or final['seeds'] != selected['seeds']:
        raise ValueError('Expected the completed three-seed experiment')
    films = baselines['by_film_bits']
    if set(films) != set(selected['paired_contrasts']['d4-r4']['by_film_seed_mean_bits']):
        raise ValueError('Results cover different films')
    center_bias = baselines['macro_film_bits']['train_kde_scott']
    names = [
        ('train_kde_scott', 'Center bias (KDE)'),
        ('neural_ar_d4', 'Visual AR'),
    ]
    benchmark = []
    for key, label in names:
        likelihood = baselines['macro_film_bits'][key]
        benchmark.append({'model': key, 'label': label, 'log2_likelihood': likelihood,
                          'ig_bits': likelihood - center_bias,
                          'training_seeds': [0] if key == 'neural_ar_d4' else [],
                          'selection': 'validation-selected' if key == 'neural_ar_d4' else 'training-only fit'})
    for key, label in [('static_d1', 'StaticD1'), ('d1', 'D1'), ('r4', 'R4'), ('d4', 'D4')]:
        likelihood = selected['macro_film_bits'][key]
        benchmark.append({'model': key, 'label': label, 'log2_likelihood': likelihood,
                          'ig_bits': likelihood - center_bias,
                          'training_seeds': [] if key == 'static_d1' else selected['seeds'],
                          'selection': 'unadapted' if key == 'static_d1' else 'validation-selected'})
    by_film = {}
    for film, row in films.items():
        d1 = row['d1_static'] + selected['paired_contrasts']['d1-static_d1']['by_film_seed_mean_bits'][film]
        d4 = d1 + selected['paired_contrasts']['d4-d1']['by_film_seed_mean_bits'][film]
        by_film[film] = {
            'center_bias_log2_likelihood': row['train_kde_scott'],
            'visual_ar_ig_bits': row['neural_ar_d4'] - row['train_kde_scott'],
            'd1_ig_bits': d1 - row['train_kde_scott'],
            'd4_ig_bits': d4 - row['train_kde_scott'],
            'd4_minus_r4_bits': selected['paired_contrasts']['d4-r4']['by_film_seed_mean_bits'][film],
        }
    return {
        'protocol': 'fovlanding-v1', 'scope': 'development_assumptions',
        'train_transitions': baselines['train_targets'],
        'validation_transitions': baselines['validation_targets'],
        'training_films': 10, 'validation_films': len(films),
        'metric': 'Film-macro mean log2(model probability / training KDE cell probability)',
        'center_bias_log2_likelihood': center_bias,
        'benchmark': benchmark, 'by_film': by_film,
        'selected_contrasts': selected['paired_contrasts'],
        'final_contrasts': final['paired_contrasts'],
        'final_ig_bits': {key: score - center_bias for key, score in final['macro_film_bits'].items()},
        'scientific_calibration_verified': False,
        'provenance': {
            'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
            'input_scan_sha256': selected['inputs']['scan_sha256'],
            'input_data_sha256': selected['inputs']['data_sha256'],
        },
        'limitations': [
            'Validation data also select checkpoints; this is not an independent test.',
            'DeepGaze uses three seeds; visual AR uses one seed and a different training budget.',
            'The independent stimulus units remain four films, not twelve film-seed combinations.',
            'Playback alignment and per-trial display geometry remain assumed.',
        ],
    }


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.25, 1]})
    keys = ['train_kde_scott', 'neural_ar_d4', 'static_d1', 'd1', 'r4', 'd4']
    rows = {row['model']: row for row in results['benchmark']}
    scores = [rows[key]['ig_bits'] for key in keys]
    labels = [rows[key]['label'] for key in keys]
    colors = ['#999999'] * 3 + ['#336699'] * 3
    left.barh(range(len(keys)), scores, color=colors, height=0.65)
    left.set_yticks(range(len(keys)), labels)
    left.invert_yaxis()
    left.set_xlim(0, max(scores) + 0.5)
    left.set_xlabel('Information gain above center bias (bits/transition)')
    left.set_title('Shared center-bias benchmark', loc='left')
    for y, score in enumerate(scores):
        left.text(score + 0.04, y, f'{score:.3f}', va='center', fontsize=9)
    names = ['batman forever', 'deep blue', 'quiz show', 'the march of the penguins']
    values = [results['by_film'][name]['d4_minus_r4_bits'] for name in names]
    labels = ['Batman Forever', 'Deep Blue', 'Quiz Show', 'March of the Penguins']
    right.axvline(0, color='#888888', linewidth=1)
    right.scatter(values, range(4), color='#336699', s=45, zorder=3)
    right.set_yticks(range(4), labels)
    right.invert_yaxis()
    right.set_xlim(-0.035, 0.14)
    right.set_xticks([0, 0.05, 0.10], ['0', '0.05', '0.10'])
    right.set_xlabel('D4 minus R4 (bits/transition)')
    right.set_title('Historical-content gain by film', loc='left')
    for y, value in enumerate(values):
        right.text(value + 0.006, y, f'{value:+.3f}', va='center', fontsize=9)
    fig.text(0.5, 0.025, 'Development validation: 4 films. DeepGaze: 3 seeds; visual AR: 1 seed. No independent test.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3)
    fig.savefig(output, dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('reports/results.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/results.png'))
    parser.add_argument('--refresh-from', type=Path, help='Completed experiment storage; aggregate records only')
    args = parser.parse_args()
    if args.refresh_from is not None:
        result = export_results(args.refresh_from)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    else:
        result = read(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(result, args.output)


if __name__ == '__main__':
    main()
