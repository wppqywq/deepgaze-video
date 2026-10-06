"""Audit full pilot exports and create a joint development report inside Slurm.

This produces evidence for a human/agent review, never continuation authorization.
It reads the campaign without modifying training code, states or selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from routeb.comparison import checked_scores, mean, select_evaluation, summarize


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def within(root, name):
    path = (root/name).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact path escapes its root')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--epoch', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--best', action='store_true', help='Select best within the same common epoch budget')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Run data processing inside Slurm')
    require(not args.output.exists(), 'Refusing to overwrite a comparison')
    config_path = args.campaign/'code/pilot_seed1.json'
    config = read_json(config_path)
    require(config['variants'] == ['d1', 'd4', 'r4'], 'Unexpected common variants')
    pairing_path = args.outputs/'vlm/sample_ids.json'
    ids = read_json(pairing_path)['validation']
    require(len(ids) == len(set(ids)) == config['validation_targets'], 'Invalid validation IDs')
    samples_path = args.outputs/'samples.jsonl'
    sample_rows = records(samples_path)
    samples = {row['id']: row for row in sample_rows}
    require(len(samples) == len(sample_rows), 'Duplicate metadata')
    metadata = {sample_id: samples[sample_id] for sample_id in ids}
    models, reports, protocols, costs = {}, {}, {}, {}
    shared_protocol = None
    for variant in config['variants']:
        folder = args.campaign/'variants'/variant
        protocol = read_json(folder/'protocol.json')
        require(protocol['variant'] == variant and protocol['config'] == config and
                protocol['config_sha256'] == digest(config_path), 'Protocol/config mismatch')
        require(protocol['inputs']['pairing_sha256'] == digest(pairing_path) and
                protocol['inputs']['samples_sha256'] == digest(samples_path), 'Input metadata changed')
        common = {key: value for key, value in protocol.items() if key != 'variant'}
        if shared_protocol is None:
            shared_protocol = common
        require(common == shared_protocol, 'Variants used different numerical/source/data protocols')
        protocols[variant] = digest(folder/'protocol.json')
        update = args.epoch*math.ceil(config['train_targets']/config['effective_batch'])
        checkpoint = folder/'checkpoints'/f'checkpoint-{update:06d}-v{args.epoch:03d}'
        manifest = read_json(checkpoint/'manifest.json')
        require(manifest['update'] == update and manifest['metadata']['protocol'] == protocol, 'Checkpoint mismatch')
        progress = manifest['metadata']['progress']
        require(progress['validated_epochs'] == args.epoch and
                progress['seen_targets'] == args.epoch*config['train_targets'] and
                progress['epoch_targets'] == 0, 'Unequal or incomplete training budget')
        require(manifest['schema'] == 1 and 'training-state.pt' in manifest['sha256'], 'Incomplete checkpoint manifest')
        for name, expected in manifest['sha256'].items():
            require(digest(within(checkpoint, name)) == expected, 'Checkpoint artifact changed: '+name)
        chosen = select_evaluation(progress['evaluations'], args.epoch, args.best)
        best = select_evaluation(progress['evaluations'], args.epoch, True)
        require(progress['best_epoch'] == best['epoch'] and progress['best_bits'] == best['mean_bits'], 'Best selection mismatch')
        steps = math.ceil(config['train_targets']/config['effective_batch'])
        require(progress['best_checkpoint'] == f"checkpoints/checkpoint-{best['epoch']*steps:06d}-v{best['epoch']:03d}",
                'Best checkpoint pointer differs from selection')
        selected_checkpoint = folder/'checkpoints'/f"checkpoint-{chosen['epoch']*steps:06d}-v{chosen['epoch']:03d}"
        selected_manifest = read_json(selected_checkpoint/'manifest.json')
        require(selected_manifest['metadata']['protocol'] == protocol and
                selected_manifest['update'] == chosen['epoch']*steps and
                selected_manifest['metadata']['progress']['validated_epochs'] == chosen['epoch'],
                'Selected model checkpoint differs from evaluation epoch')
        if selected_checkpoint != checkpoint:
            require(selected_manifest['schema'] == 1 and 'training-state.pt' in selected_manifest['sha256'],
                    'Selected checkpoint is incomplete')
            for name, expected in selected_manifest['sha256'].items():
                require(digest(within(selected_checkpoint, name)) == expected, 'Selected checkpoint artifact changed: '+name)
        report_path = within(args.campaign, chosen['report'])
        require(digest(report_path) == chosen['sha256'], 'Evaluation report changed')
        evaluation = read_json(report_path)
        require(evaluation['epoch'] == chosen['epoch'] and evaluation['targets'] == len(ids) and
                evaluation['indices'] == list(range(len(ids))), 'Incomplete validation scope')
        prediction_path = within(args.campaign, evaluation['predictions'])
        require(digest(prediction_path) == evaluation['predictions_sha256'], 'Predictions changed')
        scores = checked_scores(records(prediction_path), metadata)
        require(math.isclose(mean(row['bits'] for row in scores.values()), chosen['mean_bits'], abs_tol=1e-9, rel_tol=0), 'Export/selection mean differs')
        require(math.isclose(evaluation['mean_bits'], chosen['mean_bits'], abs_tol=1e-9, rel_tol=0), 'Report/selection mean differs')
        models[variant] = scores
        reports[variant] = {'epoch': chosen['epoch'], 'path': chosen['report'], 'sha256': chosen['sha256'],
                            'predictions_sha256': evaluation['predictions_sha256'],
                            'checkpoint': str(selected_checkpoint.relative_to(args.campaign)),
                            'checkpoint_manifest_sha256': digest(selected_checkpoint/'manifest.json'),
                            'budget_checkpoint_manifest_sha256': digest(checkpoint/'manifest.json')}
        completed_jobs = [read_json(path) for path in sorted((folder/'jobs').glob('*/result.json'))]
        costs[variant] = {'completed_driver_seconds_to_date': sum(row['elapsed_seconds'] for row in completed_jobs),
                          'selected_validation_seconds': evaluation['elapsed_seconds'],
                          'job_ids': [row['job_id'] for row in completed_jobs],
                          'note': 'Driver time includes model setup, hashing, checkpoint and validation overhead, but excludes queue wait, imports and batch prechecks. May include later segments; use Slurm accounting for total allocated time.'}

    reference_path = args.outputs/'scores.jsonl'
    reference_rows = records(reference_path)
    reference = {row['id']: row for row in reference_rows}
    require(len(reference) == len(reference_rows) and set(reference) == set(ids), 'B0 population differs')
    for name, score_key in [('prior', 'prior_bits'), ('inertia', 'inertia_bits')]:
        normalized = [{**row, 'bits': row[score_key], 'ig_bits': row[score_key]-row['prior_bits']} for row in reference_rows]
        models[name] = checked_scores(normalized, metadata)
    b1_config_path = args.outputs/'b1-seed-1/config.json'
    b1_result_path = args.outputs/'b1-seed-1/result.json'
    b1_scores_path = args.outputs/'b1-seed-1/scores.jsonl'
    b1_config, b1_result = read_json(b1_config_path), read_json(b1_result_path)
    require(b1_config['samples_sha256'] == digest(samples_path) and b1_result['status'] == 'complete', 'B1 provenance/status differs')
    b1_rows = [{**row, 'prior_bits': reference[row['id']]['prior_bits']} for row in records(b1_scores_path)]
    models['b1'] = checked_scores(b1_rows, metadata)
    require(math.isclose(mean(row['bits'] for row in models['b1'].values()), b1_result['validation_bits'], abs_tol=1e-9, rel_tol=0), 'B1 export/result differs')
    for scores in models.values():
        for sample_id in ids:
            require(math.isclose(scores[sample_id]['prior_bits'], reference[sample_id]['prior_bits'], abs_tol=1e-8, rel_tol=0), 'Training prior differs across models')
    result = {'status': 'Paired development comparison ready for joint review', 'job_id': os.environ['SLURM_JOB_ID'],
              'common_epoch_budget': args.epoch, 'selection': 'best_within_common_budget' if args.best else 'same_epoch',
              'config_sha256': digest(config_path), 'protocol_sha256': protocols, 'selected_reports': reports,
              'summary': summarize(models, metadata), 'costs': costs,
              'reference_sha256': {str(path.relative_to(args.outputs)): digest(path) for path in
                                  [reference_path, b1_config_path, b1_result_path, b1_scores_path]},
              'source_sha256': {str(path.relative_to(Path(__file__).resolve().parent.parent)): digest(path) for path in
                               [Path(__file__).resolve(), Path(__file__).resolve().parent.parent/'routeb/comparison.py']},
              'b1_selection_budget': {'epochs': b1_config['epochs'], 'best_epoch': b1_result['best_epoch'],
                                     'note': 'Transferred development reference; its search/training budget differs from this pilot.'},
              'limitations': ['Development validation only; no test evaluation.',
                             'Overlapping targets are not independent experiments; no target-level confidence intervals.',
                             'Only four validation source films; temporal benefit and generalization remain unproven.',
                             'This report does not authorize continued training or model expansion.']}
    args.output.mkdir(parents=True)
    with (args.output/'paired-scores.jsonl').open('x') as stream:
        for sample_id in ids:
            row = {'id': sample_id, 'video': metadata[sample_id]['video'], 'source_group': metadata[sample_id]['source_group'],
                   'bits': {name: scores[sample_id]['bits'] for name, scores in models.items()},
                   'prior_bits': reference[sample_id]['prior_bits']}
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    result['paired_scores_sha256'] = digest(args.output/'paired-scores.jsonl')
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Paired pilot development comparison', '',
             f"Common budget: {args.epoch} epoch(s). Selection: {result['selection']}.", '',
             '| Scope | Targets | D1 bits | D4 bits | R4 bits | B1 bits | D4-D1 | D4-R4 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    scopes = {'All targets': result['summary']['all_targets'], **result['summary']['by_source_group']}
    for name, scope in scopes.items():
        values = [scope['models'][model]['bits'] for model in ['d1', 'd4', 'r4', 'b1']]
        values += [scope['paired_differences_bits'][key] for key in ['d4-d1', 'd4-r4']]
        lines.append('| '+name.replace('|', '/')+' | '+str(scope['targets'])+' | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    lines += ['', 'Higher bits/target is better. B1 selected its best checkpoint over '+str(b1_config['epochs'])+' epochs; its budget differs.', '']
    lines += ['- '+note for note in result['limitations']]
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
