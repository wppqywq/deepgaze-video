"""Paired validation scoring and export, shared by profiling and pilots."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from gpu_train import eval_mass, file_hash, packed, require
from stage0 import grid_log_mass


def macro_film_bits(groups):
    """Give each source film equal weight, independently of target count."""
    values = [group['mean_bits'] for group in groups.values()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Missing or nonfinite film scores')
    return sum(values) / len(values)


class Inputs:
    def __init__(self, outputs, scan_report):
        require(bool(os.environ.get('SLURM_JOB_ID')), 'Load experiment data inside Slurm')
        self.outputs = Path(outputs)
        self.vlm = self.outputs / 'vlm'
        self.scan = json.loads(Path(scan_report).read_text())
        require(self.scan['status'] == 'Full processor scan passed', 'Full scan required')
        for name, expected in self.scan['data_sha256'].items():
            require(file_hash(self.vlm / name) == expected, 'Audited input changed: ' + name)
        self.ids = json.loads((self.vlm / 'sample_ids.json').read_text())
        require(len(self.ids['train']) == 7489 and len(self.ids['validation']) == 2489, 'Wrong split sizes')
        require(len(set(self.ids['train'] + self.ids['validation'])) == 9978, 'Duplicate or overlapping targets')
        with (self.outputs / 'samples.jsonl').open() as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
        self.samples = {row['id']: row for row in samples}
        require(len(self.samples) == len(samples) == 9978, 'Wrong metadata population')
        require(set(self.samples) == set(self.ids['train'] + self.ids['validation']), 'Metadata/pairing mismatch')
        baseline = json.loads((self.outputs / 'b1-seed-1/config.json').read_text())
        require(file_hash(self.outputs / 'samples.jsonl') == baseline['samples_sha256'], 'B1 and VLM metadata differ')
        self.prior = np.load(self.outputs / 'training-prior.npy', allow_pickle=False)
        require(self.prior.shape == (100, 100) and bool(np.isfinite(self.prior).all()) and
                bool((self.prior > 0).all()) and abs(float(self.prior.sum())-1) < 1e-10, 'Invalid training prior')
        self.provenance = {'scan_sha256': file_hash(scan_report),
                           'pairing_sha256': file_hash(self.vlm / 'sample_ids.json'),
                           'samples_sha256': file_hash(self.outputs / 'samples.jsonl'),
                           'prior_sha256': file_hash(self.outputs / 'training-prior.npy'),
                           'vlm_sha256': self.scan['data_sha256']}

    def rows(self, variant, split):
        require(variant in ['d1', 'd4', 'r4'] and split in ['train', 'validation'], 'Invalid variant/split')
        rows = json.loads((self.vlm / (variant + '_' + split + '.json')).read_text())
        require(len(rows) == len(self.ids[split]), 'Input/pairing length differs')
        for row, sample_id in zip(rows, self.ids[split]):
            x, y = self.samples[sample_id]['target']['cell']
            require(row['conversations'][1]['value'] == '[({:02d}, {:02d})]'.format(x, y), 'Target label differs')
        return rows


def evaluate(model, processor, inputs, variant, indices, output, rows=None, score_fn=eval_mass):
    """Score actual target mass; write one complete paired JSONL atomically.

    Timings include preprocessing, model score, metadata/prior and row encoding.
    No target ID or metadata is added to the model prompt.
    """
    output = Path(output)
    require(not output.exists(), 'Refusing to overwrite predictions')
    indices = list(indices)
    require(indices and len(indices) == len(set(indices)), 'Need unique evaluation indices')
    rows = inputs.rows(variant, 'validation') if rows is None else rows
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.incomplete')
    require(not temporary.exists(), 'Unfinished evaluation exists')
    by_group = defaultdict(list)
    scores, prior_scores, elapsed = [], [], []
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with temporary.open('x') as stream:
        for index in indices:
            before = time.perf_counter()
            sample_id = inputs.ids['validation'][index]
            sample = inputs.samples[sample_id]
            score = score_fn(model, packed(processor, rows[index], inputs.scan['max_length'])) / math.log(2)
            prior = float(grid_log_mass(inputs.prior, sample['target']['cell']) / math.log(2))
            require(math.isfinite(score) and math.isfinite(prior), 'Non-finite validation score')
            record = {'id': sample_id, 'video': sample['video'], 'source_group': sample['source_group'],
                      'bits': score, 'prior_bits': prior, 'ig_bits': score-prior}
            if inputs.provenance.get('scope') == 'development_assumptions':
                record.update(target_id=sample_id, source_film_id=sample['source_group'],
                              observer_id=sample['observer'], split='validation', log2p=score,
                              protocol_version='fovlanding-v1')
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            torch.cuda.synchronize()
            seconds = time.perf_counter()-before
            elapsed.append(seconds)
            scores.append(score)
            prior_scores.append(prior)
            by_group[sample['source_group']].append({'bits': score, 'prior_bits': prior, 'seconds': seconds})
            if len(scores) % 500 == 0:
                print(json.dumps({'validation_variant': variant, 'targets_scored': len(scores),
                                  'total': len(indices), 'elapsed_seconds': time.perf_counter() - started}), flush=True)
        stream.flush()
    os.rename(temporary, output)
    predictions_sha256 = file_hash(output)
    groups = {group: {'targets': len(items),
                'mean_bits': float(np.mean([item['bits'] for item in items])),
                'mean_ig_bits': float(np.mean([item['bits']-item['prior_bits'] for item in items])),
                'mean_target_seconds': float(np.mean([item['seconds'] for item in items]))}
                for group, items in sorted(by_group.items())}
    return {'targets': len(indices), 'mean_bits': float(np.mean(scores)),
            'macro_film_bits': macro_film_bits(groups),
            'mean_prior_bits': float(np.mean(prior_scores)),
            'mean_ig_bits': float(np.mean(np.array(scores)-np.array(prior_scores))),
            'elapsed_seconds': time.perf_counter()-started,
            'target_seconds': elapsed, 'indices': indices,
            'by_source_group': groups,
            'predictions_sha256': predictions_sha256}
