"""Causality, controls, safe retention/deduplication and selected-score contracts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run tests inside Slurm')

import numpy as np
import torch

from baseline import Population, cell_log_mass, tensors
from routeb.data import VideoInputs, record
from routeb.evaluation import Inputs
from routeb.retention import prune_intermediate
from scripts.compare_video import load_run
from scripts.deduplicate_static import deduplicate


def sample():
    return {'id': 'one', 'video': 'film', 'source_group': 'film', 'split': 'train',
            'input': {'history': [None, None, None, {'cell': [30, 40], 'xy': [.3, .4],
                      'start_relative_seconds': -.2, 'end_relative_seconds': 0., 'raw_relative_seconds': -.19}],
                      'frames': [None, {'index': 1, 'relative_seconds': -.6},
                                 {'index': 2, 'relative_seconds': -.3}, {'index': 3, 'relative_seconds': -.01}]},
            'target': {'cell': [32, 41]}}


class VideoContracts(unittest.TestCase):
    def test_completed_fovlanding_dedup_keeps_manifests_and_rejects_live_or_partial_runs(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temporary:
            runs, weights, manifests = [], [], []
            for seed in (0, 1):
                run = Path(temporary) / str(seed)
                checkpoint = run / 'checkpoints/checkpoint-000000-v000'
                adapter = checkpoint / 'adapter/adapter_model.safetensors'
                adapter.parent.mkdir(parents=True)
                adapter.write_bytes(b'identical synthetic initialization')
                protocol = {'architecture': 'single-gaze-adapter-fovlanding-v1',
                            'config': {'seed': seed, 'max_optimizer_steps': 1000, 'validation_every_updates': 100}}
                (run / 'protocol.json').write_text(json.dumps(protocol))
                result = {'status': 'complete', 'update': 1000, 'progress': {'validated_epochs': 10},
                          'frozen_weights_unchanged': True,
                          'protocol_sha256': hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest()}
                (run / 'result.json').write_text(json.dumps(result))
                manifest = checkpoint / 'manifest.json'
                manifest.write_text(json.dumps({'metadata': {'protocol': protocol},
                    'sha256': {'adapter/adapter_model.safetensors': hashlib.sha256(adapter.read_bytes()).hexdigest()}}))
                runs.append(run)
                weights.append(adapter)
                manifests.append((manifest, manifest.read_bytes()))
            self.assertEqual(len(deduplicate(training_runs=runs)['files']), 1)
            with (runs[0] / '.writer.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    deduplicate(training_runs=runs, apply=True)
            self.assertFalse(weights[0].samefile(weights[1]))
            deduplicate(training_runs=runs, apply=True)
            self.assertTrue(weights[0].samefile(weights[1]))
            for path, original in manifests:
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(deduplicate(training_runs=runs)['files'], [])
            result['progress']['validated_epochs'] = 9
            (runs[1] / 'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'fully validated'):
                deduplicate(training_runs=runs, apply=True)

    def test_b1_repeated_frame_paths_compare_the_same_type(self):
        first, second = sample(), sample()
        second['id'] = 'two'
        inputs = VideoInputs.__new__(VideoInputs)
        inputs.ids = {'train': ['one', 'two'], 'validation': []}
        inputs.samples = {'one': first, 'two': second}
        exported = [{'images': ['a.png', 'b.png', 'c.png']} for _ in range(2)]
        with patch.object(Inputs, 'rows', side_effect=lambda variant, split: exported if split == 'train' else []):
            rows, paths = inputs.baseline_rows('d4')
            self.assertEqual(len(rows), 2)
            self.assertEqual(paths, {('film', 1): Path('a.png'), ('film', 2): Path('b.png'),
                                     ('film', 3): Path('c.png')})
            exported[1]['images'][0] = 'different.png'
            with self.assertRaisesRegex(ValueError, 'Conflicting image paths'):
                inputs.baseline_rows('d4')

    def test_controls_share_text_times_and_missing_pattern(self):
        row = sample()
        paths = {1: 'a.png', 2: 'b.png', 3: 'c.png'}
        d1, d4, r4 = [record(row, variant, paths) for variant in ['d1', 'd4', 'r4']]
        self.assertEqual(d4['conversations'], r4['conversations'])
        self.assertEqual(d1['images'], ['c.png'])
        self.assertEqual(d4['images'], ['a.png', 'b.png', 'c.png'])
        self.assertEqual(r4['images'], ['c.png'] * 3)
        text = d4['conversations'][0]['value']
        self.assertIn('t=-0.010000', text)
        self.assertIn('Frame 1: missing.', text)
        self.assertNotIn('(32, 41)', text)
        for mutate in ['frame', 'history']:
            changed = copy.deepcopy(row)
            if mutate == 'frame':
                changed['input']['frames'][-1]['relative_seconds'] = .1
            else:
                changed['input']['history'][-1]['end_relative_seconds'] = .1
            with self.assertRaises(ValueError):
                record(changed, 'd4', paths)

    def test_b1_normalization_uses_only_training_data_and_likelihood_learns(self):
        train, val = sample(), sample()
        val.update(id='two', video='other', split='validation')
        cache = {(video, index): np.full(2048, index + offset, np.float32)
                 for video, offset in [('film', 0), ('other', 10)] for index in [1, 2, 3]}
        data, norm, _ = tensors([train, val], cache)
        changed = {key: value + (100 if key[0] == 'other' else 0) for key, value in cache.items()}
        _, norm2, _ = tensors([train, val], changed)
        for key in norm:
            np.testing.assert_array_equal(norm[key], norm2[key])
        model = Population()
        loss = -cell_log_mass(model(*data[:5]), data[5]).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_retention_keeps_epochs_best_initial_and_two_latest(self):
        protocol = {'architecture': 'single-gaze-adapter-v2'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for update, validated in [(0, 0), (25, 0), (50, 0), (100, 1), (125, 1), (150, 1), (175, 1)]:
                path = root / f'checkpoint-{update:06d}-v{validated:03d}'
                path.mkdir()
                manifest = {'update': update, 'metadata': {'protocol': protocol, 'progress': {
                    'validated_epochs': validated, 'best_checkpoint': 'checkpoints/checkpoint-000100-v001' if validated else None}}}
                (path / 'manifest.json').write_text(json.dumps(manifest))
            removed = prune_intermediate(root, protocol, 100)
            self.assertEqual(set(removed), {'checkpoint-000025-v000', 'checkpoint-000050-v000', 'checkpoint-000125-v001'})
            self.assertTrue((root / 'checkpoint-000100-v001/manifest.json').exists())
            self.assertEqual(len(list((root / 'retired-manifests').glob('*.json'))), 3)
            with self.assertRaises(ValueError):
                prune_intermediate(root, {'architecture': 'legacy'}, 100)

    def test_dedup_preserves_paths_bytes_and_manifest_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, manifests = [], []
            for variant in ['d1', 'd4']:
                folder = root / 'variants' / variant / 'checkpoints/checkpoint-000469-v001'
                path = folder / 'adapter/static/adapter_model.safetensors'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'not model data: synthetic identical content' * 100)
                manifest = folder / 'manifest.json'
                manifest.write_text(json.dumps({'sha256': {str(path.relative_to(folder)):
                                                          hashlib.sha256(path.read_bytes()).hexdigest()}}))
                weights.append(path)
                manifests.append((manifest, manifest.read_bytes()))
            plan = deduplicate(root)
            self.assertEqual(len(plan['files']), 1)
            self.assertFalse(weights[0].samefile(weights[1]))
            # DSS may report zero allocated blocks for a freshly written file.
            allocated = weights[1].stat().st_blocks * 512
            result = deduplicate(root, apply=True)
            self.assertTrue(weights[0].samefile(weights[1]))
            self.assertEqual(result['reclaimed_bytes'], allocated)
            for path, original in manifests:
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(deduplicate(root, apply=True)['files'], [])
            weights[0].write_bytes(b'corrupt shared inode')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                deduplicate(root)

    def test_v2_comparison_selects_requested_b1_budget_and_checks_prediction_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {'architecture': 'resnet18-gru-mdn-v2'}
            (root / 'protocol.json').write_text(json.dumps(protocol))
            metadata = {'one': {'id': 'one', 'video': 'film', 'source_group': 'film'}}
            epochs = []
            for epoch, bits in [(1, -4.), (2, -3.), (3, -5.)]:
                path = root / f'epoch-{epoch}.jsonl'
                row = {**metadata['one'], 'bits': bits, 'prior_bits': -6., 'ig_bits': bits + 6}
                path.write_text(json.dumps(row) + '\n')
                epochs.append({'epoch': epoch, 'mean_bits': bits, 'predictions': path.name,
                               'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            (root / 'result.json').write_text(json.dumps({'status': 'complete', 'epochs': epochs,
                'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()}))
            self.assertEqual(load_run(root, 3, True, metadata)[1], 2)
            self.assertEqual(load_run(root, 1, True, metadata)[1], 1)
            (root / 'epoch-2.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):
                load_run(root, 3, True, metadata)


if __name__ == '__main__':
    unittest.main()
