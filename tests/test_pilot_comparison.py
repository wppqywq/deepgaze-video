"""Pairing corruption, selection ties and the complete report-file contract."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run tests inside Slurm')

from routeb.comparison import checked_scores, select_evaluation, summarize


class PilotComparison(unittest.TestCase):
    def setUp(self):
        self.metadata = {str(i): {'id': str(i), 'video': 'film', 'source_group': 'source'} for i in range(2)}
        self.rows = [{**row, 'bits': -2., 'prior_bits': -4., 'ig_bits': 2.} for row in self.metadata.values()]

    def test_pairing_rejects_duplicate_missing_metadata_and_invalid_scores(self):
        for rows in [self.rows[:1], [self.rows[0], self.rows[0]],
                     [self.rows[0], {**self.rows[1], 'video': 'wrong'}],
                     [self.rows[0], {**self.rows[1], 'bits': float('nan')}],
                     [self.rows[0], {**self.rows[1], 'ig_bits': 3.}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                checked_scores(rows, self.metadata)

    def test_pairing_is_by_id_not_export_position(self):
        models = {}
        for name, shift in [('d1', 0), ('d4', .5), ('r4', -.5), ('b1', .25)]:
            rows = [{**row, 'bits': row['bits']+shift, 'ig_bits': row['ig_bits']+shift} for row in reversed(self.rows)]
            models[name] = checked_scores(rows, self.metadata)
        summary = summarize(models, self.metadata)
        self.assertEqual(summary['all_targets']['paired_differences_bits']['d4-d1'], .5)
        self.assertEqual(summary['by_source_group']['source']['paired_differences_bits']['d4-r4'], 1.)

    def test_best_tie_uses_earliest_and_missing_epoch_is_rejected(self):
        rows = [{'epoch': 2, 'mean_bits': -1}, {'epoch': 1, 'mean_bits': -1}]
        self.assertEqual(select_evaluation(rows, 2, True)['epoch'], 1)
        self.assertEqual(select_evaluation(rows, 2, False)['epoch'], 2)
        with self.assertRaises(ValueError):
            select_evaluation(rows[:1], 2, True)

    def test_report_reads_real_manifest_field_and_rejects_tampered_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, outputs = root/'campaign', root/'outputs'

            def write(path, value, lines=False):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))
                return hashlib.sha256(path.read_bytes()).hexdigest()

            config = {'variants': ['d1', 'd4', 'r4'], 'train_targets': 3, 'validation_targets': 2, 'effective_batch': 2}
            config_hash = write(campaign/'code/pilot_seed1.json', config)
            pairing_hash = write(outputs/'vlm/sample_ids.json', {'validation': list(self.metadata)})
            samples_hash = write(outputs/'samples.jsonl', list(self.metadata.values()), True)
            write(outputs/'scores.jsonl', [{**row, 'prior_bits': -4., 'inertia_bits': -3.} for row in self.metadata.values()], True)
            write(outputs/'b1-seed-1/config.json', {'samples_sha256': samples_hash, 'epochs': 20})
            write(outputs/'b1-seed-1/result.json', {'status': 'complete', 'validation_bits': -2., 'best_epoch': 3})
            write(outputs/'b1-seed-1/scores.jsonl', self.rows, True)
            for variant in config['variants']:
                folder = campaign/'variants'/variant
                protocol = {'variant': variant, 'config': config, 'config_sha256': config_hash,
                            'inputs': {'pairing_sha256': pairing_hash, 'samples_sha256': samples_hash}}
                write(folder/'protocol.json', protocol)
                prediction = folder/'jobs/1/validation-epoch-001.jsonl'
                prediction_hash = write(prediction, self.rows, True)
                evaluation = folder/'jobs/1/validation-epoch-001.json'
                evaluation_hash = write(evaluation, {'epoch': 1, 'targets': 2, 'indices': [0, 1],
                    'mean_bits': -2., 'predictions': str(prediction.relative_to(campaign)),
                    'predictions_sha256': prediction_hash, 'elapsed_seconds': .1})
                chosen = {'epoch': 1, 'mean_bits': -2., 'report': str(evaluation.relative_to(campaign)), 'sha256': evaluation_hash}
                checkpoint = folder/'checkpoints/checkpoint-000002-v001'
                state_hash = write(checkpoint/'training-state.pt', {'fixture': 'not a model; comparison does not deserialize state'})
                write(checkpoint/'manifest.json', {'schema': 1, 'update': 2, 'sha256': {'training-state.pt': state_hash},
                    'metadata': {'protocol': protocol, 'progress': {'validated_epochs': 1, 'seen_targets': 3,
                        'epoch_targets': 0, 'evaluations': [chosen], 'best_epoch': 1, 'best_bits': -2.,
                        'best_checkpoint': 'checkpoints/checkpoint-000002-v001'}}})
            script = Path(__file__).resolve().parent.parent/'scripts/compare_pilot.py'
            command = [sys.executable, str(script), '--campaign', str(campaign), '--outputs', str(outputs), '--output']
            run = subprocess.run(command+[str(root/'comparison')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root/'comparison/comparison.json').read_text())
            self.assertEqual(result['summary']['all_targets']['targets'], 2)
            self.assertEqual(result['b1_selection_budget']['epochs'], 20)
            prediction.write_text(prediction.read_text()+'\n')
            run = subprocess.run(command+[str(root/'corrupt-comparison')], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Predictions changed', run.stderr)
            self.assertFalse((root/'corrupt-comparison').exists())


if __name__ == '__main__':
    unittest.main()
