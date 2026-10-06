"""Exercise interrupted saves and epoch-boundary selection inside Slurm."""
import json
import os
from pathlib import Path
import tempfile
import unittest

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run tests inside Slurm')

from routeb.checkpoints import checkpoint_name, latest_checkpoint, quarantine_incomplete


class PilotCheckpoints(unittest.TestCase):
    def publish(self, root, update, validated, protocol):
        path = root / checkpoint_name(update, validated)
        path.mkdir()
        (path / 'manifest.json').write_text(json.dumps({'update': update, 'metadata': {
            'protocol': protocol, 'progress': {'validated_epochs': validated}}}))
        return path

    def test_interrupted_newer_save_does_not_hide_last_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = self.publish(root, 25, 0, {'seed': 1})
            partial = root / (checkpoint_name(50, 0) + '.incomplete')
            partial.mkdir()
            (partial / 'partial-data').write_text('preserve me')
            self.assertEqual(latest_checkpoint(root, {'seed': 1})[0], saved)
            moved = quarantine_incomplete(root, 'test-job')
            self.assertEqual(len(moved), 1)
            self.assertEqual((root / moved[0] / 'partial-data').read_text(), 'preserve me')
            self.assertFalse(partial.exists())
            newer = self.publish(root, 50, 0, {'seed': 1})
            self.assertEqual(latest_checkpoint(root, {'seed': 1})[0], newer)

    def test_validated_boundary_wins_without_extra_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.publish(root, 469, 0, {'seed': 1})
            validated = self.publish(root, 469, 1, {'seed': 1})
            self.assertEqual(latest_checkpoint(root, {'seed': 1})[0], validated)

    def test_changed_protocol_and_corrupt_counter_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = self.publish(root, 25, 0, {'seed': 1})
            with self.assertRaisesRegex(RuntimeError, 'protocol'):
                latest_checkpoint(root, {'seed': 2})
            manifest = json.loads((saved / 'manifest.json').read_text())
            manifest['update'] = 26
            (saved / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'counters'):
                latest_checkpoint(root, {'seed': 1})


if __name__ == '__main__':
    unittest.main()
