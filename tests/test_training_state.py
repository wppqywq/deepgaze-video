"""Run in Slurm: python -m unittest -v test_training_state.

Small CPU tests of resume mechanics; not evidence of 8B CUDA reproducibility.
"""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run tests inside a Slurm allocation')

import random
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from training_state import (SampleOrder, mean_backward, restore_checkpoint,
                            save_checkpoint, trainable_state)


def make_model():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.25,
                                         target_modules=['0', '2']))


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_epoch_tail_is_not_dropped_or_mixed(self):
        order = SampleOrder(range(19), seed=3)
        first, tail = order.next_batch(16), order.next_batch(16)
        self.assertEqual((len(first), len(tail)), (16, 3))
        self.assertEqual(sorted(first + tail), list(range(19)))
        self.assertEqual(order.epoch, 0)
        checkpoint = order.state_dict()
        expected = order.next_batch(16)
        restored = SampleOrder(range(19), seed=999)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.next_batch(16), expected)
        self.assertEqual(restored.epoch, 1)

    def test_partial_batch_gradient_is_sample_mean(self):
        x = torch.tensor([1., 2., 5.])
        actual = torch.nn.Parameter(torch.tensor(0.7))
        expected = torch.nn.Parameter(actual.detach().clone())
        loss = mean_backward(lambda i: (actual*x[i]-1).square(), list(range(3)))
        reference = (expected*x-1).square().mean()
        reference.backward()
        torch.testing.assert_close(actual.grad, expected.grad)
        self.assertAlmostEqual(loss, float(reference.detach()), places=5)
        with self.assertRaises(ValueError):
            mean_backward(lambda i: actual, [])

    def test_next_update_matches_after_fresh_model_restore(self):
        random.seed(31)
        np.random.seed(31)
        model = make_model()
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        order = SampleOrder(range(19), seed=7)
        x = torch.arange(57, dtype=torch.float32).reshape(19, 3) / 57
        metadata = {'dataset': 'synthetic-test', 'seed': 7, 'effective_batch': 16}

        def update(m, opt, sched, sampler):
            indices = sampler.next_batch(16)
            loss = mean_backward(lambda i: (m(x[i:i+1]).squeeze()-0.4).square(), indices)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            return indices, loss

        update(model, optimizer, scheduler, order)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint-1'
            save_checkpoint(path, model, optimizer, scheduler, order, 1, metadata)
            reference_batch, reference_loss = update(model, optimizer, scheduler, order)
            reference = trainable_state(model)
            reference_opt = optimizer.state_dict()
            reference_random = (random.random(), float(np.random.random()), torch.rand(5))
            restored = make_model()
            restored.train()
            opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.8)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
            sampler = SampleOrder(range(19), seed=123)
            self.assertEqual(restore_checkpoint(path, restored, opt, sched, sampler, metadata), 1)
            batch, loss = update(restored, opt, sched, sampler)
            self.assertEqual(batch, reference_batch)
            self.assertEqual(loss, reference_loss)
            for name, param in trainable_state(restored).items():
                torch.testing.assert_close(param, reference[name], atol=0, rtol=0)
            self.assertEqual(sched.state_dict(), scheduler.state_dict())
            self.assertEqual(opt.state_dict()['param_groups'], reference_opt['param_groups'])
            for index, state in opt.state_dict()['state'].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, reference_opt['state'][index][name], atol=0, rtol=0)
            self.assertEqual(random.random(), reference_random[0])
            self.assertEqual(float(np.random.random()), reference_random[1])
            torch.testing.assert_close(torch.rand(5), reference_random[2], atol=0, rtol=0)
            self.assertEqual(sampler.next_batch(16), order.next_batch(16))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                restore_checkpoint(path, restored, opt, sched, sampler, {'dataset': 'other'})
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, restored, opt, sched, sampler, 2, metadata)
            with (path / 'training-state.pt').open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash/path'):
                restore_checkpoint(path, restored, opt, sched, sampler, metadata)


if __name__ == '__main__':
    unittest.main()
