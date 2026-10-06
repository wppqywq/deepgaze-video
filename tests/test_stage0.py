"""Scientific interface checks with explicitly synthetic test inputs."""
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from stage0 import digit_log_mass, frame_slots, gmm_grid, grid_log_mass, quantize


class ScoringTests(unittest.TestCase):
    def test_screen_boundaries(self):
        np.testing.assert_array_equal(quantize([[0, 0], [.01, .999999]]), [[0, 0], [1, 99]])
        for xy in ([1, .5], [-.001, .2], [np.nan, 0]):
            with self.assertRaises(ValueError):
                quantize(xy)

    def test_digit_probability_matches_enumerated_grid(self):
        rng = np.random.default_rng(9)
        # Prefix-dependent distributions test the autoregressive product.
        logits = rng.normal(size=(1111, 10))
        probabilities = []
        target_logits = None
        for x in range(100):
            for y in range(100):
                a, b, c, d = x//10, x%10, y//10, y%10
                selected = logits[[0, 1+a, 11+10*a+b, 111+100*a+10*b+c]]
                probabilities.append(np.exp(digit_log_mass(selected, [x, y])))
                if (x, y) == (32, 67):
                    target_logits = selected
        self.assertAlmostEqual(sum(probabilities), 1, places=12)
        grid = np.array(probabilities).reshape(100, 100).T
        self.assertAlmostEqual(grid_log_mass(grid, [32, 67]), digit_log_mass(target_logits, [32, 67]))
        with self.assertRaises(ValueError):
            digit_log_mass(np.zeros((4, 9)), [32, 67])

    def test_gmm_cell_integral_and_screen_conditioning(self):
        weights = [.3, .7]; means = [[-.05, .3], [.8, 1.1]]; scales = [[.2, .4], [.3, .2]]
        grid = gmm_grid(weights, means, scales)
        def integral(lo, hi, m, s):
            return quad(lambda v: norm.pdf(v, loc=m, scale=s), lo, hi, epsabs=1e-12)[0]
        expected = sum(w*integral(.21,.22,m[0],s[0])*integral(.73,.74,m[1],s[1]) for w,m,s in zip(weights,means,scales))
        screen = sum(w*integral(0,1,m[0],s[0])*integral(0,1,m[1],s[1]) for w,m,s in zip(weights,means,scales))
        self.assertAlmostEqual(grid[73,21], expected/screen, places=14)
        self.assertAlmostEqual(grid.sum(), 1)

    def test_pts_cutoff_and_missing_slots(self):
        pts = np.array([0., .03, .07, .11, .5, 1.02])
        self.assertEqual(frame_slots(pts, .07)[-1]['index'], 2)
        slots = frame_slots(pts, .5)
        self.assertIsNone(slots[0])
        self.assertTrue(all(s is None or s['pts_seconds'] <= .5 for s in slots))
        self.assertEqual(frame_slots(pts, .069)[-1]['index'], 1)

    def test_non_aligned_window_start_keeps_four_frames(self):
        pts = np.arange(100) / 30
        slots = frame_slots(pts, 1.517)
        self.assertTrue(all(s is not None for s in slots))
        self.assertEqual(slots[0]['index'], 16)
        self.assertTrue(all(.517 <= s['pts_seconds'] <= 1.517 for s in slots))


if __name__ == '__main__':
    unittest.main()
