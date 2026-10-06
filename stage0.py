"""Shared grid probability and causal frame-selection functions."""
import numpy as np
from scipy.special import logsumexp, ndtr, log_ndtr

def gaussian_cell_log_mass(means, scales, cells):
    """Batched diagonal Gaussian cell mass, conditioned on the unit rectangle."""
    means, scales, cells = map(lambda value: np.asarray(value, dtype=float), (means, scales, cells))
    if (means.ndim != 2 or means.shape[1] != 2 or cells.shape != means.shape or
            scales.shape not in ((2,), means.shape) or not all(np.isfinite(v).all() for v in (means, scales, cells)) or
            (scales <= 0).any() or (cells != np.floor(cells)).any() or ((cells < 0) | (cells >= 100)).any()):
        raise ValueError('Expected finite Gaussian parameters and integer grid cells')

    def interval(lower, upper):
        # Reflect positive tails to avoid subtracting CDFs rounded to one.
        reflected = lower > 0
        low = np.where(reflected, -upper, lower)
        high = np.where(reflected, -lower, upper)
        a, b = log_ndtr(low), log_ndtr(high)
        return b + np.log(-np.expm1(a - b))

    numerator = interval((cells / 100 - means) / scales, ((cells + 1) / 100 - means) / scales)
    denominator = interval(-means / scales, (1 - means) / scales)
    scores = (numerator - denominator).sum(axis=1)
    if not np.isfinite(scores).all() or (scores > 1e-8).any():
        raise ValueError('Invalid conditional Gaussian cell probability')
    return scores



def quantize(xy):
    """Equal-width, half-open screen cells; reject rather than clip invalid gaze."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2 or not np.isfinite(xy).all() or np.any((xy < 0) | (xy >= 1)):
        raise ValueError('Coordinates must be finite and inside [0, 1).')
    return np.floor(xy * 100).astype(int)


def digit_log_mass(logits, cell):
    """Four teacher-forced digit positions, each containing all ten legal logits.

    Positions are x tens, x units, y tens, y units. Prefixes must include the
    preceding true digits and fixed punctuation. Missing top-k logits are forbidden.
    """
    logits = np.asarray(logits, dtype=float)
    cell = np.asarray(cell)
    if logits.shape != (4, 10) or not np.isfinite(logits).all():
        raise ValueError('Exact finite logits for all four sets of ten digits are required.')
    if cell.shape != (2,) or np.any(cell != cell.astype(int)) or np.any((cell < 0) | (cell > 99)):
        raise ValueError('Invalid target cell.')
    x, y = cell.astype(int)
    digits = [x // 10, x % 10, y // 10, y % 10]
    return float(np.sum(logits[np.arange(4), digits] - logsumexp(logits, axis=1)))


def gmm_grid(weights, means, scales):
    """Integrate diagonal Gaussian components, then condition mixture on screen.

    Returns probability mass indexed [y, x], matching the existing diagonal GMM.
    """
    weights, means, scales = map(lambda a: np.asarray(a, dtype=float), (weights, means, scales))
    if means.shape != (len(weights), 2) or scales.shape != means.shape:
        raise ValueError('Invalid mixture shapes.')
    if not all(np.isfinite(a).all() for a in (weights, means, scales)) or np.any(scales <= 0) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid mixture parameters.')
    edges = np.linspace(0, 1, 101)
    z = (edges[None, :, None] - means[:, None, :]) / scales[:, None, :]
    # Survival form avoids subtracting two values rounded to one in the upper tail.
    cdf = np.where(z[:, :-1] >= 0, ndtr(-z[:, :-1]) - ndtr(-z[:, 1:]), ndtr(z[:, 1:]) - ndtr(z[:, :-1]))
    grid = np.einsum('k,kx,ky->yx', weights, cdf[:, :, 0], cdf[:, :, 1])
    if not np.isfinite(grid).all() or grid.sum() <= 0:
        raise ValueError('Screen mass underflow; use higher precision or revise parameters.')
    return grid / grid.sum()


def grid_log_mass(grid, cell):
    grid = np.asarray(grid, dtype=float)
    if grid.shape != (100, 100) or not np.isfinite(grid).all() or np.any(grid < 0) or not np.isclose(grid.sum(), 1):
        raise ValueError('Expected normalized 100 by 100 cell probability mass.')
    cell = np.asarray(cell)
    if cell.shape != (2,) or np.any(cell != cell.astype(int)) or np.any((cell < 0) | (cell > 99)):
        raise ValueError('Invalid target cell.')
    x, y = cell.astype(int)
    return float(np.log(grid[y, x])) if grid[y, x] else -np.inf


def frame_slots(pts, cutoff):
    result = []
    for requested in np.linspace(cutoff - 1, cutoff, 4):
        i = int(np.searchsorted(pts, requested, side='right') - 1)
        # The left boundary lies between frames in almost every real example.
        # Choose the first in-window frame there, rather than deleting this slot.
        if requested >= 0 and (i < 0 or pts[i] < cutoff - 1):
            i = int(np.searchsorted(pts, cutoff - 1, side='left'))
        result.append(None if requested < 0 or i < 0 or i >= len(pts) or pts[i] > cutoff or pts[i] < cutoff - 1 else {
            'index': i, 'pts_seconds': float(pts[i]), 'relative_seconds': float(pts[i] - cutoff)})
    return result
