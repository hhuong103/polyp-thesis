"""Seed variation and paired, group-aware uncertainty; no training/test tuning."""
import itertools

import numpy as np


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite seed-level scores')
    return {'mean': float(values.mean()), 'std_across_seeds': float(values.std(ddof=1)) if len(values) > 1 else None,
            'seed_count': len(values), 'values': values.tolist()}


def paired_comparison(left, right, groups, repetitions=5000, seed=2026):
    """Arrays: matched seeds x matched images. Resample seed pairs and image groups."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(groups)
            or not np.isfinite(left).all() or not np.isfinite(right).all() or min(left.shape) == 0):
        raise ValueError('Comparisons require finite, aligned seed/image arrays')
    if repetitions < 100:
        raise ValueError('Use at least 100 bootstrap repetitions')
    diff = left - right
    unique = sorted(set(groups))
    indices = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    group_sums = np.stack([diff[:, index].sum(axis=1) for index in indices], axis=1)
    group_sizes = np.array([len(index) for index in indices])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        seeds = rng.integers(diff.shape[0], size=diff.shape[0])
        selected = rng.integers(len(indices), size=len(indices))
        draws.append(float(group_sums[seeds][:, selected].sum() /
                           (len(seeds) * group_sizes[selected].sum())))
    per_seed = diff.mean(axis=1)
    observed = abs(float(per_seed.mean()))
    if len(per_seed) <= 16:
        null = [abs(float((per_seed * signs).mean())) for signs in itertools.product((-1, 1), repeat=len(per_seed))]
        p = float(np.mean(np.array(null) >= observed - 1e-12))
        method = 'exact two-sided sign-flip on paired seed differences'
    else:
        null = np.abs((rng.choice((-1, 1), size=(repetitions, len(per_seed))) * per_seed).mean(axis=1))
        p = float((np.count_nonzero(null >= observed - 1e-12) + 1) / (repetitions + 1))
        method = 'Monte Carlo two-sided sign-flip on paired seed differences'
    return {'mean_difference_left_minus_right': float(diff.mean()),
            'paired_seed_differences': per_seed.tolist(),
            'hierarchical_group_bootstrap_ci95': np.quantile(draws, [.025, .975]).tolist(),
            'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
            'paired_seed_sign_flip_pvalue': p, 'pvalue_method': method,
            'group_count': len(unique), 'seed_count': diff.shape[0],
            'limitations': 'Few seeds give low statistical power. Groups are perceptual/metadata groups, not proof of patient independence.'}
