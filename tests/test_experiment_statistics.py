import unittest
try:
    import numpy as np
    from src.experiment_statistics import seed_summary, paired_comparison
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install requirements/base.txt')
class ExperimentStatistics(unittest.TestCase):
    def test_seed_sd_is_not_image_sd(self):
        summary = seed_summary([.7, .8, .9])
        self.assertAlmostEqual(summary['mean'], .8)
        self.assertAlmostEqual(summary['std_across_seeds'], .1)
        self.assertIsNone(seed_summary([.8])['std_across_seeds'])

    def test_zero_difference_and_constant_paired_effect(self):
        values = np.full((3, 4), .7)
        groups = ['a', 'a', 'b', 'c']
        same = paired_comparison(values, values, groups, repetitions=200)
        self.assertEqual(same['hierarchical_group_bootstrap_ci95'], [0, 0])
        self.assertEqual(same['paired_seed_sign_flip_pvalue'], 1)
        effect = paired_comparison(values + .1, values, groups, repetitions=200)
        self.assertAlmostEqual(effect['mean_difference_left_minus_right'], .1)
        self.assertAlmostEqual(effect['paired_seed_sign_flip_pvalue'], .25)
        for bound in effect['hierarchical_group_bootstrap_ci95']:
            self.assertAlmostEqual(bound, .1)
        self.assertEqual(effect['group_count'], 3)

    def test_alignment_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison([[1, 0]], [[1]], ['a', 'b'])
        with self.assertRaises(ValueError):
            seed_summary([float('nan')])
