import unittest
try:
    import numpy as np
    from src.metrics import scores, restore_mask
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install requirements/base.txt')
class BinaryMetrics(unittest.TestCase):
    def test_empty_and_full_conventions(self):
        empty = np.zeros((2, 3), dtype=bool)
        full = ~empty
        for pred, gold, expected in [(empty, empty, 1), (full, full, 1), (empty, full, 0), (full, empty, 0)]:
            self.assertEqual(scores(pred, gold)['dice'], expected)
            self.assertEqual(scores(pred, gold)['iou'], expected)

    def test_original_resolution_and_threshold_before_resize(self):
        probability = np.array([[.1, .6], [.9, .2]])
        expected = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(restore_mask(probability, (4, 4), .5), expected)
        self.assertEqual(scores(expected, expected)['dice'], 1)

    def test_partial_overlap_and_shape_validation(self):
        value = scores(np.array([[1, 1, 0]]), np.array([[0, 1, 1]]))
        self.assertAlmostEqual(value['dice'], .5)
        self.assertAlmostEqual(value['iou'], 1 / 3)
        self.assertEqual(value['false_positive_pixels'], 1)
        self.assertEqual(value['false_negative_pixels'], 1)
        with self.assertRaises(ValueError):
            scores(np.zeros((2, 3)), np.zeros((1, 3)))
