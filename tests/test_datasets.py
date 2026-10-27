import csv
from pathlib import Path
import tempfile
import unittest
try:
    import cv2
    import numpy as np
    import torch
    from src.datasets import PolypDataset, MEAN, STD, collate_original
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements/base.txt')
class DatasetFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data/raw').mkdir(parents=True)
        (self.root / 'data/splits').mkdir()
        self.csv = self.root / 'data/splits/train.csv'
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[4:20, 7:30] = 255
        image = np.repeat(mask[:, :, None], 3, axis=2)
        cv2.imwrite(str(self.root / 'data/raw/image.png'), image)
        cv2.imwrite(str(self.root / 'data/raw/mask.png'), mask)
        with self.csv.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'mask'])
            writer.writerow([r'data\raw\image.png', r'data\raw\mask.png'])

    def test_windows_csv_paths_work_on_host(self):
        image, mask = PolypDataset(self.csv, 64, root=self.root)[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(set(mask.unique().tolist()), {0, 1})

    def test_paired_augmentation_is_epoch_addressable(self):
        left = PolypDataset(self.csv, 64, root=self.root, augment=True, seed=43)
        right = PolypDataset(self.csv, 64, root=self.root, augment=True, seed=43)
        variants = set()
        for epoch in range(12):
            left.set_epoch(epoch)
            right.set_epoch(epoch)
            image, mask = left[0]
            other_image, other_mask = right[0]
            self.assertTrue(torch.equal(image, other_image) and torch.equal(mask, other_mask))
            restored = image.numpy().transpose(1, 2, 0) * STD + MEAN
            np.testing.assert_array_equal(restored[:, :, 0] > .5, mask[0].numpy() > .5)
            variants.add(mask.numpy().tobytes())
        self.assertGreater(len(variants), 1)
        left.set_epoch(2)
        expected = left[0]
        left.set_epoch(99)
        left[0]
        left.set_epoch(2)
        self.assertTrue(torch.equal(left[0][0], expected[0]))

    def test_original_masks_keep_shape_and_binary_01(self):
        small = np.zeros((23, 37), dtype=np.uint8)
        small[3:8, 4:11] = 1
        cv2.imwrite(str(self.root / 'data/raw/image.png'), np.repeat(small[:, :, None] * 255, 3, axis=2))
        cv2.imwrite(str(self.root / 'data/raw/mask.png'), small)
        sample = PolypDataset(self.csv, 64, root=self.root, return_original=True)[0]
        self.assertEqual(sample[2].shape, (23, 37))
        self.assertTrue(sample[1].any())
        batch = collate_original([sample, sample])
        self.assertEqual(tuple(batch[0].shape), (2, 3, 64, 64))
        self.assertEqual(batch[2][0].shape, (23, 37))
        with self.assertRaises(ValueError):
            PolypDataset(self.csv, root=self.root, augment=True, return_original=True)

    def test_missing_data_and_shape_mismatch_are_explicit(self):
        cv2.imwrite(str(self.root / 'data/raw/mask.png'), np.zeros((4, 7), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            PolypDataset(self.csv, root=self.root)[0]
        (self.root / 'data/raw/mask.png').unlink()
        with self.assertRaises(FileNotFoundError):
            PolypDataset(self.csv, root=self.root)
