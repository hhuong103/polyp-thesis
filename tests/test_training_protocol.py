"""Exercise checkpoint continuation against uninterrupted optimization on synthetic data."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
    from src import train
    from src.split_dataset import create_split
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements/train.txt')
class TrainingProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        images, masks = self.root / 'data/raw/kvasir_seg/images', self.root / 'data/raw/kvasir_seg/masks'
        images.mkdir(parents=True)
        masks.mkdir()
        rng = np.random.default_rng(412)
        for i in range(16):
            image = rng.integers(0, 256, (35 + i, 49, 3), dtype=np.uint8)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[3:17, 4:22] = 255
            cv2.imwrite(str(images / f'{i}.png'), image)
            cv2.imwrite(str(masks / f'{i}.png'), mask)
        self.split = self.root / 'data/splits/fixture'
        create_split(self.root, self.split, expected_count=16, distance=0)
        class TinyNetwork(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                                  torch.nn.Conv2d(4, 1, 1))
            def forward(self, images):
                return self.layers(images)
        self.patches = [patch.object(train, 'ROOT', self.root), patch.object(train.smp, 'Unet', TinyNetwork)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.args = ['--epochs', '2', '--batch-size', '2', '--image-size', '32', '--encoder-init', 'none',
                     '--split-dir', str(self.split), '--threads', '1', '--device', 'cpu', '--scheduler-patience', '0']

    def invoke(self, name, *extra):
        with redirect_stdout(io.StringIO()):
            return train.main(self.args + ['--run-name', name] + list(extra))

    def test_interrupted_resume_matches_uninterrupted_weights_metrics_and_scheduler(self):
        full = self.invoke('full')
        with patch.object(train, 'copy_backup', side_effect=InterruptedError('simulated interruption')):
            with self.assertRaises(InterruptedError):
                self.invoke('resumed')
        resumed = self.invoke('resumed', '--resume')
        a = torch.load(self.root / 'results/full/last_checkpoint.pth', weights_only=False)
        b = torch.load(self.root / 'results/resumed/last_checkpoint.pth', weights_only=False)
        for key in a['model_state']:
            self.assertTrue(torch.equal(a['model_state'][key], b['model_state'][key]), key)
        self.assertEqual(a['scheduler_state'], b['scheduler_state'])
        self.assertTrue(torch.equal(a['loader_rng'], b['loader_rng']))
        for left, right in zip(a['history'], b['history']):
            self.assertEqual({k: v for k, v in left.items() if k != 'seconds'},
                             {k: v for k, v in right.items() if k != 'seconds'})
        self.assertEqual(full['best_val_dice'], resumed['best_val_dice'])
        self.assertFalse(resumed['test_used_for_selection'])
        self.assertEqual(self.invoke('resumed', '--resume'), resumed)
        decision = json.loads((self.root / 'results/resumed/decision.json').read_text())
        self.assertEqual(decision['selected_on'], 'validation_only')
        self.assertEqual(decision['resolution'], 'original')

    def test_resume_rejects_changed_augmentation_policy(self):
        with patch.object(train, 'copy_backup', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                self.invoke('changed')
        with self.assertRaisesRegex(ValueError, 'config'):
            self.invoke('changed', '--resume', '--no-augmentation')

    def test_threshold_tie_prefers_predeclared_half(self):
        probabilities = [np.ones((2, 2))]
        originals = [np.ones((4, 5), dtype=bool)]
        selected, _ = train.select_threshold(probabilities, originals, [.3, .5, .7])
        self.assertEqual(selected['threshold'], .5)
