"""Complete study lifecycle on synthetic data without subprocesses or expensive networks."""
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
    from src import run_experiments as runner, train
    from src.split_dataset import create_split
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements/train.txt')
class ExperimentLifecycle(unittest.TestCase):
    def test_all_runs_frozen_before_test_and_safe_idempotent_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images, masks = root / 'data/raw/kvasir_seg/images', root / 'data/raw/kvasir_seg/masks'
            images.mkdir(parents=True)
            masks.mkdir()
            rng = np.random.default_rng(52)
            for i in range(16):
                image = rng.integers(0, 256, (34, 39, 3), dtype=np.uint8)
                mask = np.zeros((34, 39), dtype=np.uint8)
                mask[4:12, 6:15] = 255
                cv2.imwrite(str(images / f'{i}.png'), image)
                cv2.imwrite(str(masks / f'{i}.png'), mask)
            split = root / 'data/splits/fixture'
            create_split(root, split, expected_count=16, distance=0)
            class TinyNetwork(torch.nn.Module):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.layer = torch.nn.Conv2d(3, 1, 1)
                def forward(self, images):
                    return self.layer(images)
            cfg = {'name': 'fixture_study', 'models': ['unet', 'unetplusplus'], 'seeds': [42, 43],
                   'split_dir': 'data/splits/fixture', 'epochs': 1, 'batch_size': 2, 'image_size': 32,
                   'learning_rate': 1e-4, 'patience': 3, 'scheduler_patience': 0, 'augmentation': True,
                   'deterministic': True, 'calibrate_threshold': True, 'encoder_init': 'none',
                   'device': 'cpu', 'threads': 1, 'bootstrap_repetitions': 100}
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(cfg))
            actual_evaluate = runner.evaluate_run
            actual_subprocess_run = runner.subprocess.run
            evaluated = []
            def evaluate_after_all_frozen(run, frozen):
                out = root / 'results/experiments/fixture_study'
                frozen_all = json.loads((out / 'frozen_runs.json').read_text())
                self.assertEqual(len(frozen_all), 4)
                self.assertEqual(len(list((root / 'results').glob('*/run_complete.json'))), 4)
                evaluated.append(run.name)
                return actual_evaluate(run, frozen)
            def train_without_subprocess(command, **kwargs):
                if not isinstance(command, list) or len(command) < 4 or command[3] != 'src.train':
                    return actual_subprocess_run(command, **kwargs)
                with redirect_stdout(kwargs['stdout']):
                    train.main(command[4:])
            with patch.object(runner, 'ROOT', root), patch.object(train, 'ROOT', root), \
                    patch.object(train.smp, 'Unet', TinyNetwork), patch.object(train.smp, 'UnetPlusPlus', TinyNetwork), \
                    patch.object(runner.subprocess, 'run', side_effect=train_without_subprocess), \
                    patch.object(runner, 'evaluate_run', side_effect=evaluate_after_all_frozen), redirect_stdout(io.StringIO()):
                runner.main(['--config', str(config_path)])
                out = root / 'results/experiments/fixture_study'
                summary_before = json.loads((out / 'summary.json').read_text())
                self.assertEqual(len(evaluated), 4)
                self.assertEqual(json.loads((out / 'status.json').read_text())['phase'], 'complete')
                self.assertEqual(summary_before['models']['fixed_0_5']['unet']['dice']['seed_count'], 2)
                self.assertEqual(len(json.loads((out / 'deployment.json').read_text())), 2)
                runner.main(['--config', str(config_path)])
                self.assertEqual(len(evaluated), 8)
                # Restart uses saved test reports and cannot change decisions or rerun optimization.
                summary_after = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary_before['comparisons'], summary_after['comparisons'])
                target = root / 'results/unet_fixture_study_seed42/decision.json'
                target.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'decision'):
                    runner.freeze_run(target.parent)
