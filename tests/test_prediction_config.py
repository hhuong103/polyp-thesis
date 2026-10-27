import json
from pathlib import Path
import tempfile
import unittest
try:
    from src.predict import prediction_config
    from src.paths import sha256_file
except ImportError:
    prediction_config = None


@unittest.skipIf(prediction_config is None, 'Install requirements/base.txt')
class PredictionConfiguration(unittest.TestCase):
    def test_architecture_and_threshold_follow_bound_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'best_weights.pth'
            weights.write_bytes(b'test artifact')
            (root / 'config.json').write_text(json.dumps({'model': 'unetplusplus', 'image_size': 64, 'mask_threshold': .5}))
            self.assertEqual(prediction_config(weights)['model'], 'unetplusplus')
            decision = {'threshold': .65, 'best_weights_sha256': sha256_file(weights)}
            (root / 'decision.json').write_text(json.dumps(decision))
            self.assertEqual(prediction_config(weights)['threshold'], .65)
            weights.write_bytes(b'changed artifact')
            with self.assertRaisesRegex(ValueError, 'calibration'):
                prediction_config(weights)

    def test_missing_config_does_not_guess_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prediction_config(Path(directory) / 'custom.pth')
