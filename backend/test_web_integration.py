"""Run against a local server: python -m backend.test_web_integration."""
import base64
import csv
import io
import json
import os
import unittest
import urllib.error
import urllib.request
import uuid
import tempfile
from unittest.mock import patch
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("POLYP_TEST_URL", "http://127.0.0.1:8082")


def request(path, fields=None, files=None):
    data, headers = None, {}
    if fields is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in fields.items():
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode())
        for key, raw in (files or {}).items():
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n').encode() + raw + b'\r\n')
        data = b''.join(parts) + f'--{boundary}--\r\n'.encode()
        headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@unittest.skipIf(np is None, 'Install backend requirements for HTTP tests')
class WebsiteIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('POLYP_RUN_INTEGRATION') != '1':
            raise unittest.SkipTest('Set POLYP_RUN_INTEGRATION=1 to run HTTP integration tests')
        if not (ROOT / 'data/splits/test.csv').is_file():
            raise unittest.SkipTest('Real dataset is unavailable')
        try:
            _, health = request('/health')
        except (OSError, urllib.error.URLError) as exc:
            raise unittest.SkipTest(f'HTTP server unavailable at {BASE}: {exc}')
        if json.loads(health).get('status') != 'ok':
            raise unittest.SkipTest('HTTP server requires both loaded checkpoints')
        with (ROOT / 'data/splits/test.csv').open(newline='') as stream:
            row = next(csv.DictReader(stream))
        image_path = ROOT / row['image'].replace('\\', '/')
        truth_path = ROOT / row['mask'].replace('\\', '/')
        if not image_path.is_file() or not truth_path.is_file():
            raise unittest.SkipTest('Real image/mask pair is unavailable')
        cls.image = image_path.read_bytes()
        cls.truth = truth_path.read_bytes()
        cls.fields = {'model_a': 'final-unet', 'model_b': 'unetplusplus'}

    def test_pages(self):
        for path in ('/', '/assets/style.css', '/assets/app.js', '/api/models', '/api/history'):
            self.assertEqual(request(path)[0], 200, path)

    def test_comparison_metrics(self):
        status, raw = request('/api/compare', self.fields, {'file': self.image, 'ground_truth': self.truth})
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        self.assertEqual(len(data['results']), 2)
        self.assertEqual([r['id'] for r in data['results']], ['final-unet', 'unetplusplus'])
        gold = np.array(Image.open(io.BytesIO(self.truth)).convert('L')) > 127
        for result in data['results']:
            from backend.comparison import config, load_network
            network = load_network(config(result['id']))
            self.assertEqual(result['parameter_count'], sum(p.numel() for p in network.parameters()))
            self.assertEqual(result['encoder'], 'resnet34')
            self.assertIn(result['experiment'], ('E3', 'E4'))
            mask = np.array(Image.open(io.BytesIO(base64.b64decode(result['mask']))))
            self.assertEqual(mask.shape, gold.shape)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
            pred = mask > 0
            intersection = int((pred & gold).sum())
            denominator = int(pred.sum() + gold.sum())
            union = int((pred | gold).sum())
            self.assertAlmostEqual(result['metrics']['dice'], 2 * intersection / denominator if denominator else 1)
            self.assertAlmostEqual(result['metrics']['iou'], intersection / union if union else 1)

    def test_evaluation_mode(self):
        status, raw = request('/api/evaluate-image', self.fields,
                              {'file': self.image, 'ground_truth': self.truth})
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        self.assertIsNotNone(data['ground_truth'])
        self.assertEqual(len(data['results']), 2)
        for result in data['results']:
            self.assertIsNotNone(result['metrics'])
            self.assertGreater(result['parameter_count'], 0)
            self.assertEqual(result['foreground_pixels'], int((np.array(Image.open(
                io.BytesIO(base64.b64decode(result['mask'])))) > 0).sum()))
        status, _ = request('/api/evaluate-image', self.fields, {'file': self.image})
        self.assertEqual(status, 422, 'Evaluation Mode must require Ground Truth')
        small = io.BytesIO()
        Image.new('L', (2, 2)).save(small, format='PNG')
        status, raw = request('/api/evaluate-image', self.fields,
                            {'file': self.image, 'ground_truth': small.getvalue()})
        self.assertEqual(status, 400, 'Ground Truth must keep its original resolution')
        self.assertIn('2 × 2', json.loads(raw)['detail'])
        width, height = Image.open(io.BytesIO(self.image)).size
        self.assertIn(f'{width} × {height}', json.loads(raw)['detail'])

    def test_single_models_match_comparison(self):
        status, raw = request('/api/compare', self.fields, {'file': self.image})
        self.assertEqual(status, 200)
        compared = {r['id']: r for r in json.loads(raw)['results']}
        for selected, expected_id, checkpoint in (
            ('unet', 'final-unet', 'models/final_polyp_model.pth'),
            ('unetplusplus', 'unetplusplus', 'results/unetplusplus_resnet34_seed42/best_weights.pth'),
        ):
            with self.subTest(model=selected):
                status, raw = request('/api/predict', {'model': selected}, {'file': self.image})
                self.assertEqual(status, 200, raw[:300])
                data = json.loads(raw)
                self.assertEqual(len(data['results']), 1)
                result = data['results'][0]
                self.assertEqual(data['model_id'], expected_id)
                self.assertEqual(result['checkpoint'], checkpoint)
                self.assertEqual(result['mask'], compared[expected_id]['mask'])
                self.assertEqual(result['overlay'], compared[expected_id]['overlay'])
                self.assertEqual(data['mask_base64'], result['mask'])
                self.assertGreater(data['inference_time_ms'], 0)
                mask = np.array(Image.open(io.BytesIO(base64.b64decode(result['mask'])))) > 0
                self.assertEqual(result['foreground_pixels'], int(mask.sum()))
                self.assertEqual(result['empty_mask'], not bool(mask.any()))
                original = np.array(Image.open(io.BytesIO(self.image)).convert('RGB'))
                overlay = np.array(Image.open(io.BytesIO(base64.b64decode(result['overlay']))))
                self.assertEqual(mask.shape, original.shape[:2])
                self.assertEqual(overlay.shape, original.shape)
                self.assertTrue(np.array_equal(overlay[~mask], original[~mask]))

    def test_resize_ground_truth_is_explicit_and_binary(self):
        # A 0/1 PNG exercises binary decoding as well as asymmetric nearest resizing.
        source = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(source).save(stream, format='PNG')
        width, height = Image.open(io.BytesIO(self.image)).size
        expected = source[(np.arange(height) * 2 // height)[:, None],
                          (np.arange(width) * 3 // width)[None, :]] > 0
        for endpoint, selection in (
            ('/api/evaluate-image', self.fields),
            ('/api/compare', self.fields),
            ('/api/predict', {'model': 'unet'}),
            ('/predict', {'model': 'unetplusplus'}),
        ):
            with self.subTest(endpoint=endpoint):
                files = {'file': self.image, 'ground_truth': stream.getvalue()}
                status, _ = request(endpoint, selection, files)
                self.assertEqual(status, 400, 'Resizing requires an explicit choice')
                status, raw = request(endpoint, {**selection, 'resize_ground_truth': 'true'}, files)
                self.assertEqual(status, 200, raw[:300])
                data = json.loads(raw)
                self.assertEqual(data['ground_truth_alignment'], {
                    'original_width': 3, 'original_height': 2, 'target_width': width,
                    'target_height': height, 'resized': True, 'method': 'nearest'})
                gold = np.array(Image.open(io.BytesIO(base64.b64decode(data['ground_truth']))))
                self.assertTrue(set(np.unique(gold)).issubset({0, 255}))
                np.testing.assert_array_equal(gold > 0, expected)
                for result in data['results']:
                    pred = np.array(Image.open(io.BytesIO(base64.b64decode(result['mask'])))) > 0
                    intersection = int((pred & expected).sum())
                    self.assertAlmostEqual(result['metrics']['dice'],
                                           2 * intersection / int(pred.sum() + expected.sum()))
                    self.assertAlmostEqual(result['metrics']['iou'],
                                           intersection / int((pred | expected).sum()))

    def test_single_with_ground_truth(self):
        status, raw = request('/api/predict', {'model': 'unetplusplus'},
                              {'file': self.image, 'ground_truth': self.truth})
        self.assertEqual(status, 200, raw[:300])
        result = json.loads(raw)['results'][0]
        self.assertIsNotNone(result['metrics'])
        self.assertIn('errors', result)

    def test_legacy_predict_url(self):
        for fields, expected in (({}, 'final-unet'), ({'model': 'unetplusplus'}, 'unetplusplus')):
            status, raw = request('/predict', fields, {'file': self.image})
            self.assertEqual(status, 200)
            data = json.loads(raw)
            self.assertEqual(data['model_id'], expected)
            self.assertIn('overlay_base64', data)
            self.assertIn('inference_time_ms', data)

    def test_predict_accepts_png(self):
        stream = io.BytesIO()
        Image.open(io.BytesIO(self.image)).save(stream, format='PNG')
        status, raw = request('/api/predict', {'model': 'unetplusplus'}, {'file': stream.getvalue()})
        self.assertEqual(status, 200, raw[:300])

    def test_predict_rejects_invalid_model_and_upload(self):
        for fields, content, expected in (
            ({'model': '../weights.pth'}, self.image, 400),
            ({'model': 'unet'}, b'', 400),
            ({'model': 'unetplusplus'}, b'not-an-image', 400),
            ({'model': 'unet'}, b'x' * (10 * 1024 * 1024 + 1), 413),
        ):
            self.assertEqual(request('/api/predict', fields, {'file': content})[0], expected)

    def test_both_models_loaded_in_health(self):
        status, raw = request('/health')
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual({m['id'] for m in data['models']}, {'final-unet', 'unetplusplus'})
        self.assertTrue(all(m['loaded'] for m in data['models']))

    def test_no_ground_truth(self):
        status, raw = request('/api/compare', self.fields, {'file': self.image})
        self.assertEqual(status, 200)
        self.assertTrue(all(r['metrics'] is None for r in json.loads(raw)['results']))

    def test_invalid_inputs(self):
        small = io.BytesIO()
        Image.new('L', (2, 2)).save(small, format='PNG')
        for fields, files in [
            (self.fields, {'file': b'bad'}),
            (self.fields, {'file': self.image, 'ground_truth': small.getvalue()}),
            ({'model_a': 'final-unet', 'model_b': 'final-unet'}, {'file': self.image}),
            ({'model_a': 'unknown', 'model_b': 'final-unet'}, {'file': self.image}),
            ({'model_a': 'unet', 'model_b': 'final-unet'}, {'file': self.image}),
        ]:
            self.assertEqual(request('/api/compare', fields, files)[0], 400)


@unittest.skipIf(np is None, 'Install backend requirements for checkpoint tests')
class RealCheckpointIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('POLYP_RUN_INTEGRATION') != '1':
            raise unittest.SkipTest('Set POLYP_RUN_INTEGRATION=1 for real checkpoint tests')
        from backend import comparison
        if not all((ROOT / comparison.config(key)['weights']).is_file() for key in ('unet', 'unetplusplus')):
            raise unittest.SkipTest('Both real checkpoints are required')

    def test_loader_uses_distinct_architectures_and_shared_cache(self):
        from backend import comparison
        for model_id, class_name in (('unet', 'Unet'), ('unetplusplus', 'UnetPlusPlus')):
            network = comparison.load_network(comparison.config(model_id))
            self.assertEqual(type(network).__name__, class_name)
            self.assertIs(network, comparison.load_network(comparison.config(model_id)))



if __name__ == '__main__':
    unittest.main()
