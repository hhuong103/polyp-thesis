import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import shutdown_after_training as job


class ShutdownSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.patches = [patch.object(job, 'RUN', self.run),
                        patch.object(job, 'STATUS', self.run / 'shutdown_status.json'),
                        patch.object(job, 'CANCEL', self.run / 'shutdown_cancel.json')]
        for item in self.patches:
            item.start()
        self.write('config.json', {'model': 'unetplusplus'})
        self.config_hash = job.digest(self.run / 'config.json')
        for name in ('best_weights.pth', 'last_checkpoint.pth', 'test_evaluation.csv'):
            (self.run / name).write_bytes(b'verified output')
        with (self.run / 'history.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'val_dice'])
            writer.writerows((i, .8) for i in range(1, 11))
        self.metrics = [{'id': name, 'count': 150, 'dice': .8, 'iou': .7}
                        for name in ('final-unet', 'unetplusplus')]
        self.summary = {'status': 'complete', 'epochs': 10, 'best_epoch': 2,
                        'best_weights_sha256': job.digest(self.run / 'best_weights.pth'),
                        'test_results': self.metrics}
        self.write('completion.json', self.summary)
        self.write('lifecycle.json', {'phase': 'complete', 'summary': self.summary})
        self.write('test_evaluation.json', {'models': [{**m, 'samples': [{}] * 150} for m in self.metrics]})

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding='utf-8')

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_complete_run_requests_shutdown_once(self):
        shutdown = Mock()
        with patch('builtins.print'):
            job.monitor(shutdown=shutdown, grace_seconds=0)
        shutdown.assert_called_once_with()

    def test_training_is_not_completion(self):
        self.write('lifecycle.json', {'phase': 'training', 'epochs': 10})
        self.assertFalse(job.completed(self.config_hash))

    def test_training_error_does_not_shutdown(self):
        self.write('lifecycle.json', {'phase': 'error', 'error': 'failed'})
        shutdown = Mock()
        with patch('builtins.print'), self.assertRaises(ValueError):
            job.monitor(shutdown=shutdown, grace_seconds=0)
        shutdown.assert_not_called()

    def test_changed_weights_do_not_shutdown(self):
        (self.run / 'best_weights.pth').write_bytes(b'changed')
        shutdown = Mock()
        with patch('builtins.print'), self.assertRaises(ValueError):
            job.monitor(shutdown=shutdown, grace_seconds=0)
        shutdown.assert_not_called()

    def test_incomplete_test_report_rejected(self):
        self.write('test_evaluation.json', {'models': [{**m, 'samples': []} for m in self.metrics]})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            job.completed(self.config_hash)

    def test_cancel_prevents_shutdown(self):
        def cancel_during_countdown(seconds):
            state = job.read_json(job.STATUS)
            job.write_json(job.CANCEL, {'request_id': state['request_id']})
        shutdown = Mock()
        with patch('builtins.print'), patch.object(job.time, 'sleep', side_effect=cancel_during_countdown):
            job.monitor(shutdown=shutdown, grace_seconds=120)
        shutdown.assert_not_called()
        self.assertEqual(job.read_json(job.STATUS)['phase'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
