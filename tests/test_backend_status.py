"""Backend unit tests use synthetic files, never a real dataset or HTTP server."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from backend import comparison
except ImportError:
    comparison = None


@unittest.skipIf(comparison is None, 'Install backend requirements')
class BackendStatusTests(unittest.TestCase):
    def test_study_only_exposes_summary_after_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'results/experiments/grouped_v2_cpu'
            folder.mkdir(parents=True)
            (folder / 'summary.json').write_text(json.dumps({'result': 'verified'}))
            (folder / 'status.json').write_text(json.dumps({'phase': 'training'}))
            with patch.object(comparison, 'ROOT', root):
                self.assertIsNone(comparison.study()['summary'])
                (folder / 'status.json').write_text(json.dumps({'phase': 'complete'}))
                self.assertEqual(comparison.study()['summary'], {'result': 'verified'})

    def test_new_model_progress(self):
        from backend import comparison
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {'id': 'new', 'weights': 'best.pth', 'history': 'history.csv',
                     'training': {'max_epochs': 3, 'patience': 2}}
            with patch.object(comparison, 'ROOT', root), patch.object(comparison, 'REGISTRY', [entry]):
                state = comparison.model_status(entry)
                self.assertFalse(state['available'])
                self.assertEqual(state['progress']['state'], 'awaiting_checkpoint')
                self.assertEqual(comparison.evaluation_entries(), [])
                (root / 'best.pth').touch()
                (root / 'history.csv').write_text('epoch,val_dice\n1,0.5\n', encoding='utf-8')
                state = comparison.model_status(entry)
                self.assertEqual(state['progress']['epochs'], 1)
                self.assertEqual(state['progress']['state'], 'checkpoint_available')
                self.assertFalse(state['ready_for_evaluation'])
                (root / 'history.csv').write_text('epoch,val_dice\n1,0.5\n2,0.6\n3,0.7\n', encoding='utf-8')
                (root / 'last_checkpoint.pth').touch()
                state = comparison.model_status(entry)
                self.assertEqual(state['progress']['state'], 'complete')
                self.assertTrue(state['ready_for_evaluation'])

    def test_missing_checkpoint_is_503(self):
        from backend import comparison
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, 'ROOT', Path(directory)):
            with self.assertRaises(HTTPException) as caught:
                comparison.load_network(comparison.config('unetplusplus'))
        self.assertEqual(caught.exception.status_code, 503)
