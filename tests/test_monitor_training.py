"""Monitoring reads existing artifacts and never starts or signals training."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src import monitor_training as monitor


class EpochMonitor(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'name': 'fixture', 'models': ['unet', 'unetplusplus'], 'seeds': [42], 'epochs': 50}
        self.study = self.root / 'results/experiments/fixture'
        self.study.mkdir(parents=True)
        self.first = self.root / 'results/unet_fixture_seed42'
        self.first.mkdir()
        self.write_state('unet_fixture_seed42')
        self.write_history(self.first, 3)

    def write_state(self, name):
        (self.study / 'status.json').write_text(json.dumps({'phase': 'training', 'run': name, 'runner_pid': os.getpid()}))

    def write_history(self, run, count):
        with (run / 'history.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'learning_rate', 'seconds'])
            for epoch in range(1, count + 1):
                writer.writerow([epoch, 1 / epoch, .8 / epoch, .7 + epoch * .01, .6 + epoch * .01, 1e-4, 300])

    def view(self, alive=True):
        with patch.object(monitor, 'process_alive', return_value=alive):
            return monitor.snapshot(self.config, self.root)

    def test_displays_saved_metrics_and_recent_epoch_duration(self):
        view = self.view()
        self.assertEqual(view['runs'][0]['saved_epoch'], 3)
        output = monitor.render(view, tail=2)
        self.assertIn('runner: ACTIVE', output)
        self.assertIn('Val Dice', output)
        self.assertIn('0.7300', output)
        self.assertIn('Recent average per epoch: 05:00', output)
        self.assertIn('Waiting for completed epoch 4', output)
        self.assertNotIn('\n    1  ', output)

    def test_switches_run_and_counts_actual_completions(self):
        (self.first / 'run_complete.json').write_text(json.dumps({'status': 'complete', 'stop_reason': 'early_stopping'}))
        second = self.root / 'results/unetplusplus_fixture_seed42'
        second.mkdir()
        self.write_history(second, 1)
        self.write_state(second.name)
        view = self.view()
        self.assertEqual(view['active_run'], second.name)
        self.assertEqual(view['completed_runs'], 1)
        output = monitor.render(view)
        self.assertIn('Runs complete: 1/2', output)
        self.assertIn('Current run: unetplusplus_fixture_seed42', output)
        self.assertIn('Waiting for completed epoch 2', output)

    def test_stale_status_does_not_claim_training_is_active(self):
        output = monitor.render(self.view(alive=False))
        self.assertIn('runner: STOPPED', output)
        self.assertIn('stored phase is not a live heartbeat', output)
        self.assertNotIn('Waiting for completed epoch', output)

    def test_reading_progress_does_not_modify_any_artifact(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.view()
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_once_prints_without_waiting_and_ctrl_c_only_stops_viewer(self):
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config))
        view = self.view()
        output = io.StringIO()
        with patch.object(monitor, 'snapshot', return_value=view), patch.object(monitor.time, 'sleep') as sleep, redirect_stdout(output):
            monitor.main(['--config', str(config_path), '--once'])
        sleep.assert_not_called()
        self.assertIn('LIVE EPOCH MONITOR', output.getvalue())
        output = io.StringIO()
        with patch.object(monitor, 'snapshot', return_value=view), patch.object(monitor.time, 'sleep', side_effect=KeyboardInterrupt), redirect_stdout(output):
            monitor.main(['--config', str(config_path)])
        self.assertIn('Background training was not interrupted', output.getvalue())

    def test_pid_check_is_read_only_and_detects_current_process(self):
        self.assertTrue(monitor.process_alive(os.getpid()))
        self.assertIsNone(monitor.process_alive(None))
        self.assertFalse(monitor.process_alive(2147483647))

    def test_foreign_os_pid_is_not_misreported_as_stopped_or_active(self):
        foreign = 'Linux-test' if os.name == 'nt' else 'Windows-test'
        (self.first / 'environment.json').write_text(json.dumps({'platform': foreign}))
        with patch.object(monitor, 'process_alive') as probe:
            view = monitor.snapshot(self.config, self.root)
        probe.assert_not_called()
        self.assertIsNone(view['runner_alive'])
        self.assertIn('runner: UNKNOWN', monitor.render(view))
        self.assertNotIn('runner: STOPPED', monitor.render(view))
