"""Windows regression: a live legacy runner locks the byte before a launcher probes it."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import launch_experiments as launcher
from src.paths import ROOT


class LauncherLockTests(unittest.TestCase):
    def start_legacy_lock_holder(self, path):
        source = ('import sys; from src.paths import file_lock; '
                  'lock=file_lock(sys.argv[1]); lock.__enter__(); '
                  'print("locked",flush=True); sys.stdin.read(); lock.__exit__(None,None,None)')
        child = subprocess.Popen([sys.executable, '-u', '-c', source, str(path)], cwd=ROOT,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        def cleanup():
            if child.poll() is None:
                child.communicate(input='', timeout=10)
        self.addCleanup(cleanup)
        self.assertEqual(child.stdout.readline().strip(), 'locked')
        return child

    def test_competing_process_returns_busy_then_released_lock_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runner.lock'
            child = self.start_legacy_lock_holder(path)
            try:
                with self.assertRaises(launcher.StudyAlreadyRunning):
                    with launcher.launcher_lock(path):
                        self.fail('Launched despite the existing runner')
            finally:
                child.communicate(input='', timeout=10)
            with launcher.launcher_lock(path):
                self.assertEqual(path.stat().st_size, 1)

    def test_existing_runner_is_successful_noop_without_spawn_or_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'results/experiments/fixture'
            folder.mkdir(parents=True)
            config = root / 'config.json'
            config.write_text(json.dumps({'name': 'fixture'}))
            checkpoint = folder / 'last_checkpoint.pth'
            checkpoint.write_bytes(b'protected checkpoint')
            child = self.start_legacy_lock_holder(folder / 'runner.lock')
            try:
                with patch.object(launcher, 'ROOT', root), patch.object(launcher.subprocess, 'Popen') as spawn, redirect_stdout(io.StringIO()):
                    result = launcher.main(['--config', str(config)])
                self.assertEqual(result['status'], 'already_running')
                spawn.assert_not_called()
                self.assertEqual(checkpoint.read_bytes(), b'protected checkpoint')
            finally:
                child.communicate(input='', timeout=10)

    def test_free_stale_file_launches_once_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'results/experiments/fixture'
            folder.mkdir(parents=True)
            (folder / 'runner.lock').write_bytes(b'0')
            config = root / 'config.json'
            config.write_text(json.dumps({'name': 'fixture'}))
            with patch.object(launcher, 'ROOT', root), patch.object(launcher.subprocess, 'Popen') as spawn, redirect_stdout(io.StringIO()):
                spawn.return_value.pid = 123
                result = launcher.main(['--config', str(config)])
            self.assertEqual(result['status'], 'launched')
            spawn.assert_called_once()
            with launcher.launcher_lock(folder / 'runner.lock'):
                pass

    def test_real_filesystem_permission_error_is_not_reported_as_running(self):
        with patch.object(Path, 'open', side_effect=PermissionError('filesystem permission denied')):
            with self.assertRaises(PermissionError):
                with launcher.launcher_lock(Path('forbidden.lock')):
                    self.fail('Acquired forbidden file')
