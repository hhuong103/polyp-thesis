from pathlib import Path
import tempfile
import unittest
from src.paths import dataset_path, file_lock


class PortablePaths(unittest.TestCase):
    def test_windows_and_posix_paths_resolve_identically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(dataset_path(r'data\raw\images\a.png', root), root / 'data/raw/images/a.png')
            self.assertEqual(dataset_path('data/raw/images/a.png', root), root / 'data/raw/images/a.png')

    def test_absolute_and_escaping_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in ('../outside', 'data/../../outside', '/etc/passwd', r'C:\data\a.png', r'\\host\share\a.png'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    dataset_path(value, Path(folder))

    def test_competing_writer_cannot_acquire_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.lock'
            with file_lock(path):
                with self.assertRaises(OSError):
                    with file_lock(path):
                        self.fail('A competing writer acquired the held lock')
            with file_lock(path):
                pass
