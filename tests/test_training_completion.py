"""Guard against data leakage and premature completion in the training supervisor."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import complete_training as job


class CompletionSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "results" / "run"
        self.run.mkdir(parents=True)
        self.config = {"model": "unetplusplus", "encoder": "resnet34", "encoder_init": "imagenet",
                       "image_size": 256, "batch_size": 4, "learning_rate": 1e-4, "patience": 8,
                       "seed": 42, "split_sha256": {}}
        (self.root / "data/splits").mkdir(parents=True)
        (self.root / "data/raw").mkdir(parents=True)
        number = 0
        for split, count in (("train", 700), ("val", 150), ("test", 150)):
            path = self.root / "data/splits" / (split + ".csv")
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image", "mask"])
                for _ in range(count):
                    name = str(number)
                    (self.root / "data/raw" / name).write_bytes(name.encode())
                    writer.writerow(["data/raw/" + name, "data/raw/" + name])
                    number += 1
            self.config["split_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_config()
        self.root_patch = patch.object(job, "ROOT", self.root)
        self.run_patch = patch.object(job, "RUN", self.run)
        self.root_patch.start()
        self.run_patch.start()

    def save_config(self):
        (self.run / "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def tearDown(self):
        self.root_patch.stop()
        self.run_patch.stop()
        self.temp.cleanup()

    def test_audit_tracks_actual_data_changes(self):
        before = job.audit()
        self.assertEqual(before["counts"], {"train": 700, "val": 150, "test": 150})
        (self.root / "data/raw/0").write_bytes(b"changed")
        self.assertNotEqual(before["dataset_sha256"], job.audit()["dataset_sha256"])

    def test_duplicate_image_rejected(self):
        (self.root / "data/raw/999").write_bytes((self.root / "data/raw/0").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            job.audit()

    def test_changed_split_rejected(self):
        with (self.root / "data/splits/test.csv").open("a") as stream:
            stream.write("data/raw/1,data/raw/1\n")
        with self.assertRaisesRegex(ValueError, "Split changed"):
            job.audit()

    def test_changed_hyperparameters_rejected(self):
        self.config["batch_size"] = 2
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            job.audit()

    def test_epoch_one_is_not_complete(self):
        self.assertFalse(job.terminal({"epoch": 1, "stale": 0}))
        self.assertTrue(job.terminal({"epoch": 50, "stale": 0}))
        self.assertTrue(job.terminal({"epoch": 12, "stale": 8}))


if __name__ == "__main__":
    unittest.main()
