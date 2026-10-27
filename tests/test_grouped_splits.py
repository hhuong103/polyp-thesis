import csv
from pathlib import Path
import tempfile
import unittest
try:
    import cv2
    import numpy as np
    from src.split_dataset import connected_groups, allocate_groups, create_split, validate_split
    from src.paths import read_csv
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install requirements/base.txt')
class GroupedSplits(unittest.TestCase):
    def test_transitive_near_duplicates_and_patient_groups_stay_together(self):
        names = ['a', 'b', 'c', 'd']
        fingerprints = {n: {'image_sha256': n, 'decoded_sha256': n, 'phash': h}
                        for n, h in zip(names, ['0', '3', 'f', 'fffffffffffffff'])}
        metadata = {n: {'patient_id': 'same' if n in ('c', 'd') else n} for n in names}
        groups, edges = connected_groups(names, fingerprints, metadata, distance=2)
        self.assertEqual(groups, [names])
        self.assertTrue(any('patient_id' in e['reasons'] for e in edges))

    def test_allocate_never_breaks_groups_and_is_reproducible(self):
        groups = [['a', 'b', 'c']] + [[str(i)] for i in range(27)]
        a = allocate_groups(groups, 42)
        self.assertEqual(a, allocate_groups(groups, 42))
        for group in groups:
            owners = {s for s, members in a.items() if any(n in members for n in group)}
            self.assertEqual(len(owners), 1)
        self.assertEqual(sum(map(len, a.values())), 30)

    def test_versioned_creation_hash_checks_and_content_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images, masks = root / 'data/raw/kvasir_seg/images', root / 'data/raw/kvasir_seg/masks'
            images.mkdir(parents=True)
            masks.mkdir()
            rng = np.random.default_rng(123)
            source = None
            for i in range(12):
                image = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
                if i == 0:
                    source = image
                if i == 1:
                    image = source
                cv2.imwrite(str(images / f'{i}.png'), image)
                cv2.imwrite(str(masks / f'{i}.png'), np.zeros((32, 40), dtype=np.uint8))
            out = root / 'data/splits/v1'
            manifest = create_split(root, out, expected_count=12)
            for filename in ('train.csv', 'val.csv', 'test.csv', 'groups.csv', 'manifest.json'):
                self.assertNotIn(b'\r\n', (out / filename).read_bytes())
            other = create_split(root, root / 'data/splits/v2', expected_count=12)
            self.assertEqual(manifest, other)
            owners = {}
            for split in ('train', 'val', 'test'):
                for row in read_csv(out / (split + '.csv')):
                    owners[Path(row['image']).stem] = split
                    self.assertNotIn('\\', row['image'])
            self.assertEqual(owners['0'], owners['1'])
            with self.assertRaises(FileExistsError):
                create_split(root, out, expected_count=12)
            (images / '0.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                validate_split(out, root, verify_content=True)
            with (out / 'test.csv').open('a') as stream:
                stream.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_split(out, root)
