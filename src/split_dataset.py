"""Create a versioned, group-disjoint split; never overwrite legacy experiment CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

try:
    from .paths import ROOT, dataset_path, read_csv, save_json, sha256_file
except ImportError:
    from paths import ROOT, dataset_path, read_csv, save_json, sha256_file

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def list_files(folder):
    files = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in files:
                raise ValueError(f'Duplicate stem: {path.stem}')
            files[path.stem] = path
    return files


def perceptual_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    values = low.flatten()[1:]
    return sum(int(bit) << i for i, bit in enumerate(values > np.median(values)))


def connected_groups(names, fingerprints, metadata=None, distance=8):
    """Conservative transitive grouping, including re-encoded visual duplicates."""
    if not 0 <= distance <= 63:
        raise ValueError('pHash distance must be between 0 and 63')
    parent = {name: name for name in names}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    edges = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = fingerprints[left], fingerprints[right]
            reasons = []
            if a['image_sha256'] == b['image_sha256'] or a['decoded_sha256'] == b['decoded_sha256']:
                reasons.append('exact_duplicate')
            hamming = (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count()
            if hamming <= distance:
                reasons.append('perceptual_candidate')
            if metadata:
                for key in ('patient_id', 'video_id', 'group_id'):
                    if metadata[left].get(key) and metadata[left].get(key) == metadata[right].get(key):
                        reasons.append(key)
            if reasons:
                union(left, right)
                edges.append({'left': left, 'right': right, 'reasons': reasons, 'hamming': hamming})
    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values()), edges


def allocate_groups(groups, seed=42, ratios=(.7, .15, .15)):
    if len(groups) < 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError('Need at least 3 groups and three positive ratios summing to 1')
    groups = [sorted(g) for g in groups]
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(map(len, groups))
    targets = [int(total * ratios[0]), int(total * ratios[1])]
    targets.append(total - sum(targets))
    selected = {s: [] for s in ('train', 'val', 'test')}
    keys = list(selected)
    for group in groups:
        index = max(range(3), key=lambda i: targets[i] - len(selected[keys[i]]))
        selected[keys[index]].extend(group)
    if any(not v for v in selected.values()):
        raise ValueError('Grouping leaves an empty split; inspect oversized groups')
    return selected


def create_split(root=ROOT, output=None, seed=42, distance=8, metadata_path=None, expected_count=1000):
    root = Path(root).resolve()
    output = Path(output or root / 'data/splits/grouped_v2').resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Immutable split directory already exists: {output}')
    images = list_files(root / 'data/raw/kvasir_seg/images')
    masks = list_files(root / 'data/raw/kvasir_seg/masks')
    names = sorted(images)
    if set(images) != set(masks) or len(names) != expected_count:
        raise ValueError(f'Expected {expected_count} matching pairs; got {len(images)} images/{len(masks)} masks')
    metadata = None
    if metadata_path:
        entries = read_csv(metadata_path)
        metadata = {Path(row['image'].replace('\\', '/')).stem: row for row in entries}
        if len(entries) != len(metadata) or set(metadata) != set(names):
            raise ValueError('Metadata must have exactly one row per image')
        if any(not any(row.get(k) for k in ('patient_id', 'video_id', 'group_id')) for row in entries):
            raise ValueError('Every metadata row needs patient_id, video_id or group_id')
    fingerprints = {}
    for name in names:
        image = cv2.imread(str(images[name]))
        mask = cv2.imread(str(masks[name]), 0)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f'Invalid image/mask pair: {name}')
        fingerprints[name] = {'image_sha256': sha256_file(images[name]), 'mask_sha256': sha256_file(masks[name]),
                              'decoded_sha256': hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
                              'phash': format(perceptual_hash(image), '016x')}
    groups, edges = connected_groups(names, fingerprints, metadata, distance)
    assignments = allocate_groups(groups, seed)
    group_ids = {name: 'g_' + hashlib.sha256('\n'.join(group).encode()).hexdigest()[:16]
                 for group in groups for name in group}
    output.mkdir(parents=True, exist_ok=True)
    for split, selected in assignments.items():
        with (output / (split + '.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image', 'mask', 'group_id'], lineterminator='\n')
            writer.writeheader()
            for name in sorted(selected):
                writer.writerow({'image': images[name].relative_to(root).as_posix(),
                                 'mask': masks[name].relative_to(root).as_posix(), 'group_id': group_ids[name]})
    with (output / 'groups.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['image', 'group_id'])
        writer.writerows((images[name].relative_to(root).as_posix(), group_ids[name]) for name in names)
    manifest = {'schema_version': 1, 'seed': seed, 'counts': {k: len(v) for k, v in assignments.items()},
                'group_count': len(groups), 'nontrivial_groups': [g for g in groups if len(g) > 1],
                'phash': {'method': '63 AC bits of 8x8 DCT', 'max_hamming': distance,
                          'policy': 'Conservatively merge all candidates and transitive components'},
                'patient_video_independence': 'metadata_grouped' if metadata else 'not_verified_metadata_unavailable',
                'metadata_sha256': sha256_file(metadata_path) if metadata_path else None,
                'limitations': ['Perceptual matching cannot prove patient/video independence or find all related frames.'],
                'fingerprints': fingerprints, 'edges': edges,
                'files_sha256': {name: sha256_file(output / name) for name in ('train.csv', 'val.csv', 'test.csv', 'groups.csv')}}
    save_json(output / 'manifest.json', manifest)
    validate_split(output, root, verify_content=True)
    return manifest


def validate_split(directory, root=ROOT, verify_content=False):
    """Verify manifests, group isolation and (before train/eval) actual dataset bytes."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'Split manifest hash mismatch: {name}')
    owners, seen, counts = {}, set(), {}
    for split in ('train', 'val', 'test'):
        rows = read_csv(directory / (split + '.csv'))
        counts[split] = len(rows)
        for row in rows:
            name = Path(row['image'].replace('\\', '/')).stem
            if name in seen:
                raise ValueError(f'Image overlaps splits: {name}')
            seen.add(name)
            group = row['group_id']
            if group in owners and owners[group] != split:
                raise ValueError(f'Group leaks across splits: {group}')
            owners[group] = split
            if verify_content:
                for key in ('image', 'mask'):
                    if sha256_file(dataset_path(row[key], root)) != manifest['fingerprints'][name][key + '_sha256']:
                        raise ValueError(f'Dataset content changed: {row[key]}')
    if counts != manifest['counts'] or seen != set(manifest['fingerprints']):
        raise ValueError('Split counts or membership differ from manifest')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/splits/grouped_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phash-distance', type=int, default=8)
    parser.add_argument('--metadata', type=Path, help='CSV: image plus patient_id/video_id/group_id')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = (validate_split(args.output, verify_content=True) if args.validate_only else
              create_split(output=args.output, seed=args.seed, distance=args.phash_distance, metadata_path=args.metadata))
    print(json.dumps({k: result[k] for k in ('counts', 'group_count', 'patient_video_independence')}, indent=2))


if __name__ == '__main__':
    main()
