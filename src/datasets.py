"""Portable dataset with paired, epoch-addressable training augmentation."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .paths import ROOT, dataset_path
except ImportError:
    from paths import ROOT, dataset_path

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)
AUGMENTATION_POLICY = {'name': 'paired_flips_rot90_v1', 'horizontal_flip_p': .5,
                       'vertical_flip_p': .5, 'rotation_quarter_turns': [0, 1, 2, 3]}


class PolypDataset(Dataset):
    def __init__(self, csv_path, image_size=256, *, root=ROOT, augment=False,
                 seed=42, return_original=False):
        self.csv_path = Path(csv_path).resolve()
        self.root = Path(root).resolve()
        if image_size <= 0:
            raise ValueError('image_size must be positive')
        self.image_size = image_size
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.return_original = return_original
        if augment and return_original:
            raise ValueError('Original-resolution evaluation cannot use augmentation')
        with self.csv_path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {'image', 'mask'}.issubset(reader.fieldnames or []):
                raise ValueError('Split CSV must contain image and mask columns')
            self.samples = list(reader)
        if not self.samples:
            raise ValueError(f'Empty split: {self.csv_path}')
        for row in self.samples:
            for key in ('image', 'mask'):
                path = dataset_path(row[key], self.root)
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {key}: {path}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image_path = dataset_path(row['image'], self.root)
        mask_path = dataset_path(row['mask'], self.root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f'Cannot read image or mask: {row}')
        if image.shape[:2] != mask.shape:
            raise ValueError(f'Image/mask dimensions differ: {row}')
        original = mask > (0 if mask.max() <= 1 else 127)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(original.astype(np.uint8), (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            for axis in (0, 1):
                if rng.random() < .5:
                    image, mask = np.flip(image, axis), np.flip(mask, axis)
            turns = int(rng.integers(4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))
        return (image, mask, original) if self.return_original else (image, mask)


def collate_original(samples):
    images, masks, originals = zip(*samples)
    return torch.stack(images), torch.stack(masks), originals
