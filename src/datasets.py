
from pathlib import Path
import csv
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

class PolypDataset(Dataset):
    def __init__(self, csv_path, image_size=256):
        self.csv_path = Path(csv_path).resolve()
        self.root = self.csv_path.parents[2]
        self.image_size = image_size

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]

        image_path = self.root / row["image"]
        mask_path = self.root / row["mask"]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(
                f"Cannot read image or mask: {row}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0

        # ImageNet normalization for pretrained ResNet34
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        mask = torch.from_numpy(
            mask[None, :, :].copy()
        )

        return image, mask
