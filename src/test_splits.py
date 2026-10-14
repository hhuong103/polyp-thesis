
from pathlib import Path
from datasets import PolypDataset

ROOT = Path(__file__).resolve().parents[1]

for split in ["train", "val", "test"]:
    csv_path = ROOT / "data" / "splits" / f"{split}.csv"

    dataset = PolypDataset(
        csv_path=csv_path,
        image_size=256
    )

    image, mask = dataset[0]

    print(
        split,
        "size:", len(dataset),
        "image:", tuple(image.shape),
        "mask:", tuple(mask.shape)
    )
