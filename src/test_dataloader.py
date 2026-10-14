
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import PolypDataset

ROOT = Path(__file__).resolve().parents[1]

train_csv = ROOT / "data" / "splits" / "train.csv"

dataset = PolypDataset(
    csv_path=train_csv,
    image_size=256
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, masks = next(iter(loader))

print("Dataset size:", len(dataset))
print("Images:", images.shape)
print("Masks:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Mask values:", masks.unique())
