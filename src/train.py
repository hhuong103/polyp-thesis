
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from datasets import PolypDataset

# 1. CONFIGURATION
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2. DATASET
train_dataset = PolypDataset(
    ROOT / "data/splits/train.csv",
    image_size=256
)

val_dataset = PolypDataset(
    ROOT / "data/splits/val.csv",
    image_size=256
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 3. MODEL
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

bce_loss = torch.nn.BCEWithLogitsLoss()

dice_loss = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# 4. DICE METRIC
def dice_per_image(logits, targets):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()

    intersection = (predictions * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = predictions.sum(
        dim=(1, 2, 3)
    ) + targets.sum(dim=(1, 2, 3))

    return (2 * intersection + 1e-7) / (
        denominator + 1e-7
    )

# 5. TRAINING
best_dice = -1.0

checkpoint_dir = ROOT / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = (
            bce_loss(logits, masks)
            + dice_loss(logits, masks)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    # 6. VALIDATION
    model.eval()
    dice_sum = 0.0

    with torch.inference_mode():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            scores = dice_per_image(logits, masks)

            dice_sum += scores.sum().item()

    val_dice = dice_sum / len(val_dataset)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # 7. SAVE BEST MODEL
    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            model.state_dict(),
            checkpoint_dir / "best_unet.pth"
        )

        print("Best checkpoint saved!")

print("Training finished.")
print("Best validation Dice:", best_dice)
