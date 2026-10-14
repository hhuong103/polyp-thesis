
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "kvasir_seg"

IMAGE_DIR = DATA / "images"
MASK_DIR = DATA / "masks"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay thu muc: {folder}"
        )
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.suffix.lower() in EXTENSIONS
    }

images = get_files(IMAGE_DIR)
masks = get_files(MASK_DIR)

common_names = sorted(set(images) & set(masks))

if not common_names:
    raise RuntimeError(
        "Khong tim thay cap anh-mask phu hop"
    )

# Lay cap anh-mask dau tien
name = common_names[0]

image = cv2.imread(str(images[name]))
mask = cv2.imread(
    str(masks[name]),
    cv2.IMREAD_GRAYSCALE
)

if image is None or mask is None:
    raise RuntimeError("Khong doc duoc anh hoac mask")

if image.shape[:2] != mask.shape[:2]:
    raise RuntimeError("Kich thuoc anh va mask khong khop")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyen mask sang nhi phan
binary_mask = mask > 127

# Tao anh overlay
overlay = image.copy()

overlay[binary_mask] = (
    0.6 * image[binary_mask]
    + 0.4 * np.array([255, 0, 0])
).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(binary_mask, cmap="gray")
axes[1].set_title("Ground Truth Mask")

axes[2].imshow(overlay)
axes[2].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Image:", images[name].name)
print("Mask:", masks[name].name)
print("Visualization completed!")
