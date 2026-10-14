
from pathlib import Path
import cv2

# 1. Xac dinh thu muc du lieu
# check_dataset.py is located in the project root.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "kvasir_seg"

IMAGE_DIR = DATA / "images"
MASK_DIR = DATA / "masks"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_files(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

# 2. Kiem tra thu muc ton tai
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc images: {IMAGE_DIR}"
    )

if not MASK_DIR.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc masks: {MASK_DIR}"
    )

# 3. Doc danh sach file
images = get_files(IMAGE_DIR)
masks = get_files(MASK_DIR)

print("=== KIEM TRA KVASIR-SEG ===")
print("So anh:", len(images))
print("So mask:", len(masks))

# 4. Kiem tra ghep cap
image_names = set(images)
mask_names = set(masks)

missing_masks = sorted(image_names - mask_names)
missing_images = sorted(mask_names - image_names)

print("Anh thieu mask:", len(missing_masks))
print("Mask thieu anh:", len(missing_images))

# 5. Kiem tra kha nang doc va kich thuoc
invalid = []
valid = 0

for name in sorted(image_names & mask_names):
    image = cv2.imread(str(images[name]))
    mask = cv2.imread(
        str(masks[name]),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        invalid.append((name, "Cannot read"))
        continue

    if image.shape[:2] != mask.shape[:2]:
        invalid.append((name, "Size mismatch"))
        continue

    valid += 1

print("Cap anh-mask hop le:", valid)
print("Cap khong hop le:", len(invalid))

if missing_masks:
    print("Vi du anh thieu mask:", missing_masks[:5])

if missing_images:
    print("Vi du mask thieu anh:", missing_images[:5])

if invalid:
    print("Vi du loi:", invalid[:5])

# 6. Ket luan
if (
    len(images) == 1000
    and len(masks) == 1000
    and valid == 1000
    and not missing_masks
    and not missing_images
    and not invalid
):
    print("PASS: DATASET READY")
else:
    print("CHECK: DATASET NEEDS REVIEW")
