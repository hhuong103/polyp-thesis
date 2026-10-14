
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "kvasir_seg"
SPLITS = ROOT / "data" / "splits"

IMAGE_DIR = DATA / "images"
MASK_DIR = DATA / "masks"

EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

def list_files(folder):
    files = {}

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in files:
                raise ValueError(f"Duplicate stem: {path.stem}")
            files[path.stem] = path

    return files

images = list_files(IMAGE_DIR)
masks = list_files(MASK_DIR)

if set(images) != set(masks):
    raise RuntimeError("Image and mask names do not match")

names = sorted(images)

if len(names) != 1000:
    raise RuntimeError(f"Expected 1000 images, got {len(names)}")

# Kiem tra anh trung hoan toan bang SHA256
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hashes = {}

for name in names:
    digest = sha256(images[name])
    hashes.setdefault(digest, []).append(name)

duplicates = [v for v in hashes.values() if len(v) > 1]

if duplicates:
    print("Exact duplicate groups:", duplicates[:10])
    raise RuntimeError("Review duplicate images before splitting")

# Random split with fixed seed
rng = random.Random(SEED)
rng.shuffle(names)

train = names[:700]
val = names[700:850]
test = names[850:]

SPLITS.mkdir(parents=True, exist_ok=True)

def save_split(filename, selected):
    with open(SPLITS / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask"])

        for name in selected:
            writer.writerow([
                str(images[name].relative_to(ROOT)),
                str(masks[name].relative_to(ROOT))
            ])

save_split("train.csv", train)
save_split("val.csv", val)
save_split("test.csv", test)

assert not (set(train) & set(val))
assert not (set(train) & set(test))
assert not (set(val) & set(test))

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Split files saved in:", SPLITS)
print("WARNING: Near-duplicate and patient/video grouping")
print("must be reviewed before final thesis experiments.")
