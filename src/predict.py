from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


ROOT = Path(__file__).resolve().parents[1]

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def load_model(weights_path, device):

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


def preprocess(image, size=256):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            image.transpose(2, 0, 1)
        )
    ).unsqueeze(0)

    return tensor


def predict(image_path, weights_path, output_dir):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    model = load_model(weights_path, device)

    tensor = preprocess(image).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Prediction at the 256x256 model resolution
    mask = (
        probability[0, 0].cpu().numpy() >= 0.5
    ).astype(np.uint8) * 255

    # Restore to original image resolution
    height, width = image.shape[:2]

    mask_original = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = image.copy()
    selected = mask_original > 0

    red = np.array([0, 0, 255], dtype=np.float32)

    overlay[selected] = (
        0.6 * image[selected].astype(np.float32)
        + 0.4 * red
    ).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "prediction_mask.png"),
        mask_original
    )

    cv2.imwrite(
        str(output_dir / "overlay.png"),
        overlay
    )

    print("Device:", device)
    print("Model: U-Net + ResNet34 + Augmentation")
    print("Inference time (ms):", round(elapsed * 1000, 2))
    print("Results saved:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models/final_polyp_model.pth"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/inference"
    )

    args = parser.parse_args()

    predict(
        args.image,
        args.weights,
        args.output
    )
