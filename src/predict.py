"""Predict using the actual architecture/config and a validation-calibrated threshold."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

try:
    from .datasets import MEAN, STD
    from .metrics import restore_mask
    from .paths import ROOT, sha256_file
except ImportError:
    from datasets import MEAN, STD
    from metrics import restore_mask
    from paths import ROOT, sha256_file


def load_model(weights_path, device, architecture='unet'):
    cls = {'unet': smp.Unet, 'unetplusplus': smp.UnetPlusPlus}[architecture]
    model = cls(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    return model.to(device).eval()


def preprocess(image, size=256):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)


def prediction_config(weights_path, explicit=None):
    weights_path = Path(weights_path)
    candidate = Path(explicit) if explicit else weights_path.parent / 'config.json'
    if not explicit and weights_path.resolve() == (ROOT / 'models/final_polyp_model.pth').resolve():
        candidate = ROOT / 'models/model_config.json'
    if not candidate.is_file():
        raise FileNotFoundError('Model config not found; supply --config rather than guessing architecture')
    config = json.loads(candidate.read_text(encoding='utf-8'))
    config = {**config, 'threshold': config.get('mask_threshold', .5)}
    decision_path = weights_path.parent / 'decision.json'
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        if decision['best_weights_sha256'] != sha256_file(weights_path):
            raise ValueError('Threshold calibration does not belong to these weights')
        config['threshold'] = decision['threshold']
    return config


def predict(image_path, weights_path, output_dir, config_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    config = prediction_config(weights_path, config_path)
    model = load_model(weights_path, device, config['model'])
    tensor = preprocess(image, config['image_size']).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = restore_mask(probability, image.shape[:2], config['threshold'])
    overlay = image.copy()
    overlay[pred] = (.6 * image[pred] + .4 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (('prediction_mask.png', pred.astype(np.uint8) * 255), ('overlay.png', overlay)):
        if not cv2.imwrite(str(output_dir / name), array):
            raise RuntimeError(f'Could not save {name}')
    augmentation = (config.get('augmentation') if config.get('schema_version') == 2
                    or config.get('augmentation_verified') else 'not verified')
    print('Device:', device)
    print('Model:', config['model'], '+', config['encoder'])
    print('Training augmentation:', augmentation)
    print('Threshold:', config['threshold'])
    print('Inference time (ms):', round(elapsed * 1000, 2))
    print('Results saved:', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/final_polyp_model.pth')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/inference')
    args = parser.parse_args()
    predict(args.image, args.weights, args.output, args.config)
