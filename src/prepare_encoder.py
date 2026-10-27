"""Freeze a single ImageNet encoder artifact for every architecture and seed."""
import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

try:
    from .paths import ROOT, save_json, sha256_file
except ImportError:
    from paths import ROOT, save_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/initialization/resnet34_imagenet.pth')
    args = parser.parse_args()
    metadata = args.output.with_suffix('.json')
    if args.output.exists():
        import json
        if not metadata.is_file() or json.loads(metadata.read_text())['sha256'] != sha256_file(args.output):
            raise ValueError('Existing encoder artifact lacks matching provenance')
        print('Verified encoder:', args.output, flush=True)
        return
    net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    torch.save(net.encoder.state_dict(), temporary)
    temporary.replace(args.output)
    save_json(metadata, {'source': 'segmentation_models_pytorch ResNet34 imagenet pretrained encoder',
                         'smp_version': smp.__version__, 'sha256': sha256_file(args.output)})
    print('Frozen encoder:', args.output, flush=True)


if __name__ == '__main__':
    main()
