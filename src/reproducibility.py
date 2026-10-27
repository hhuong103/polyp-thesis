"""Explicit reproducibility controls and a recorded execution environment."""
import importlib.metadata
import os
import platform
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=True, threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'opencv-python', 'segmentation-models-pytorch'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'threads': torch.get_num_threads()}
