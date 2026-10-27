"""Binary, per-image metrics shared by training, calibration and the API."""
import cv2
import numpy as np


def scores(pred, gold):
    pred, gold = np.asarray(pred, dtype=bool), np.asarray(gold, dtype=bool)
    if pred.shape != gold.shape or pred.ndim != 2:
        raise ValueError('Prediction and ground truth must have the same 2D shape')
    tp = int(np.count_nonzero(pred & gold))
    fp = int(np.count_nonzero(pred & ~gold))
    fn = int(np.count_nonzero(~pred & gold))
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'false_positive_pixels': fp, 'false_negative_pixels': fn}


def restore_mask(probability, shape, threshold=.5):
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between 0 and 1')
    pred = (np.asarray(probability) >= threshold).astype(np.uint8)
    return cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
