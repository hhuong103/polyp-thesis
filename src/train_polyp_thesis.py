"""Reproducible baseline trainer for Kvasir-SEG / SMP.

Place as polyp-thesis/src/train.py. Expects src/datasets.py with PolypDataset.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from datasets import PolypDataset

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['unet', 'unetplusplus'], default='unet')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--backup-dir', type=str, default=None,
                   help='Example: /content/drive/MyDrive/PolypThesis/experiments')
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, history):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou'])
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def atomic_torch_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def copy_backup(local_dir, backup_dir):
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ('best_weights.pth', 'last_checkpoint.pth', 'history.csv', 'config.json'):
        src = local_dir / name
        if src.is_file():
            dst = backup_dir / name
            tmp = dst.with_suffix(dst.suffix + '.tmp')
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def metric_sums(logits, targets):
    pred = torch.sigmoid(logits) >= 0.5
    gold = targets >= 0.5
    axes = (1, 2, 3)
    intersection = (pred & gold).sum(dim=axes).float()
    p = pred.sum(dim=axes).float()
    g = gold.sum(dim=axes).float()
    union = p + g - intersection
    # Both empty -> perfect score. This convention must be documented.
    dice = torch.where(p + g == 0, torch.ones_like(p), 2 * intersection / (p + g).clamp_min(1))
    iou = torch.where(union == 0, torch.ones_like(union), intersection / union.clamp_min(1))
    return dice.sum().item(), iou.sum().item()


def main():
    args = arguments()
    if min(args.epochs, args.batch_size, args.patience, args.image_size) <= 0 or args.lr <= 0:
        raise ValueError('epochs, batch-size, patience, image-size, lr must be positive')
    if args.run_name is None:
        args.run_name = f'{args.model}_resnet34_seed{args.seed}'
    if not args.run_name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('run-name may contain only letters, numbers, hyphens, underscores')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    split_dir = ROOT / 'data' / 'splits'
    split_hashes = {x: sha256_file(split_dir / f'{x}.csv') for x in ('train', 'val', 'test')}
    config = {
        'model': args.model, 'encoder': 'resnet34', 'encoder_init': 'imagenet',
        'image_size': args.image_size, 'batch_size': args.batch_size,
        'learning_rate': args.lr, 'patience': args.patience, 'seed': args.seed,
        'optimizer': 'AdamW', 'loss': 'BCEWithLogits + SMP DiceLoss',
        'mask_threshold': 0.5, 'split_sha256': split_hashes,
    }
    run_dir = ROOT / 'results' / args.run_name
    backup_root = Path(args.backup_dir) if args.backup_dir else None
    backup_dir = backup_root / args.run_name if backup_root is not None else None
    best_path = run_dir / 'best_weights.pth'
    last_path = run_dir / 'last_checkpoint.pth'
    hist_path = run_dir / 'history.csv'
    config_path = run_dir / 'config.json'

    if not args.resume:
        if last_path.exists() or (backup_dir and (backup_dir / last_path.name).exists()):
            raise FileExistsError('Run already exists; use --resume or a different --run-name')
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not last_path.exists():
            if backup_dir and (backup_dir / last_path.name).exists():
                for fname in ('last_checkpoint.pth', 'best_weights.pth', 'history.csv', 'config.json'):
                    original = backup_dir / fname
                    if original.exists():
                        shutil.copy2(original, run_dir / fname)
            else:
                raise FileNotFoundError('No last_checkpoint.pth found locally or in backup-dir')

    train_data = PolypDataset(split_dir / 'train.csv', image_size=args.image_size)
    val_data = PolypDataset(split_dir / 'val.csv', image_size=args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_class = smp.Unet if args.model == 'unet' else smp.UnetPlusPlus
    # On resume, weights are loaded from checkpoint; avoid unnecessary pretrained download.
    model = model_class(
        encoder_name='resnet34',
        encoder_weights=None if args.resume else 'imagenet',
        in_channels=3, classes=1,
    ).to(device)
    bce = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def loss_fn(logits, mask):
        return bce(logits, mask) + dice_loss(logits, mask)

    history = []
    best_score = -1.0
    stale = 0
    start_epoch = 0

    if args.resume:
        # A checkpoint from your own training; do not load untrusted pickle files.
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Checkpoint configuration/split hash differs from this run')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch = state['epoch']
        history = state['history']
        best_score = state['best_score']
        stale = state['stale']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resumed at epoch {start_epoch}', flush=True)

    print(f'Device: {device}; model: {args.model}; train: {len(train_data)}; val: {len(val_data)}', flush=True)
    print(f'Train batches: {len(train_loader)}; Validation batches: {len(val_loader)}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            n = images.size(0)
            train_total += loss.item() * n
            train_count += n

        model.eval()
        val_total = dice_total = iou_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss = loss_fn(logits, masks)
                ds, js = metric_sums(logits, masks)
                n = images.size(0)
                val_total += val_loss.item() * n
                dice_total += ds
                iou_total += js
                val_count += n

        row = dict(epoch=epoch + 1, train_loss=train_total / train_count,
                   val_loss=val_total / val_count, val_dice=dice_total / val_count,
                   val_iou=iou_total / val_count)
        history.append(row)
        write_csv(hist_path, history)
        print('Epoch {epoch}/{epochs} | train={train_loss:.4f} | val_loss={val_loss:.4f} | '
              'Dice={val_dice:.4f} | IoU={val_iou:.4f}'.format(epochs=args.epochs, **row), flush=True)

        improved = row['val_dice'] > best_score + 1e-4
        if improved:
            best_score = row['val_dice']
            stale = 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            stale += 1

        state = dict(epoch=epoch + 1, model_state=model.state_dict(),
                     optimizer_state=optimizer.state_dict(), history=history,
                     best_score=best_score, stale=stale, config=config,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        atomic_torch_save(state, last_path)
        copy_backup(run_dir, backup_dir)
        if stale >= args.patience:
            print('Early stopping: validation Dice plateaued', flush=True)
            break

    print(f'Done. Best validation Dice: {best_score:.4f}; outputs: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
