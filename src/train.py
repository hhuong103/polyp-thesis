"""One trainer for both architectures: matched augmentation, original-size validation, exact resume."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

try:
    from .datasets import AUGMENTATION_POLICY, PolypDataset, collate_original
    from .metrics import restore_mask, scores
    from .paths import ROOT, file_lock, save_json, sha256_file
    from .reproducibility import environment, seed_everything
    from .split_dataset import validate_split
except ImportError:
    from datasets import AUGMENTATION_POLICY, PolypDataset, collate_original
    from metrics import restore_mask, scores
    from paths import ROOT, file_lock, save_json, sha256_file
    from reproducibility import environment, seed_everything
    from split_dataset import validate_split

PROTOCOL = 'grouped_original_resolution_v2'
HISTORY_FIELDS = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'learning_rate', 'seconds']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['unet', 'unetplusplus'], default='unet')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--scheduler-patience', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--split-dir', type=Path, default=ROOT / 'data/splits/grouped_v2')
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--backup-dir', type=Path)
    p.add_argument('--augmentation', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--scheduler', choices=['plateau', 'none'], default='plateau')
    p.add_argument('--encoder-init', choices=['imagenet', 'none'], default='imagenet')
    p.add_argument('--encoder-weights', type=Path, help='Optional local ImageNet ResNet34 state dict')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--calibrate-threshold', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def write_csv(path, history):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS, lineterminator='\n')
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
    for name in ('best_weights.pth', 'last_checkpoint.pth', 'history.csv', 'config.json',
                 'environment.json', 'decision.json', 'run_complete.json'):
        src = local_dir / name
        if src.is_file():
            dst = backup_dir / name
            tmp = dst.with_suffix(dst.suffix + '.tmp')
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def metric_sums(logits, targets):
    """Compatibility helper for tensor-resolution metrics; training selects at original size."""
    pred, gold = torch.sigmoid(logits) >= .5, targets >= .5
    intersection = (pred & gold).sum(dim=(1, 2, 3)).float()
    p, g = pred.sum(dim=(1, 2, 3)).float(), gold.sum(dim=(1, 2, 3)).float()
    union = p + g - intersection
    dice = torch.where(p + g == 0, torch.ones_like(p), 2 * intersection / (p + g).clamp_min(1))
    iou = torch.where(union == 0, torch.ones_like(p), intersection / union.clamp_min(1))
    return dice.sum().item(), iou.sum().item()


def pipeline_hashes():
    names = ('train.py', 'datasets.py', 'metrics.py', 'paths.py', 'reproducibility.py', 'split_dataset.py')
    return {name: sha256_file(Path(__file__).parent / name) for name in names}


def select_threshold(probabilities, originals, candidates):
    table = []
    for threshold in candidates:
        metrics = [scores(restore_mask(prob, gold.shape, threshold), gold)
                   for prob, gold in zip(probabilities, originals)]
        table.append({'threshold': threshold, 'dice': float(np.mean([m['dice'] for m in metrics])),
                      'iou': float(np.mean([m['iou'] for m in metrics]))})
    # Predeclared tie break: closest to 0.5, then smaller threshold.
    selected = max(table, key=lambda r: (r['dice'], -abs(r['threshold'] - .5), -r['threshold']))
    return selected, table


def _train(argv=None):
    args = arguments(argv)
    if (min(args.epochs, args.batch_size, args.patience, args.image_size, args.threads) <= 0
            or args.scheduler_patience < 0 or not math.isfinite(args.lr) or args.lr <= 0 or args.seed < 0):
        raise ValueError('Invalid training parameters')
    if args.image_size % 32:
        raise ValueError('image-size must be a multiple of 32 for ResNet34')
    if args.encoder_weights and args.encoder_init != 'imagenet':
        raise ValueError('Local encoder weights require encoder-init=imagenet')
    args.split_dir = args.split_dir.resolve()
    manifest = validate_split(args.split_dir, ROOT, verify_content=True)
    args.run_name = args.run_name or f'{args.model}_resnet34_{args.split_dir.name}_seed{args.seed}'
    if not args.run_name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('run-name may contain only letters, numbers, hyphens, underscores')
    seed_everything(args.seed, args.deterministic, args.threads)
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    device = torch.device(device_name)
    config = {'schema_version': 2, 'protocol': PROTOCOL, 'model': args.model, 'encoder': 'resnet34',
              'encoder_init': args.encoder_init, 'image_size': args.image_size, 'batch_size': args.batch_size,
              'learning_rate': args.lr, 'weight_decay': .01, 'patience': args.patience, 'seed': args.seed,
              'max_epochs': args.epochs,
              'augmentation': AUGMENTATION_POLICY if args.augmentation else None,
              'optimizer': 'AdamW', 'loss': 'BCEWithLogits + SMP DiceLoss',
              'scheduler': args.scheduler, 'scheduler_patience': args.scheduler_patience,
              'scheduler_factor': .5, 'min_lr': 1e-7, 'selection_min_delta': 1e-4,
              'mask_threshold': .5, 'metric_resolution': 'original', 'metric_aggregation': 'macro_per_image',
              'empty_empty': 1, 'calibrate_threshold': args.calibrate_threshold,
              'threshold_candidates': [round(.3 + i * .05, 2) for i in range(9)],
              'deterministic': args.deterministic, 'threads': args.threads, 'device': device_name,
              'split_dir': args.split_dir.relative_to(ROOT).as_posix(),
              'split_sha256': {s: sha256_file(args.split_dir / (s + '.csv')) for s in ('train', 'val', 'test')},
              'split_manifest_sha256': sha256_file(args.split_dir / 'manifest.json'),
              'patient_video_independence': manifest['patient_video_independence'],
              'local_encoder_sha256': sha256_file(args.encoder_weights) if args.encoder_weights else None,
              'source_sha256': pipeline_hashes()}
    run = ROOT / 'results' / args.run_name
    backup = args.backup_dir / args.run_name if args.backup_dir else None
    best_path, last_path = run / 'best_weights.pth', run / 'last_checkpoint.pth'
    run.mkdir(parents=True, exist_ok=True)
    if args.resume and not last_path.is_file() and backup and (backup / 'last_checkpoint.pth').is_file():
        for name in ('last_checkpoint.pth', 'best_weights.pth', 'history.csv', 'config.json',
                     'environment.json', 'decision.json', 'run_complete.json'):
            if (backup / name).is_file():
                shutil.copy2(backup / name, run / name)
    if not args.resume:
        if any(p.name != 'train.lock' for p in run.iterdir()):
            raise FileExistsError(f'Run already exists: {run}; use --resume or a new name')
        save_json(run / 'config.json', config)
        save_json(run / 'environment.json', environment())
    elif not last_path.is_file():
        raise FileNotFoundError(f'No checkpoint to resume: {last_path}')

    state = torch.load(last_path, map_location='cpu', weights_only=False) if args.resume else None
    if state and state['config'] != config:
        raise ValueError('Checkpoint protocol/config/split/source changed; legacy checkpoints cannot resume under v2')
    if state and json.loads((run / 'config.json').read_text()) != config:
        raise ValueError('Saved config differs from checkpoint')
    if state:
        previous_env = json.loads((run / 'environment.json').read_text())
        if previous_env != environment():
            raise ValueError('Resume environment differs; use the pinned environment for exact continuation')
        if (run / 'run_complete.json').is_file():
            completed = json.loads((run / 'run_complete.json').read_text())
            if completed.get('best_weights_sha256') != sha256_file(best_path):
                raise ValueError('Completed best checkpoint has changed')
            print(f'Already complete: {run}', flush=True)
            return completed

    train_data = PolypDataset(args.split_dir / 'train.csv', args.image_size, root=ROOT, augment=args.augmentation, seed=args.seed)
    val_data = PolypDataset(args.split_dir / 'val.csv', args.image_size, root=ROOT, return_original=True)
    loader_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=loader_rng)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_original)
    cls = smp.Unet if args.model == 'unet' else smp.UnetPlusPlus
    model = cls(encoder_name='resnet34', encoder_weights=(None if args.resume or args.encoder_weights
                or args.encoder_init == 'none' else 'imagenet'), in_channels=3, classes=1)
    if args.encoder_weights and not args.resume:
        model.encoder.load_state_dict(torch.load(args.encoder_weights, map_location='cpu', weights_only=True))
    model = model.to(device)
    bce, dice_loss = torch.nn.BCEWithLogitsLoss(), smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5,
                 patience=args.scheduler_patience, min_lr=1e-7) if args.scheduler == 'plateau' else None)
    history, best_score, stale, start_epoch = [], -1., 0, 0
    if state:
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler_state'])
        start_epoch, history, best_score, stale = state['epoch'], state['history'], state['best_score'], state['stale']
        loader_rng.set_state(state['loader_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        # The checkpoint is authoritative if a previous process stopped between file writes.
        write_csv(run / 'history.csv', history)
        del state
    print(f'Device {device}; {args.model}; train={len(train_data)}, val={len(val_data)}; original-size metrics', flush=True)
    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience:
            break
        tick = time.monotonic()
        train_data.set_epoch(epoch)
        model.train()
        train_total = train_count = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite train loss')
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(images)
            train_count += len(images)
        model.eval()
        val_loss = val_count = 0
        metrics = []
        with torch.inference_mode():
            for images, masks, originals in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * len(images)
                val_count += len(images)
                probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
                metrics.extend(scores(restore_mask(prob, gold.shape), gold) for prob, gold in zip(probs, originals))
        row = {'epoch': epoch + 1, 'train_loss': train_total / train_count, 'val_loss': val_loss / val_count,
               'val_dice': float(np.mean([m['dice'] for m in metrics])),
               'val_iou': float(np.mean([m['iou'] for m in metrics])),
               'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic() - tick}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError('Non-finite validation/history value')
        history.append(row)
        if row['val_dice'] > best_score + 1e-4:
            best_score, stale = row['val_dice'], 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            stale += 1
        if scheduler:
            scheduler.step(row['val_dice'])
        checkpoint = dict(epoch=epoch + 1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                          scheduler_state=scheduler.state_dict() if scheduler else None,
                          history=history, best_score=best_score, stale=stale, config=config,
                          loader_rng=loader_rng.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        atomic_torch_save(checkpoint, last_path)
        del checkpoint
        write_csv(run / 'history.csv', history)
        copy_backup(run, backup)
        print(json.dumps(row), flush=True)
    # Calibration is exclusively on validation after best-checkpoint selection.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    probabilities, originals_all = [], []
    with torch.inference_mode():
        for images, _, originals in val_loader:
            probabilities.extend(torch.sigmoid(model(images.to(device)))[:, 0].cpu().numpy())
            originals_all.extend(originals)
    selected, threshold_table = select_threshold(probabilities, originals_all,
        config['threshold_candidates'] if args.calibrate_threshold else [.5])
    baseline, _ = select_threshold(probabilities, originals_all, [.5])
    if abs(baseline['dice'] - best_score) > 1e-6:
        raise ValueError('Best checkpoint does not reproduce validation selection score')
    validate_split(args.split_dir, ROOT, verify_content=True)
    if pipeline_hashes() != config['source_sha256']:
        raise ValueError('Training source changed during the run')
    decision = {'threshold': selected['threshold'], 'selected_on': 'validation_only',
                'resolution': 'original', 'validation': selected, 'candidates': threshold_table,
                'best_weights_sha256': sha256_file(best_path), 'val_sha256': config['split_sha256']['val']}
    save_json(run / 'decision.json', decision)
    summary = {'status': 'complete', 'protocol': PROTOCOL, 'epochs': len(history),
               'best_epoch': next(r['epoch'] for r in history if r['val_dice'] == best_score),
               'best_val_dice': best_score, 'verified_validation': baseline,
               'stop_reason': 'early_stopping' if stale >= args.patience else 'max_epochs',
               'threshold': decision['threshold'], 'best_weights_sha256': decision['best_weights_sha256'],
               'decision_sha256': sha256_file(run / 'decision.json'),
               'config_sha256': sha256_file(run / 'config.json'), 'test_used_for_selection': False}
    save_json(run / 'run_complete.json', summary)
    copy_backup(run, backup)
    print('Training complete: ' + json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    args = arguments(argv)
    name = args.run_name or f'{args.model}_resnet34_{args.split_dir.name}_seed{args.seed}'
    if not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid run name')
    run = ROOT / 'results' / name
    run.mkdir(parents=True, exist_ok=True)
    with file_lock(run / 'train.lock'):
        return _train(argv)


if __name__ == '__main__':
    main()
