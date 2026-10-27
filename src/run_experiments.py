"""Train a matched multi-seed study, freeze every decision, then evaluate test once."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

try:
    from .datasets import PolypDataset, collate_original
    from .experiment_statistics import paired_comparison, seed_summary
    from .metrics import restore_mask, scores
    from .paths import ROOT, file_lock, save_json, sha256_file
    from .reproducibility import environment, seed_everything
    from .split_dataset import validate_split
    from .train import pipeline_hashes
except ImportError:
    from datasets import PolypDataset, collate_original
    from experiment_statistics import paired_comparison, seed_summary
    from metrics import restore_mask, scores
    from paths import ROOT, file_lock, save_json, sha256_file
    from reproducibility import environment, seed_everything
    from split_dataset import validate_split
    from train import pipeline_hashes


def timestamp():
    return datetime.now(timezone.utc).isoformat()


exclusive_lock = file_lock


def command_for(config, model, seed, run_name):
    command = [sys.executable, '-u', '-m', 'src.train', '--model', model, '--seed', str(seed),
               '--run-name', run_name, '--split-dir', str(ROOT / config['split_dir']),
               '--epochs', str(config['epochs']), '--batch-size', str(config['batch_size']),
               '--lr', str(config['learning_rate']), '--patience', str(config['patience']),
               '--scheduler-patience', str(config['scheduler_patience']), '--image-size', str(config['image_size']),
               '--device', config['device'], '--threads', str(config['threads']),
               '--scheduler', config.get('scheduler', 'plateau'),
               '--encoder-init', config['encoder_init']]
    for name in ('augmentation', 'deterministic', 'calibrate_threshold'):
        command.append('--' + ('' if config[name] else 'no-') + name.replace('_', '-'))
    if config.get('encoder_weights'):
        command.extend(['--encoder-weights', str(ROOT / config['encoder_weights'])])
    if config.get('backup_dir'):
        command.extend(['--backup-dir', str(config['backup_dir'])])
    if (ROOT / 'results' / run_name / 'last_checkpoint.pth').is_file():
        command.append('--resume')
    return command


def freeze_run(run):
    complete = json.loads((run / 'run_complete.json').read_text())
    if complete['status'] != 'complete' or complete['test_used_for_selection']:
        raise ValueError(f'Run is not frozen for testing: {run.name}')
    if complete['best_weights_sha256'] != sha256_file(run / 'best_weights.pth'):
        raise ValueError('Weights differ from verified best checkpoint')
    if complete['config_sha256'] != sha256_file(run / 'config.json'):
        raise ValueError('Training config differs from completion record')
    if complete['decision_sha256'] != sha256_file(run / 'decision.json'):
        raise ValueError('Threshold decision differs from completion record')
    decision = json.loads((run / 'decision.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if (decision['selected_on'] != 'validation_only'
            or decision['best_weights_sha256'] != complete['best_weights_sha256']
            or decision['val_sha256'] != config['split_sha256']['val']):
        raise ValueError('Threshold calibration provenance mismatch')
    return {name: sha256_file(run / name) for name in
            ('config.json', 'best_weights.pth', 'decision.json', 'run_complete.json')}


def verify_matched_runs(runs, study_config):
    reference = None
    for run in runs:
        config = json.loads((run / 'config.json').read_text())
        if config['model'] not in study_config['models'] or config['seed'] not in study_config['seeds']:
            raise ValueError('Run architecture/seed differs from study')
        if config['source_sha256'] != pipeline_hashes():
            raise ValueError('Training source changed between paired runs')
        comparable = {key: value for key, value in config.items() if key not in ('model', 'seed')}
        if reference is not None and reference != comparable:
            raise ValueError('Architecture comparison has mismatched training settings')
        reference = comparable


def evaluate_run(run, frozen):
    if freeze_run(run) != frozen:
        raise ValueError('Frozen run changed before test')
    output = run / 'test_evaluation_v2.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if previous['frozen_run'] != frozen:
            raise ValueError('Existing test result has different provenance')
        return previous
    config = json.loads((run / 'config.json').read_text())
    decision = json.loads((run / 'decision.json').read_text())
    split_dir = ROOT / config['split_dir']
    validate_split(split_dir, ROOT, verify_content=True)
    seed_everything(config['seed'], config['deterministic'], config['threads'])
    device = torch.device(config['device'])
    cls = smp.Unet if config['model'] == 'unet' else smp.UnetPlusPlus
    net = cls(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).to(device)
    net.load_state_dict(torch.load(run / 'best_weights.pth', map_location=device, weights_only=True))
    net.eval()
    dataset = PolypDataset(split_dir / 'test.csv', config['image_size'], root=ROOT, return_original=True)
    samples = []
    with torch.inference_mode():
        for images, _, originals in DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_original):
            probabilities = torch.sigmoid(net(images.to(device)))[:, 0].cpu().numpy()
            for probability, gold in zip(probabilities, originals):
                row = dataset.samples[len(samples)]
                samples.append({'image': row['image'], 'group_id': row['group_id'],
                                'fixed_0_5': scores(restore_mask(probability, gold.shape, .5), gold),
                                'validation_selected': scores(restore_mask(probability, gold.shape, decision['threshold']), gold)})
    if freeze_run(run) != frozen:
        raise ValueError('Frozen checkpoint changed during test')
    validate_split(split_dir, ROOT, verify_content=True)
    report = {'created_at': timestamp(), 'model': config['model'], 'seed': config['seed'],
              'count': len(samples), 'threshold': decision['threshold'], 'resolution': 'original',
              'frozen_run': frozen, 'environment': environment(), 'samples': samples,
              'means': {policy: {key: float(np.mean([s[policy][key] for s in samples])) for key in ('dice', 'iou')}
                        for policy in ('fixed_0_5', 'validation_selected')}}
    save_json(output, report)
    return report


def summarize(reports, config, manifest):
    models, seeds = config['models'], config['seeds']
    lookup = {(r['model'], r['seed']): r for r in reports}
    reference = lookup[(models[0], seeds[0])]['samples']
    identities = [(r['image'], r['group_id']) for r in reference]
    for report in reports:
        if [(r['image'], r['group_id']) for r in report['samples']] != identities:
            raise ValueError('Test image/group order differs across paired runs')
    groups = [r['group_id'] for r in reference]
    result = {'created_at': timestamp(), 'models': {}, 'comparisons': {},
              'patient_video_independence': manifest['patient_video_independence'],
              'limitations': manifest['limitations'], 'seeds': seeds,
              'metric_resolution': 'original', 'test_used_for_selection': False,
              'primary_endpoint': 'fixed_0_5.dice',
              'secondary_endpoints': 'IoU and validation-calibrated results are secondary/exploratory; p-values are unadjusted.'}
    for policy in ('fixed_0_5', 'validation_selected'):
        result['models'][policy], result['comparisons'][policy] = {}, {}
        for model in models:
            result['models'][policy][model] = {
                metric: seed_summary([lookup[(model, seed)]['means'][policy][metric] for seed in seeds])
                for metric in ('dice', 'iou')}
        for metric in ('dice', 'iou'):
            arrays = [np.array([[s[policy][metric] for s in lookup[(model, seed)]['samples']]
                                for seed in seeds]) for model in models]
            result['comparisons'][policy][metric] = paired_comparison(*arrays, groups,
                repetitions=config['bootstrap_repetitions'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment_cpu.json')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if (set(config['models']) != {'unet', 'unetplusplus'} or len(config['models']) != 2
            or len(config['seeds']) < 2 or len(set(config['seeds'])) != len(config['seeds'])):
        raise ValueError('A study requires both architectures and at least two unique paired seeds')
    if not config['name'].replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid experiment name')
    split = ROOT / config['split_dir']
    manifest = validate_split(split, ROOT, verify_content=True)
    out = ROOT / 'results/experiments' / config['name']
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / 'runner.lock'):
        bound = {**config, 'split_manifest_sha256': sha256_file(split / 'manifest.json'),
                 'training_source_sha256': pipeline_hashes(),
                 'runner_source_sha256': sha256_file(Path(__file__)),
                 'statistics_source_sha256': sha256_file(Path(__file__).with_name('experiment_statistics.py'))}
        config_path = out / 'experiment.json'
        if config_path.exists() and json.loads(config_path.read_text()) != bound:
            raise ValueError('Experiment config/source/split changed; choose a new study name')
        save_json(config_path, bound)
        def status(phase, **fields):
            state = {'phase': phase, 'updated_at': timestamp(), 'runner_pid': os.getpid(), **fields}
            save_json(out / 'status.json', state)
            print(json.dumps(state), flush=True)
        try:
            runs = []
            total = len(config['models']) * len(config['seeds'])
            for seed in config['seeds']:
                for model in config['models']:
                    name = f'{model}_{config["name"]}_seed{seed}'
                    status('training', run=name, completed_runs=len(runs), total_runs=total)
                    with (out / (name + '.log')).open('a', encoding='utf-8') as log:
                        subprocess.run(command_for(config, model, seed, name), cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    runs.append(ROOT / 'results' / name)
            # Freeze ALL architectures/seeds before exposing any test scores.
            verify_matched_runs(runs, config)
            frozen = {run.name: freeze_run(run) for run in runs}
            freeze_path = out / 'frozen_runs.json'
            if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
                raise ValueError('Frozen study changed after test access')
            save_json(freeze_path, frozen)
            reports = []
            for run in runs:
                status('evaluating', run=run.name, completed_runs=len(reports), total_runs=total)
                reports.append(evaluate_run(run, frozen[run.name]))
            summary = summarize(reports, config, manifest)
            save_json(out / 'summary.json', summary)
            with (out / 'test_samples.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(['model', 'seed', 'image', 'group_id', 'policy', 'dice', 'iou'])
                for report in reports:
                    for sample in report['samples']:
                        for policy in ('fixed_0_5', 'validation_selected'):
                            writer.writerow([report['model'], report['seed'], sample['image'], sample['group_id'],
                                             policy, sample[policy]['dice'], sample[policy]['iou']])
            # A predeclared seed for deployment avoids selecting a seed on its test score.
            selected_seed = config['seeds'][0]
            deployment = []
            for model in config['models']:
                run = ROOT / 'results' / f'{model}_{config["name"]}_seed{selected_seed}'
                decision = json.loads((run / 'decision.json').read_text())
                deployment.append({'id': model + '-grouped', 'name': model + ' · grouped v2',
                                   'architecture': model, 'encoder': 'resnet34', 'image_size': config['image_size'],
                                   'weights': (run / 'best_weights.pth').relative_to(ROOT).as_posix(),
                                   'history': (run / 'history.csv').relative_to(ROOT).as_posix(),
                                   'threshold': decision['threshold'], 'experiment': config['name'],
                                   'weights_sha256': decision['best_weights_sha256'],
                                   'note': 'Split theo nhóm ảnh gần trùng; chưa xác minh độc lập bệnh nhân/video.'})
            save_json(out / 'deployment.json', deployment)
            status('complete', completed_runs=total, total_runs=total,
                   summary=out.relative_to(ROOT).as_posix() + '/summary.json')
        except Exception as exc:
            status('error', error=str(exc))
            raise


if __name__ == '__main__':
    main()
