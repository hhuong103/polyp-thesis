"""Read-only live epoch display for an existing study; Ctrl+C stops only the viewer."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

try:
    from .paths import ROOT
except ImportError:
    from paths import ROOT

VIETNAM = timezone(timedelta(hours=7))


def process_alive(pid):
    """Check the PID without sending signals or interacting with training locks."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        return None


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def snapshot(config, root=ROOT):
    root = Path(root)
    name = config['name']
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid study name')
    folder = root / 'results/experiments' / name
    state = read_json(folder / 'status.json', {'phase': 'not_started'})
    runs = []
    for seed in config['seeds']:
        for model in config['models']:
            if model not in ('unet', 'unetplusplus') or not isinstance(seed, int) or seed < 0:
                raise ValueError('Invalid architecture or seed')
            run_name = f'{model}_{name}_seed{seed}'
            run_dir = root / 'results' / run_name
            history_path = run_dir / 'history.csv'
            rows = []
            modified = None
            try:
                with history_path.open(newline='', encoding='utf-8-sig') as stream:
                    for row in csv.DictReader(stream):
                        rows.append({'epoch': int(row['epoch']), **{key: float(row[key]) for key in
                                     ('train_loss', 'val_loss', 'val_dice', 'val_iou', 'learning_rate', 'seconds')}})
                modified = history_path.stat().st_mtime
            except FileNotFoundError:
                pass
            completion = read_json(run_dir / 'run_complete.json', {})
            runs.append({'name': run_name, 'model': model, 'seed': seed, 'rows': rows,
                         'saved_epoch': rows[-1]['epoch'] if rows else 0,
                         'complete': completion.get('status') == 'complete',
                         'stop_reason': completion.get('stop_reason'), 'history_modified': modified})
    current = next((r for r in runs if r['name'] == state.get('run')), None)
    execution = read_json(root / 'results' / current['name'] / 'environment.json', {}) if current else {}
    execution_platform = execution.get('platform', '')
    foreign_os = ((execution_platform.startswith('Windows') and os.name != 'nt') or
                  (execution_platform.startswith('Linux') and os.name == 'nt'))
    note = ('Runner belongs to another OS; use its Python to verify the PID.' if foreign_os else '')
    return {'name': name, 'folder': str(folder), 'phase': state.get('phase', 'unknown'),
            'runner_alive': None if foreign_os else process_alive(state.get('runner_pid')),
            'process_check_note': note,
            'active_run': state.get('run'), 'error': state.get('error'),
            'epoch_limit': config['epochs'], 'runs': runs,
            'completed_runs': sum(r['complete'] for r in runs)}


def duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes:02d}:{seconds:02d}'


def render(view, tail=8, now=None):
    now = time.time() if now is None else now
    alive = {True: 'ACTIVE', False: 'STOPPED', None: 'UNKNOWN'}[view['runner_alive']]
    lines = ['POLYP THESIS - LIVE EPOCH MONITOR',
             datetime.fromtimestamp(now, VIETNAM).strftime('%d/%m/%Y %H:%M:%S (Vietnam)'),
             f'Study: {view["name"]} | phase: {view["phase"]} | runner: {alive}',
             f'Runs complete: {view["completed_runs"]}/{len(view["runs"])}', '',
             'Model          Seed   Saved epoch   State']
    for run in view['runs']:
        state = ('complete' if run['complete'] else 'active' if run['name'] == view['active_run']
                 and view['runner_alive'] is True else 'unverified' if run['name'] == view['active_run']
                 and view['runner_alive'] is None else 'paused' if run['rows'] else 'queued')
        lines.append(f'{run["model"]:<14} {run["seed"]:<6} {run["saved_epoch"]:>3}/{view["epoch_limit"]:<7} {state}')
    current = next((r for r in view['runs'] if r['name'] == view['active_run']), None)
    if current:
        lines.extend(['', f'Current run: {current["name"]}',
                      'Epoch  Train loss  Val loss  Val Dice  Val IoU    LR          Time'])
        for row in current['rows'][-tail:]:
            lines.append(f'{row["epoch"]:>5}  {row["train_loss"]:>10.4f}  {row["val_loss"]:>8.4f}  '
                         f'{row["val_dice"]:>8.4f}  {row["val_iou"]:>7.4f}  '
                         f'{row["learning_rate"]:>10.2e}  {duration(row["seconds"])}')
        if current['rows']:
            average = sum(r['seconds'] for r in current['rows'][-5:]) / len(current['rows'][-5:])
            lines.append(f'Recent average per epoch: {duration(average)} (train + validation).')
        if current['history_modified'] is not None:
            lines.append(f'Time since history was written: {duration(now - current["history_modified"])}.')
        if (view['phase'] == 'training' and view['runner_alive'] is True and not current['complete']
                and current['saved_epoch'] < view['epoch_limit']):
            lines.append(f'Waiting for completed epoch {current["saved_epoch"] + 1} or finalization.')
    if view['error']:
        lines.extend(['', 'ERROR: ' + str(view['error'])])
    if view.get('process_check_note'):
        lines.extend(['', view['process_check_note']])
    if view['phase'] != 'complete' and view['runner_alive'] is False:
        lines.extend(['', 'Runner is not alive. The stored phase is not a live heartbeat.',
                      'Resume separately with: python -m src.launch_experiments --config configs/experiment_cpu.json'])
    lines.extend(['', 'Metrics are saved AFTER each epoch; no batch-level progress is available.',
                  'Early stopping can finish a run before the epoch limit.',
                  'Ctrl+C closes this monitor only; background training continues.'])
    return '\n'.join(lines)


def enable_terminal_refresh(stream):
    if not stream.isatty():
        return False
    if os.name != 'nt':
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = kernel.GetStdHandle(wintypes.DWORD(-11))
    mode = wintypes.DWORD()
    return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)) and kernel.SetConsoleMode(handle, mode.value | 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment_cpu.json')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--tail', type=int, default=8)
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0 or args.tail <= 0:
        parser.error('interval and tail must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    refresh = not args.once and enable_terminal_refresh(sys.stdout)
    previous = None
    try:
        while True:
            try:
                view = snapshot(config)
                output = render(view, args.tail)
                fingerprint = json.dumps(view, sort_keys=True)
                finished = view['phase'] == 'complete'
            except (OSError, ValueError, KeyError, csv.Error) as exc:
                output = 'Cannot read progress yet: ' + str(exc)
                fingerprint, finished = output, False
            if refresh or args.once or fingerprint != previous:
                if refresh:
                    print('\033[2J\033[H', end='')
                print(output, flush=True)
            previous = fingerprint
            if args.once or finished:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nMonitor stopped. Background training was not interrupted.', flush=True)


if __name__ == '__main__':
    main()
