"""Launch or resume a study; an existing runner is a normal, successful no-op."""
import argparse
from contextlib import contextmanager, ExitStack
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .paths import ROOT
except ImportError:
    from paths import ROOT


class StudyAlreadyRunning(BlockingIOError):
    """Another process owns the study lock."""


@contextmanager
def launcher_lock(path):
    """Acquire before reading/writing: Windows denies I/O on another process's locked byte.

    Kept in the launcher so fixing startup does not change fingerprinted training
    sources or invalidate an existing checkpoint's exact-resume provenance.
    """
    with Path(path).open('a+b', buffering=0) as stream:
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise StudyAlreadyRunning('Study lock is held by another process') from exc
            raise
        try:
            # Locking beyond EOF is supported. Initialize only after acquiring it.
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b'0')
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment_cpu.json')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not config['name'].replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid experiment name')
    out = ROOT / 'results/experiments' / config['name']
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            stack.enter_context(launcher_lock(out / 'runner.lock'))
        except StudyAlreadyRunning:
            result = {'status': 'already_running',
                      'message': 'Study already running; no duplicate worker was started.',
                      'status_file': str(out / 'status.json')}
            print(json.dumps(result), flush=True)
            return result
        with (out / 'runner.log').open('a', encoding='utf-8') as log:
            options = ({'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == 'nt' else {'start_new_session': True})
            process = subprocess.Popen([sys.executable, '-u', '-m', 'src.run_experiments',
                                        '--config', str(args.config.resolve())], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       close_fds=True, **options)
    result = {'status': 'launched', 'launcher_pid': process.pid, 'status_file': str(out / 'status.json'),
              'log_file': str(out / 'runner.log')}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
