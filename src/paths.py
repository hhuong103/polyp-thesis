"""Portable paths and immutable artifact fingerprints (no ML dependencies)."""
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


@contextmanager
def file_lock(path):
    """A process-lifetime, nonblocking lock on Windows and POSIX."""
    with Path(path).open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def dataset_path(value, root=ROOT):
    """Read Windows or POSIX CSV paths, confined to the project data directory."""
    value = str(value)
    if PureWindowsPath(value).is_absolute() or Path(value).is_absolute():
        raise ValueError(f'Dataset paths must be project relative: {value}')
    path = (Path(root) / value.replace('\\', '/')).resolve()
    if not path.is_relative_to((Path(root) / 'data').resolve()):
        raise ValueError(f'Dataset path escapes data directory: {value}')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)
