"""User-authorized Windows shutdown only after the local run and test succeed.

Arm: python src/shutdown_after_training.py --watch
Cancel: python src/shutdown_after_training.py --cancel
Inspect: python src/shutdown_after_training.py --status
The 120-second grace period is implemented here; shutdown /t 0 avoids implicit /f.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/unetplusplus_resnet34_seed42"
STATUS = RUN / "shutdown_status.json"
CANCEL = RUN / "shutdown_cancel.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def completed(config_hash):
    lifecycle = read_json(RUN / "lifecycle.json")
    if lifecycle.get("phase") == "error":
        raise ValueError("Training/evaluation failed: " + str(lifecycle.get("error")))
    if lifecycle.get("phase") != "complete":
        return False
    if digest(RUN / "config.json") != config_hash:
        raise ValueError("Run configuration changed; shutdown cancelled")
    summary = read_json(RUN / "completion.json")
    if summary.get("status") != "complete" or lifecycle.get("summary") != summary:
        raise ValueError("Final completion records do not match")
    if not (1 <= summary["best_epoch"] <= summary["epochs"] <= 50):
        raise ValueError("Invalid completed epoch count")
    report = read_json(RUN / "test_evaluation.json")
    if {m["id"] for m in report["models"]} != {"final-unet", "unetplusplus"}:
        raise ValueError("Final report must contain both models")
    expected = {m["id"]: m for m in summary["test_results"]}
    if set(expected) != {"final-unet", "unetplusplus"}:
        raise ValueError("Missing final test results")
    for model in report["models"]:
        if model["count"] != 150 or len(model["samples"]) != 150:
            raise ValueError("Test evaluation is incomplete")
        for key in ("dice", "iou"):
            value = model[key]
            if not math.isfinite(value) or not 0 <= value <= 1 or value != expected[model["id"]][key]:
                raise ValueError("Final test metrics do not match")
    with (RUN / "history.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != summary["epochs"] or int(rows[-1]["epoch"]) != summary["epochs"]:
        raise ValueError("Epoch history is incomplete")
    for name in ("best_weights.pth", "last_checkpoint.pth", "test_evaluation.csv"):
        if not (RUN / name).is_file() or not (RUN / name).stat().st_size:
            raise ValueError("Missing saved output: " + name)
    if digest(RUN / "best_weights.pth") != summary["best_weights_sha256"]:
        raise ValueError("Weights differ from verified checkpoint")
    return True


def shutdown_windows():
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/shutdown.exe"
    # No /f and no positive /t: allow Windows to protect applications with unsaved work.
    subprocess.run([str(executable), "/s", "/t", "0"], check=True, timeout=20)


def monitor(shutdown=shutdown_windows, grace_seconds=120, poll_seconds=10):
    request_id = uuid.uuid4().hex
    config_hash = digest(RUN / "config.json")
    def status(phase, **fields):
        data = {"request_id": request_id, "phase": phase, "watcher_pid": os.getpid(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_hash, **fields}
        write_json(STATUS, data)
        print(json.dumps(data), flush=True)
    def cancelled():
        return CANCEL.is_file() and read_json(CANCEL).get("request_id") == request_id
    try:
        status("waiting_for_completion")
        while not completed(config_hash):
            if cancelled():
                status("cancelled")
                return
            time.sleep(poll_seconds)
        deadline = time.monotonic() + grace_seconds
        status("countdown", seconds_remaining=grace_seconds)
        while time.monotonic() < deadline:
            if cancelled():
                status("cancelled")
                return
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
        if cancelled():
            status("cancelled")
            return
        if not completed(config_hash):
            raise ValueError("Completion state changed during countdown")
        if cancelled():
            status("cancelled")
            return
        status("shutdown_requested")
        shutdown()
    except Exception as exc:
        status("error", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--cancel", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.cancel:
        state = read_json(STATUS)
        write_json(CANCEL, {"request_id": state["request_id"]})
        print("Auto-shutdown cancellation recorded.")
    elif args.status:
        print(json.dumps(read_json(STATUS), indent=2))
    else:
        if os.name != "nt":
            parser.error("Automatic laptop shutdown must run in Windows Python")
        import msvcrt
        with (RUN / "shutdown.lock").open("a+b") as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            monitor()


if __name__ == "__main__":
    main()
