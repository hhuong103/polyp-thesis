"""Finish an existing local run, verify it, then evaluate the frozen model.

Windows example (attach to the actual Python worker, not the venv launcher):
  python src/complete_training.py --wait-pid 12760
No hyperparameters are changed and test metrics are never used for selection.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "unetplusplus_resnet34_seed42"
RUN = ROOT / "results" / RUN_NAME
MAX_EPOCHS = 50
PATIENCE = 8


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(phase, **fields):
    state = {"phase": phase, "updated_at": utc(), "worker_pid": os.getpid(), **fields}
    history = RUN / "history.csv"
    if history.is_file():
        with history.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            state.update(epochs=int(rows[-1]["epoch"]), last_val_dice=float(rows[-1]["val_dice"]))
    save_json(RUN / "lifecycle.json", state)
    print(json.dumps(state, ensure_ascii=True), flush=True)


def audit():
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    expected = {"model": "unetplusplus", "encoder": "resnet34", "encoder_init": "imagenet",
                "image_size": 256, "batch_size": 4, "learning_rate": 1e-4, "patience": 8, "seed": 42}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Configuration changed: {key}")
    seen, counts = {}, {}
    dataset_digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "splits" / (split + ".csv")
        if sha256(path) != config["split_sha256"][split]:
            raise ValueError("Split changed: " + split)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        counts[split] = len(rows)
        for row in rows:
            paths = [(ROOT / row[key].replace("\\", "/")).resolve() for key in ("image", "mask")]
            if any(not p.is_relative_to((ROOT / "data").resolve()) or not p.is_file() for p in paths):
                raise ValueError("Invalid dataset path")
            digest = sha256(paths[0])
            for path, value in ((paths[0], digest), (paths[1], sha256(paths[1]))):
                dataset_digest.update((str(path.relative_to(ROOT)).replace("\\", "/") + ":" + value + "\n").encode())
            if digest in seen:
                raise ValueError(f"Duplicate image: {seen[digest]} / {split}")
            seen[digest] = split
    if counts != {"train": 700, "val": 150, "test": 150}:
        raise ValueError("Unexpected split sizes")
    return {"counts": counts, "split_sha256": config["split_sha256"], "duplicate_images": 0,
            "dataset_sha256": dataset_digest.hexdigest(),
            "patient_video_independence": "Not verified; source metadata not provided"}


def wait_for_windows_process(pid):
    """Hold a handle to the original process so PID reuse cannot change the target."""
    if os.name != "nt":
        raise RuntimeError("--wait-pid requires Windows")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # Process already exited.
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if Path(buffer.value).name.lower() not in ("python.exe", "pythonw.exe"):
            raise ValueError("Target is not a Python process")
        while True:
            result = kernel.WaitForSingleObject(handle, 30000)
            if result == 0:
                return
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            progress("training", trainer_pid=pid)
    finally:
        kernel.CloseHandle(handle)


def checkpoint():
    import torch
    # Trusted checkpoint produced by this project's own trainer, with RNG state.
    return torch.load(RUN / "last_checkpoint.pth", map_location="cpu", weights_only=False)


def terminal(state):
    return state["epoch"] >= MAX_EPOCHS or state["stale"] >= PATIENCE


def finish():
    import torch
    import numpy as np
    import segmentation_models_pytorch as smp
    audit_result = audit()
    preflight = json.loads((RUN / "preflight.json").read_text(encoding="utf-8"))
    if audit_result != preflight:
        raise ValueError("Dataset/config changed after completion supervision started")
    state = checkpoint()
    if not terminal(state):
        raise RuntimeError("Training has not reached max epochs or early stopping")
    history = state["history"]
    if len(history) != state["epoch"] or [r["epoch"] for r in history] != list(range(1, state["epoch"] + 1)):
        raise ValueError("Incomplete epoch history")
    if not all(np.isfinite(r[key]) for r in history for key in ("train_loss", "val_loss", "val_dice", "val_iou")):
        raise ValueError("Non-finite training metrics")
    if state["config"] != json.loads((RUN / "config.json").read_text(encoding="utf-8")):
        raise ValueError("Checkpoint config mismatch")
    best, best_epoch, stale = -1.0, 0, 0
    for row in history:
        if row["val_dice"] > best + 1e-4:
            best, best_epoch, stale = row["val_dice"], row["epoch"], 0
        else:
            stale += 1
    if abs(best - state["best_score"]) > 1e-8 or stale != state["stale"]:
        raise ValueError("Checkpoint selection inconsistent with history")
    with (RUN / "history.csv").open(newline="", encoding="utf-8-sig") as stream:
        csv_history = list(csv.DictReader(stream))
    if len(csv_history) != len(history) or any(float(a[k]) != float(b[k]) for a, b in zip(csv_history, history) for k in a):
        raise ValueError("CSV history differs from checkpoint")
    weights = torch.load(RUN / "best_weights.pth", map_location="cpu", weights_only=True)
    network = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    network.load_state_dict(weights, strict=True)
    if not all(bool(torch.isfinite(v).all()) for v in weights.values()):
        raise ValueError("Non-finite best weights")
    network.eval()
    with torch.inference_mode():
        prediction = network(torch.zeros(1, 3, 256, 256))
    if tuple(prediction.shape) != (1, 1, 256, 256) or not bool(torch.isfinite(prediction).all()):
        raise ValueError("Best checkpoint cannot perform inference")
    # Recompute validation scores from best weights, independently of saved metrics.
    sys.path.insert(0, str(ROOT))
    from src.datasets import PolypDataset
    from torch.utils.data import DataLoader
    validation = PolypDataset(ROOT / "data/splits/val.csv", image_size=256)
    loader = DataLoader(validation, batch_size=4, shuffle=False, num_workers=0)
    dice_values, iou_values = [], []
    with torch.inference_mode():
        for images, masks in loader:
            predicted = (torch.sigmoid(network(images)) >= .5).numpy()
            actual = (masks >= .5).numpy()
            for pred, gold in zip(predicted, actual):
                intersection = int(np.count_nonzero(pred & gold))
                denominator = int(pred.sum() + gold.sum())
                union = int(np.count_nonzero(pred | gold))
                dice_values.append(2 * intersection / denominator if denominator else 1.)
                iou_values.append(intersection / union if union else 1.)
    verified_dice, verified_iou = float(np.mean(dice_values)), float(np.mean(iou_values))
    if abs(verified_dice - best) > 1e-4 or abs(verified_iou - history[best_epoch - 1]["val_iou"]) > 1e-4:
        raise ValueError("Best weights do not reproduce recorded validation metrics")
    summary = {"status": "complete", "finished_at": utc(), "epochs": state["epoch"],
               "max_epochs": MAX_EPOCHS, "stop_reason": "early_stopping" if state["stale"] >= PATIENCE else "max_epochs",
               "best_epoch": best_epoch, "best_val_dice": best,
               "best_val_iou": history[best_epoch - 1]["val_iou"], "augmentation": False,
               "verified_val_dice": verified_dice, "verified_val_iou": verified_iou,
               "best_weights_sha256": sha256(RUN / "best_weights.pth"), "audit": audit_result,
               "validation_resolution": [256, 256], "threshold": 0.5,
               "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version},
               "weights_stamp": [(RUN / "best_weights.pth").stat().st_size, (RUN / "best_weights.pth").stat().st_mtime_ns]}
    save_json(RUN / "completion.json", summary)
    # Release verification network before loading both deployed models for evaluation.
    del network, weights, state
    sys.path.insert(0, str(ROOT))
    from backend import comparison
    frozen = [e for e in comparison.REGISTRY if e["id"] in ("final-unet", "unetplusplus")]
    if len(frozen) != 2:
        raise ValueError("Website registry does not contain both final models")
    comparison.REGISTRY = frozen  # Only affects this evaluator process, not the website.
    progress("evaluating", best_epoch=best_epoch)
    comparison.evaluate()
    if comparison.JOB["status"] != "complete":
        raise RuntimeError(comparison.JOB["error"] or "Evaluation failed")
    report = json.loads(comparison.REPORT.read_text(encoding="utf-8"))
    if sha256(RUN / "best_weights.pth") != summary["best_weights_sha256"]:
        raise ValueError("Best weights changed during final test evaluation")
    save_json(RUN / "test_evaluation.json", report)
    (RUN / "test_evaluation.csv").write_text(comparison.export_evaluation().body.decode("utf-8"), encoding="utf-8")
    summary["test_results"] = [{key: m[key] for key in ("id", "count", "dice", "iou", "inference_ms")} for m in report["models"]]
    save_json(RUN / "completion.json", summary)
    progress("complete", summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    # Retain a Windows file lock for the entire lifecycle; no competing supervisor.
    lock = (RUN / "completion.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        preflight = audit()
        saved_audit = RUN / "preflight.json"
        if saved_audit.is_file():
            if json.loads(saved_audit.read_text(encoding="utf-8")) != preflight:
                raise ValueError("Dataset changed since previous supervision")
        else:
            save_json(saved_audit, preflight)
        if args.wait_pid:
            progress("training", trainer_pid=args.wait_pid)
            wait_for_windows_process(args.wait_pid)
        state = checkpoint()
        needs_resume = not terminal(state)
        del state
        if needs_resume:
            progress("resuming")
            command = [sys.executable, "-u", str(ROOT / "src/train.py"), "--model", "unetplusplus",
                       "--epochs", str(MAX_EPOCHS), "--batch-size", "4", "--lr", "0.0001", "--patience", str(PATIENCE),
                       "--seed", "42", "--image-size", "256", "--run-name", RUN_NAME, "--resume"]
            with (RUN / "training_resume.log").open("a", encoding="utf-8") as log:
                trainer = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while trainer.poll() is None:
                    progress("training", trainer_pid=trainer.pid)
                    time.sleep(30)
                if trainer.returncode:
                    raise RuntimeError(f"Trainer exited {trainer.returncode}; see training_resume.log")
        progress("verifying")
        finish()
    except Exception as exc:
        progress("error", error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
