"""Local comparison and held-out evaluation; metrics at original image resolution."""
import base64
import csv
import hashlib
import io
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "models/registry.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = {}
LOCK = threading.RLock()
JOB_LOCK = threading.Lock()
JOB = {"status": "idle", "completed": 0, "total": 0, "error": None}
REPORT = ROOT / "results/web_evaluation/report.json"
router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
MODEL_ALIASES = {"unet": "final-unet", "unet++": "unetplusplus"}


def config(model_id):
    model_id = MODEL_ALIASES.get(model_id, model_id)
    for entry in REGISTRY:
        if entry["id"] == model_id:
            return entry
    raise HTTPException(400, "Unknown model")


def png(array):
    ok, data = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(data).decode("ascii")


def decode_upload(upload, mask=False):
    try:
        raw = upload.file.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Mỗi file tối đa 10 MB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise HTTPException(400, "Chỉ hỗ trợ JPEG và PNG")
            if image.width * image.height > 16_000_000:
                raise HTTPException(413, "Ảnh tối đa 16 triệu pixel")
            if mask:
                arr = np.array(image.convert("L"))
                # JPEG masks have compression fringes; use the dataset threshold.
                return arr > (0 if arr.max() <= 1 else 127)
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "File ảnh rỗng, hỏng hoặc không hợp lệ") from exc
    finally:
        upload.file.close()


def scores(pred, gold):
    tp = int(np.count_nonzero(pred & gold))
    fp = int(np.count_nonzero(pred & ~gold))
    fn = int(np.count_nonzero(~pred & gold))
    return {"dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "false_positive_pixels": fp, "false_negative_pixels": fn}


def load_network(entry):
    """Shared cache for single prediction, comparison, and test evaluation."""
    with LOCK:
        path = ROOT / entry["weights"]
        if not path.is_file():
            CACHE.pop(entry["id"], None)
            raise HTTPException(503, "Chưa có checkpoint: " + entry["name"])
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        cached = CACHE.get(entry["id"])
        if cached is None or cached[0] != stamp:
            try:
                cls = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}[entry["architecture"]]
                net = cls(encoder_name=entry["encoder"], encoder_weights=None, in_channels=3, classes=1)
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                CACHE[entry["id"]] = (stamp, net.to(DEVICE).eval())
            except Exception as exc:
                CACHE.pop(entry["id"], None)
                logger.exception("Cannot load checkpoint for %s", entry["id"])
                raise HTTPException(503, "Không nạp được checkpoint của " + entry["name"]) from exc
        return CACHE[entry["id"]][1]


def infer(entry, image):
    start = time.perf_counter()
    with LOCK:
        net = load_network(entry)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (entry["image_size"], entry["image_size"])).astype(np.float32) / 255
        rgb = (rgb - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            pred = (torch.sigmoid(net(tensor))[0, 0] >= .5).cpu().numpy().astype(np.uint8)
        inference_ms = (time.perf_counter() - tick) * 1000
        mask = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, round(inference_ms, 2), round((time.perf_counter() - start) * 1000, 2)


def result(entry, image, gold=None):
    pred, ms, total = infer(entry, image)
    overlay = image.copy()
    overlay[pred] = (.6 * image[pred] + .4 * np.array([0, 0, 255])).astype(np.uint8)
    item = {"id": entry["id"], "name": entry["name"], "inference_ms": ms, "processing_ms": total,
            "mask": png(pred.astype(np.uint8) * 255), "overlay": png(overlay), "metrics": None,
            "architecture": entry["architecture"], "checkpoint": entry["weights"], "threshold": .5,
            "foreground_pixels": int(pred.sum()), "foreground_percent": round(float(pred.mean()) * 100, 3),
            "empty_mask": not bool(pred.any())}
    if gold is not None:
        item["metrics"] = scores(pred, gold)
        errors = np.zeros_like(image)
        errors[pred & gold] = (80, 190, 40)
        errors[pred & ~gold] = (50, 70, 245)
        errors[~pred & gold] = (245, 160, 50)
        item["errors"] = png(errors)
    return item


@router.get("/models")
def models():
    return {"device": str(DEVICE), "models": [model_status(e) for e in REGISTRY]}


def model_status(entry):
    available = (ROOT / entry["weights"]).is_file()
    progress = None
    if entry.get("training"):
        progress = {"state": "awaiting_checkpoint", "epochs": 0,
                    "max_epochs": entry["training"]["max_epochs"], "best_val_dice": None,
                    "last_val_dice": None, "error": None}
        path = ROOT / entry["history"]
        rows = []
        if path.is_file():
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                best, stale = -1.0, 0
                for row in rows:
                    dice = float(row["val_dice"])
                    if not np.isfinite(dice):
                        raise ValueError("Invalid Dice")
                    if dice > best + 1e-4:
                        best, stale = dice, 0
                    else:
                        stale += 1
                if rows:
                    progress.update(epochs=int(rows[-1]["epoch"]), best_val_dice=best,
                                    last_val_dice=float(rows[-1]["val_dice"]))
                    checkpoint = path.parent / "last_checkpoint.pth"
                    saved = checkpoint.is_file() and checkpoint.stat().st_mtime_ns >= path.stat().st_mtime_ns
                    terminal = progress["epochs"] >= progress["max_epochs"] or stale >= entry["training"]["patience"]
                    if available and saved and terminal:
                        progress["state"] = "complete"
            except (OSError, ValueError, KeyError):
                progress["error"] = "Chưa đọc được lịch sử huấn luyện hợp lệ"
        if available and progress["state"] != "complete":
            progress["state"] = "checkpoint_available"
        lifecycle = path.parent / "lifecycle.json"
        if lifecycle.is_file():
            try:
                worker = json.loads(lifecycle.read_text(encoding="utf-8"))
                progress["phase"] = worker.get("phase")
                progress["updated_at"] = worker.get("updated_at")
                if worker.get("phase") == "error":
                    progress["error"] = worker.get("error", "Quy trình huấn luyện gặp lỗi")
                    progress["state"] = "checkpoint_available" if available else "awaiting_checkpoint"
            except (OSError, ValueError):
                pass
    return {**entry, "available": available, "progress": progress,
            "ready_for_evaluation": entry.get("evaluation_enabled", True) and available and (progress is None or progress["state"] == "complete")}


def evaluation_entries():
    return [entry for entry in REGISTRY if entry.get("evaluation_enabled", True) and model_status(entry)["ready_for_evaluation"]]


def segment_upload(file, model_ids, ground_truth=None):
    started = time.perf_counter()
    try:
        entries = [config(model_id) for model_id in model_ids]
        if len({e["id"] for e in entries}) != len(entries):
            raise HTTPException(400, "Hãy chọn hai mô hình khác nhau")
        for entry in entries:
            if not (ROOT / entry["weights"]).is_file():
                raise HTTPException(503, "Chưa có checkpoint: " + entry["name"])
        image = decode_upload(file)
        gold = decode_upload(ground_truth, mask=True) if ground_truth else None
        if gold is not None and gold.shape != image.shape[:2]:
            raise HTTPException(400, "Mask chuẩn phải cùng kích thước ảnh gốc; không tự co giãn mask để chấm điểm")
        output = [result(entry, image, gold) for entry in entries]
        return {"original": png(image), "ground_truth": png(gold.astype(np.uint8)*255) if gold is not None else None,
                "device": str(DEVICE), "image_width": image.shape[1], "image_height": image.shape[0],
                "results": output, "total_ms": round((time.perf_counter()-started)*1000, 2)}
    finally:
        file.file.close()
        if ground_truth is not None:
            ground_truth.file.close()


@router.post("/predict")
def predict(file: UploadFile = File(...), model: str = Form("unet"),
            ground_truth: UploadFile | None = File(None)):
    """model: unet (or final-unet), unetplusplus (or unet++)."""
    response = segment_upload(file, [model], ground_truth)
    prediction = response["results"][0]
    # Preserve the original /predict fields for existing API clients.
    response.update(model=prediction["name"], model_id=prediction["id"],
                    inference_time_ms=prediction["inference_ms"],
                    mask_base64=prediction["mask"], overlay_base64=prediction["overlay"])
    return response


@router.post("/compare")
def compare(file: UploadFile = File(...), ground_truth: UploadFile | None = File(None),
            model_a: str = Form("unet"), model_b: str = Form("unetplusplus")):
    """Decode once, run both selected checkpoints on exactly the same image."""
    return segment_upload(file, [model_a, model_b], ground_truth)


@router.get("/history")
def history():
    output = []
    for entry in REGISTRY:
        path = ROOT / entry["history"]
        rows, error = [], None
        if path.is_file():
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        values = {key: float(row[key]) for key in ("epoch", "train_loss", "val_loss", "val_dice", "val_iou")}
                        if not all(np.isfinite(v) for v in values.values()):
                            raise ValueError("Non-finite history")
                        rows.append(values)
            except (ValueError, KeyError):
                error = "CSV không đúng định dạng epoch, train_loss, val_loss, val_dice, val_iou"
                rows = []
        output.append({"id": entry["id"], "name": entry["name"], "path": entry["history"], "rows": rows, "error": error})
    return output


def split_rows():
    with (ROOT / "data/splits/test.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Tập test rỗng")
    return rows


def dataset_path(value):
    path = (ROOT / value.replace("\\", "/")).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Đường dẫn dataset không hợp lệ")
    return path


def pair(row):
    image = cv2.imread(str(dataset_path(row["image"])))
    mask = cv2.imread(str(dataset_path(row["mask"])), 0)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError("Ảnh/mask test thiếu hoặc khác kích thước")
    return image, mask > 127


def fingerprint(entries=None):
    if entries is None:
        entries = evaluation_entries()
    paths = [ROOT / "models/registry.json", ROOT / "data/splits/test.csv"] + [ROOT / e["weights"] for e in entries]
    digests = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        digests[str(path.relative_to(ROOT))] = digest.hexdigest()
    return digests


def source_stamps(entries=None):
    if entries is None:
        entries = evaluation_entries()
    paths = [ROOT / "models/registry.json", ROOT / "data/splits/test.csv"] + [ROOT / e["weights"] for e in entries]
    paths += [dataset_path(row[key]) for row in split_rows() for key in ("image", "mask")]
    return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def evaluate():
    try:
        entries = evaluation_entries()
        if not entries:
            raise ValueError("Chưa có checkpoint sẵn sàng đánh giá")
        rows = split_rows()
        provenance = fingerprint(entries)
        stamps = source_stamps(entries)
        with JOB_LOCK:
            JOB["total"] = len(rows) * len(entries)
        output = []
        for entry in entries:
            samples = []
            for index, row in enumerate(rows):
                image, gold = pair(row)
                pred, ms, _ = infer(entry, image)
                samples.append({"index": index, "image": row["image"], **scores(pred, gold), "inference_ms": ms})
                with JOB_LOCK:
                    JOB["completed"] += 1
            output.append({"id": entry["id"], "name": entry["name"], "note": entry["note"], "count": len(samples),
                           "dice": float(np.mean([s["dice"] for s in samples])),
                           "iou": float(np.mean([s["iou"] for s in samples])),
                           "inference_ms": float(np.mean([s["inference_ms"] for s in samples])),
                           "samples": sorted(samples, key=lambda s: s["dice"])})
        if provenance != fingerprint(entries) or stamps != source_stamps(entries):
            raise ValueError("Checkpoint hoặc split đã thay đổi trong lúc chạy; hãy chạy lại")
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "fingerprints": provenance, "source_stamps": stamps, "models": output,
                  "method": "Macro mean per image; original resolution; threshold 0.5; empty/empty = 1; test.csv. Dataset independence not verified."}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        tmp = REPORT.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        tmp.replace(REPORT)
        with JOB_LOCK:
            JOB["status"] = "complete"
    except Exception as exc:
        with JOB_LOCK:
            JOB.update(status="error", error=str(exc))


@router.post("/evaluation")
def start_evaluation():
    with JOB_LOCK:
        if JOB["status"] == "running":
            raise HTTPException(409, "Đang đánh giá, vui lòng đợi")
        JOB.update(status="running", completed=0, total=0, error=None)
        threading.Thread(target=evaluate, daemon=True).start()
        return dict(JOB)


@router.get("/evaluation")
def evaluation():
    with JOB_LOCK:
        job = dict(JOB)
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else None
    if report:
        try:
            report["stale"] = report.get("source_stamps") != source_stamps()
        except (OSError, ValueError):
            report["stale"] = True
    return {"job": job, "report": report}


@router.get("/evaluation/case/{model_id}/{index}")
def case(model_id: str, index: int):
    entry = config(model_id)
    rows = split_rows()
    if not 0 <= index < len(rows):
        raise HTTPException(404, "Không có ảnh test này")
    image, gold = pair(rows[index])
    return {"original": png(image), "ground_truth": png(gold.astype(np.uint8)*255),
            "results": [result(entry, image, gold)], "total_ms": None}


@router.get("/evaluation.csv")
def export_evaluation():
    if not REPORT.is_file():
        raise HTTPException(404, "Chưa có báo cáo kiểm thử")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["model", "image", "dice", "iou", "false_positive_pixels", "false_negative_pixels", "inference_ms"])
    for entry in report["models"]:
        for sample in entry["samples"]:
            writer.writerow([entry["id"]] + [sample[key] for key in ("image", "dice", "iou", "false_positive_pixels", "false_negative_pixels", "inference_ms")])
    return Response(stream.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=test-evaluation.csv"})
