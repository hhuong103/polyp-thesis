"""FastAPI application for the trained U-Net and U-Net++ models."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.comparison import (
    CACHE, DEVICE, LOCK, REGISTRY, ROOT,
    load_network, model_status, predict, router,
)

logger = logging.getLogger(__name__)


def deployed_models():
    return [entry for entry in REGISTRY if entry["id"] in ("final-unet", "unetplusplus")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Each checkpoint is loaded once into the same cache used by all endpoints.
    for entry in deployed_models():
        try:
            load_network(entry)
            logger.info("Loaded %s", entry["name"])
        except HTTPException:
            logger.warning("%s is unavailable; other models remain usable", entry["name"])
    yield
    with LOCK:
        CACHE.clear()


app = FastAPI(
    title="Polyp Segmentation API",
    description="Segment with U-Net E3 or U-Net++ E4, compare both, and evaluate against ground truth with optional nearest-neighbor resizing.",
    version="3.1.0",
    lifespan=lifespan,
)
app.include_router(router)
# Backwards-compatible URL; shares all validation and checkpoint selection.
app.add_api_route("/predict", predict, methods=["POST"], tags=["Segmentation"])
app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "frontend/index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    states = []
    with LOCK:
        for entry in deployed_models():
            state = model_status(entry)
            path = ROOT / entry["weights"]
            cached = CACHE.get(entry["id"])
            try:
                stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            except OSError:
                stamp = None
            states.append({
                "id": entry["id"], "name": entry["name"],
                "available": state["available"],
                "loaded": bool(cached and cached[0] == stamp),
            })
    loaded = sum(state["loaded"] for state in states)
    return {
        "status": "ok" if loaded == len(states) and states else "degraded" if loaded else "unavailable",
        "model_loaded": bool(loaded),
        "model": "U-Net / U-Net++ + ResNet34",
        "device": str(DEVICE),
        "models": states,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8081,
                reload=True, reload_dirs=[str(ROOT / "backend")])
