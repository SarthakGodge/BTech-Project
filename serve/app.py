"""
FastAPI inference server (v2).

POST /predict - supply a full (T, F) window of RAW (unscaled) features; stateless.
POST /flow    - supply ONE flow; the server keeps a circular buffer and predicts
                once it has T flows.

IMPORTANT (train/serve consistency): the model is trained on windows of
consecutive flows in TIME order across ALL hosts (flows from different hosts
interleaved), because the CIC CSVs mostly have no usable source column.
GLOBAL_BUFFER=1 (default) therefore keeps ONE buffer for the whole feed so
serving sees the same kind of window. Set GLOBAL_BUFFER=0 to buffer per
`source` (only sensible if you also trained on per-host windows).

Artifacts are looked up in $IDS_ARTIFACT_DIR, then ./artifacts, then ../artifacts.
Run: uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import sys
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import List

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))              # common.py must be importable to unpickle scaler.pkl
sys.path.insert(0, str(HERE.parent))
import common  # noqa: F401,E402  (signed_log1p lives here)


def find_artifacts() -> Path:
    cands = [os.environ.get("IDS_ARTIFACT_DIR"), HERE / "artifacts",
             HERE.parent / "artifacts", Path.cwd() / "artifacts"]
    for c in cands:
        if c and (Path(c) / "selected_features.json").exists():
            return Path(c)
    raise FileNotFoundError("artifacts/ not found - set IDS_ARTIFACT_DIR")


ART = find_artifacts()
CLASS_NAMES = ["No Attack", "Pre-Attack", "Attack"]
ACTIONS = {0: "log", 1: "alert", 2: "block"}
GLOBAL_BUFFER = os.environ.get("GLOBAL_BUFFER", "1") != "0"
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))

app = FastAPI(title="Hybrid CNN-LSTM IDS", version="2.0")

FEATURES: List[str] = json.loads((ART / "selected_features.json").read_text())
SCALER = joblib.load(ART / "scaler.pkl")
MODEL = tf.saved_model.load(str(ART / os.environ.get("IDS_SAVED_MODEL", "saved_model")))
INFER = MODEL.signatures["serving_default"]
OUT_KEY = list(INFER.structured_outputs.keys())[0]
WINDOW = int(INFER.inputs[0].shape[1])
BUFFERS: dict = defaultdict(lambda: deque(maxlen=WINDOW))

# Class-probability scales tuned on the validation set by 05_train.py (optional).
_th = ART / os.environ.get("IDS_THRESHOLDS", "thresholds.json")
CLASS_SCALE = (np.array(json.loads(_th.read_text())["class_scale"], dtype=np.float32)
               if _th.exists() else np.ones(3, dtype=np.float32))


class WindowRequest(BaseModel):
    features: List[List[float]] = Field(
        ..., description=f"Shape ({WINDOW}, {len(FEATURES)}), unscaled, order = GET /features")


class FlowRequest(BaseModel):
    source: str = Field("global", description="Source IP / flow key (used only if GLOBAL_BUFFER=0)")
    features: List[float]


def run_model(window_raw: np.ndarray) -> dict:
    t0 = time.perf_counter()
    x = SCALER.transform(window_raw.astype(np.float32))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)[None, ...].astype(np.float32)
    proba = INFER(tf.constant(x))[OUT_KEY].numpy()[0]
    scaled = proba * CLASS_SCALE
    cls = int(scaled.argmax())
    conf = float(proba[cls])
    action = ACTIONS[cls] if conf >= CONFIDENCE_THRESHOLD else "log"
    return {
        "class_id": cls, "class_name": CLASS_NAMES[cls], "confidence": round(conf, 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, proba)},
        "action": action, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "window": WINDOW, "n_features": len(FEATURES),
            "global_buffer": GLOBAL_BUFFER, "tracked_sources": len(BUFFERS),
            "class_scale": CLASS_SCALE.tolist()}


@app.get("/features")
def features():
    return {"order": FEATURES}


@app.post("/predict")
def predict(req: WindowRequest):
    arr = np.asarray(req.features, dtype=np.float32)
    if arr.shape != (WINDOW, len(FEATURES)):
        raise HTTPException(422, f"expected ({WINDOW}, {len(FEATURES)}), got {arr.shape}")
    return run_model(arr)


@app.post("/flow")
def flow(req: FlowRequest):
    if len(req.features) != len(FEATURES):
        raise HTTPException(422, f"expected {len(FEATURES)} features, got {len(req.features)}")
    key = "global" if GLOBAL_BUFFER else req.source
    buf = BUFFERS[key]
    buf.append(req.features)
    if len(buf) < WINDOW:
        return {"status": "buffering", "have": len(buf), "need": WINDOW}
    result = run_model(np.asarray(buf, dtype=np.float32))
    result.update(source=req.source, status="predicted")
    return result


@app.delete("/flow/{source}")
def clear(source: str):
    BUFFERS.pop("global" if GLOBAL_BUFFER else source, None)
    return {"cleared": source}
