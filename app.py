"""
FastAPI inference server.

Two endpoints:
  POST /predict  - you supply a full (T, F) window; stateless.
  POST /flow     - you supply ONE flow; the server keeps a per-source
                   circular buffer and predicts once it has T flows.

/flow is what CICFlowMeter actually feeds in production: flows arrive one
at a time, and the sequence context has to be reconstructed server-side.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ART = Path(__file__).parent.parent / "artifacts"
CLASS_NAMES = ["No Attack", "Pre-Attack", "Attack"]
ACTIONS = {0: "log", 1: "alert", 2: "block"}

app = FastAPI(title="Hybrid CNN-LSTM IDS", version="1.0")

FEATURES: List[str] = json.loads((ART / "selected_features.json").read_text())
SCALER = joblib.load(ART / "scaler.pkl")
MODEL = tf.saved_model.load(str(ART / "saved_model"))
INFER = MODEL.signatures["serving_default"]
OUT_KEY = list(INFER.structured_outputs.keys())[0]

WINDOW = int(INFER.inputs[0].shape[1])
BUFFERS: dict = defaultdict(lambda: deque(maxlen=WINDOW))

# Below this confidence we do not act - the flow is logged for review.
CONFIDENCE_THRESHOLD = 0.60


class WindowRequest(BaseModel):
    features: List[List[float]] = Field(
        ..., description=f"Shape ({WINDOW}, {len(FEATURES)}), unscaled")


class FlowRequest(BaseModel):
    source: str = Field(..., description="Source IP or flow key")
    features: List[float]


def run_model(window_raw: np.ndarray) -> dict:
    """window_raw: (T, F) unscaled -> prediction dict."""
    t0 = time.perf_counter()
    x = SCALER.transform(window_raw.astype(np.float32))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x[None, ...].astype(np.float32)

    proba = INFER(tf.constant(x))[OUT_KEY].numpy()[0]
    cls = int(proba.argmax())
    conf = float(proba[cls])
    action = ACTIONS[cls] if conf >= CONFIDENCE_THRESHOLD else "log"

    return {
        "class_id": cls,
        "class_name": CLASS_NAMES[cls],
        "confidence": round(conf, 4),
        "probabilities": {n: round(float(p), 4)
                          for n, p in zip(CLASS_NAMES, proba)},
        "action": action,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "window": WINDOW, "n_features": len(FEATURES),
            "tracked_sources": len(BUFFERS)}


@app.get("/features")
def features():
    return {"order": FEATURES}


@app.post("/predict")
def predict(req: WindowRequest):
    arr = np.asarray(req.features, dtype=np.float32)
    if arr.shape != (WINDOW, len(FEATURES)):
        raise HTTPException(422, f"expected ({WINDOW}, {len(FEATURES)}), "
                                 f"got {arr.shape}")
    return run_model(arr)


@app.post("/flow")
def flow(req: FlowRequest):
    if len(req.features) != len(FEATURES):
        raise HTTPException(422, f"expected {len(FEATURES)} features, "
                                 f"got {len(req.features)}")
    buf = BUFFERS[req.source]
    buf.append(req.features)

    if len(buf) < WINDOW:
        return {"status": "buffering", "have": len(buf), "need": WINDOW}

    result = run_model(np.asarray(buf, dtype=np.float32))
    result["source"] = req.source
    result["status"] = "predicted"
    return result


@app.delete("/flow/{source}")
def clear(source: str):
    BUFFERS.pop(source, None)
    return {"cleared": source}
