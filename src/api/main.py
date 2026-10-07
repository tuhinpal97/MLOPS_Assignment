from __future__ import annotations

import json
import logging
from pathlib import Path
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import HeartFeatures, PredictionResponse

MODEL_PATH = Path("artifacts/model/model.joblib")
META_PATH = Path("artifacts/model/metadata.json")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart-api")

REQUESTS = Counter("heart_api_requests_total", "API requests", ["method", "path", "status"])
PREDICTIONS = Counter("heart_predictions_total", "Predictions", ["class"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Prediction latency")

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
model = None
metadata = {"model_name": "unknown"}

@app.on_event("startup")
def startup() -> None:
    global model, metadata
    if not MODEL_PATH.exists():
        raise RuntimeError("Model artifact missing. Run: python -m src.train")
    model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text())

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        status = locals().get("status", 500)
        REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        logger.info(json.dumps({
            "event": "api_request", "request_id": request_id,
            "method": request.method, "path": request.url.path,
            "status": status, "latency_ms": round(duration * 1000, 2)
        }))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([payload.model_dump()])
    pred = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0, pred])
    PREDICTIONS.labels(str(pred)).inc()
    # Deliberately avoid logging raw patient feature values.
    logger.info(json.dumps({"event": "prediction", "class": pred, "confidence": round(confidence, 4)}))
    return PredictionResponse(
        prediction=pred,
        label="heart_disease_risk" if pred == 1 else "no_heart_disease_risk",
        confidence=round(confidence, 6),
        model_name=metadata.get("model_name", "unknown"),
    )
