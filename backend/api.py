from __future__ import annotations

import os
import threading
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/model"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
app = FastAPI(title="Social Sentiment API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
tokenizer = model = None
load_error = None
predictions = deque(maxlen=10000)
model_lock = threading.Lock()


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


def load_model():
    global tokenizer, model, load_error
    try:
        if not (MODEL_DIR / "config.json").exists():
            # First boot trains a reproducible small sample; set TRAIN_SAMPLE_SIZE=0 for all data.
            import subprocess
            subprocess.run(["python", "-m", "backend.train"], check=True, env=os.environ.copy())
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        load_error = None
    except Exception as exc:  # exposed through /health; API remains inspectable
        load_error = str(exc)


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_unavailable",
            "model_loaded": model is not None, "error": load_error}


def classify(texts: list[str]):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {load_error or 'loading'}")
    with model_lock, torch.inference_mode():
        inputs = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()
    results = []
    for text, row in zip(texts, probs):
        label_id = int(row.argmax())
        result = {"text": text, "sentiment": "positive" if label_id == 1 else "negative",
                  "confidence": round(float(row[label_id]), 4),
                  "probabilities": {"negative": round(float(row[0]), 4), "positive": round(float(row[1]), 4)},
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        predictions.append(result)
        results.append(result)
    return results


@app.post("/predict")
def predict(request: PredictionRequest):
    return classify([request.text])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="Texts cannot be blank")
    return {"predictions": classify(request.texts)}


@app.get("/api/trends")
def trends(hours: int = 24):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, min(hours, 168)))
    recent = [p for p in predictions if datetime.fromisoformat(p["timestamp"]) >= cutoff]
    counts = Counter(p["sentiment"] for p in recent)
    buckets: dict[str, Counter] = {}
    for item in recent:
        moment = datetime.fromisoformat(item["timestamp"]).replace(minute=0, second=0, microsecond=0)
        key = moment.isoformat()
        buckets.setdefault(key, Counter())[item["sentiment"]] += 1
    timeline = [{"hour": hour, "positive": count["positive"], "negative": count["negative"]}
                for hour, count in sorted(buckets.items())]
    return {"total": len(recent), "positive": counts["positive"], "negative": counts["negative"],
            "timeline": timeline, "recent": list(reversed(recent))[:50]}
