from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clean_social.utils.paths import project_root

app = FastAPI(title="CleanSocial Sentiment API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for sentiment prediction")


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float


def _default_model_path() -> Path:
    return project_root() / "artifacts" / "models" / "deployment_model.joblib"


MODEL_PATH = _default_model_path()
MODEL = None


@app.on_event("startup")
def load_model() -> None:
    global MODEL
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    probs = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    best_index = int(probs.argmax())

    return PredictResponse(sentiment=str(classes[best_index]), confidence=float(probs[best_index]))
