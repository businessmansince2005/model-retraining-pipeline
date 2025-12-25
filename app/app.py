from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import joblib

app = FastAPI(title="Model Retraining Pipeline")

class Features(BaseModel):
    feature1: float
    feature2: float

def load_latest_model():
    model_path = os.getenv("MODEL_PATH", "models/latest/model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_latest_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: Features):
    if model is None:
        return {"error": "No model loaded yet"}
    df = pd.DataFrame([payload.dict()])
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(df)[0][1]
    else:
        pred = model.predict(df)[0]
    return {"prediction": float(pred)}