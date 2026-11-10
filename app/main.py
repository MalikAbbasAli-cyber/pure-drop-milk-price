"""FastAPI service for PureDrop Dairy milk price prediction.

POST /predict takes the date to forecast and the most recent daily prices,
builds the same features used in training (notebooks/03), and returns the
predicted price from the saved model (models/best_model.pkl).
"""
import pickle
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
# day_num in training = days since the first record of the dataset
BASE_DATE = pd.Timestamp("2022-01-01")
MIN_HISTORY = 30  # the longest lag / rolling window used by the model

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]

app = FastAPI(
    title="PureDrop Milk Price API",
    description="Forecast the selling price of a milk delivery.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    date: Date = Field(..., description="Day to forecast (YYYY-MM-DD)")
    recent_prices: list[float] = Field(
        ...,
        description=(
            "Daily prices for the days BEFORE `date`, oldest first, ending on "
            f"the day immediately before `date`. At least {MIN_HISTORY} values."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2026-11-01",
                "recent_prices": [0.62 + 0.001 * (i % 5) for i in range(30)],
            }
        }
    }


class PredictResponse(BaseModel):
    date: Date
    predicted_price: float
    model: str


def build_features(target_date: Date, recent_prices: list[float]) -> pd.DataFrame:
    """Rebuild the training features from a date and its preceding prices.

    Mirrors notebooks/03_time_series_features.ipynb: lags and rolling means use
    only days before the forecast date.
    """
    p = np.asarray(recent_prices, dtype=float)
    ts = pd.Timestamp(target_date)
    doy = ts.dayofyear
    row = {
        "day_num": (ts - BASE_DATE).days,
        "month": ts.month,
        "dayofweek": ts.dayofweek,
        "dayofyear": doy,
        "sin_year": np.sin(2 * np.pi * doy / 365.25),
        "cos_year": np.cos(2 * np.pi * doy / 365.25),
        "lag_1": p[-1],
        "lag_7": p[-7],
        "lag_30": p[-30],
        "roll_mean_7": p[-7:].mean(),
        "roll_mean_30": p[-30:].mean(),
    }
    return pd.DataFrame([row])[FEATURES]


@app.get("/")
def root():
    return {"service": "PureDrop Milk Price API", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model": BUNDLE["model_name"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.recent_prices) < MIN_HISTORY:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {MIN_HISTORY} recent prices, got {len(req.recent_prices)}.",
        )
    if any(x <= 0 or not np.isfinite(x) for x in req.recent_prices):
        raise HTTPException(status_code=422, detail="Prices must be positive numbers.")
    X = build_features(req.date, req.recent_prices)
    pred = float(MODEL.predict(X)[0])
    return PredictResponse(date=req.date, predicted_price=round(pred, 4), model=BUNDLE["model_name"])
