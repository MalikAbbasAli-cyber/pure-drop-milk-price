"""Run with: pytest -q  (from the project root)"""
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, MODEL, FEATURES

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_matches_training_features():
    """The API must rebuild exactly the features the notebook produced."""
    df = pd.read_csv("data/milk_features.csv", parse_dates=["date"])
    prices = pd.read_csv("data/milk_prices.csv", parse_dates=["date"])
    row = df.iloc[500]
    hist = prices[prices["date"] < row["date"]]["price"].tolist()[-30:]
    r = client.post("/predict", json={"date": str(row["date"].date()), "recent_prices": hist})
    assert r.status_code == 200
    expected = float(MODEL.predict(row[FEATURES].to_frame().T.astype(float))[0])
    assert abs(r.json()["predicted_price"] - expected) < 1e-4


def test_too_few_prices():
    r = client.post("/predict", json={"date": "2026-11-01", "recent_prices": [0.6] * 10})
    assert r.status_code == 422


def test_bad_date():
    r = client.post("/predict", json={"date": "not-a-date", "recent_prices": [0.6] * 30})
    assert r.status_code == 422
