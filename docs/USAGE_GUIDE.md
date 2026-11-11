# Usage Guide

Live service: **https://pure-drop-milk-price.onrender.com**
Interactive docs (try it in the browser): https://pure-drop-milk-price.onrender.com/docs

> The free hosting plan sleeps when idle. The first request after a quiet spell can take 30-60 seconds; later requests are fast.

## 1. Endpoints
| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms the service is up and shows the model name |
| `POST /predict` | Forecast the price for one date |

## 2. Calling `/predict`
**Request body**
```json
{
  "date": "2026-11-12",
  "recent_prices": [0.61, 0.62, "... 30 or more daily prices ..."]
}
```
- `date`: the day to forecast, `YYYY-MM-DD`.
- `recent_prices`: daily prices for the days *before* `date`, **oldest first**, ending on the day immediately before `date`. At least 30 values are required (the model uses the price 30 days back and averages over the last 7 and 30 days).

**Response**
```json
{"date": "2026-11-12", "predicted_price": 0.6205, "model": "Ridge"}
```

**Errors:** fewer than 30 prices, a non-positive price, or a malformed date returns HTTP 422 with a message.

### Python example
```python
import pandas as pd, requests

prices = pd.read_csv("data/milk_prices.csv")["price"].tail(30).tolist()
r = requests.post(
    "https://pure-drop-milk-price.onrender.com/predict",
    json={"date": "2026-11-01", "recent_prices": prices},
    timeout=90,   # allow for the free plan waking up
)
print(r.json())
```
Or run `python scripts/example_request.py https://pure-drop-milk-price.onrender.com`.

## 3. Run the API on your own machine
```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs. Run the tests with `pip install pytest httpx` then `pytest -q`.

## 4. Swap in PureDrop's real data and retrain
1. Create `data/milk_prices.csv` with two columns: `date` (YYYY-MM-DD, one row per day) and `price`. Fill or remove missing days first: the lag features assume consecutive days.
2. If the first date is not 2022-01-01, update `BASE_DATE` in `app/main.py`.
3. Run `notebooks/03_time_series_features.ipynb` (creates `data/milk_features.csv`), then `notebooks/04_model_comparison.ipynb` (compares models and saves `models/best_model.pkl`).
4. Check the new scores in the notebook output, then run `pytest -q`.
5. Commit and push. Render redeploys automatically from the `main` branch.

Keep scikit-learn at the version pinned in `requirements-api.txt`, or retrain and redeploy together if you change it.

## 5. Troubleshooting
| Symptom | Likely cause |
|---|---|
| First request is very slow or times out | Free plan waking from sleep; wait a minute and retry |
| HTTP 422 | Fewer than 30 prices, a bad date, or a non-positive price |
| Model fails to load after a change | scikit-learn version differs from the one the model was saved with |
