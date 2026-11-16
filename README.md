# PureDrop Dairy: Milk Price Prediction

Machine learning project for PureDrop Dairy: forecasting milk selling prices
end to end (data, model, FastAPI deployment).

## Status
**All 6 tasks complete.** Live API: https://pure-drop-milk-price.onrender.com

| # | Task | Status |
|---|------|--------|
| 1 | Repo, data, initial notebook | Done |
| 2 | Baseline linear regression | Done |
| 3 | Time-series feature engineering | Done |
| 4 | Compare advanced regressors | Done |
| 5 | FastAPI service + free deployment | Done |
| 6 | Model card and hand-over | Done |

## Documentation
- [Model card](docs/MODEL_CARD.md): what the model is, how it was evaluated, and its limits
- [Usage guide](docs/USAGE_GUIDE.md): how to call the API, run locally, and retrain
- [Hand-over](docs/HANDOVER.md): what is delivered, what PureDrop must do next

## Project structure
```
pure-drop-milk-price/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── milk_prices.csv        # clean copy: columns date, price
│   └── milk_prices_fe.csv     # Task 3: with lag_1, ma_7, day_of_week, month
├── app/
│   └── main.py                # FastAPI service (/health, /predict)
├── tests/
│   └── test_api.py            # API tests
├── requirements-api.txt       # minimal deps for deployment
├── render.yaml                # Render deployment config
├── models/
│   ├── model_baseline.pkl     # Task 2 baseline (linear regression)
│   ├── model_fe.pkl           # Task 3 feature-engineered model
│   ├── best_model.pkl         # selected model + feature list
│   └── model_comparison.csv   # CV and test scores for all models
├── scripts/
│   ├── download_data.py       # fetches data and saves the CSV
│   ├── train_baseline.py      # trains and saves the baseline, prints RMSE
│   ├── feature_engineering.py # adds features, trains model_fe.pkl, compares RMSE
│   └── example_request.py     # sample API client
└── notebooks/
    ├── 01_eda.ipynb           # summary stats + price-over-time plots
    ├── 02_baseline_model.ipynb  # baseline linear regression + evaluation
    ├── 03_time_series_features.ipynb  # calendar, seasonality, lag, rolling features
    └── 04_model_comparison.ipynb      # compare regressors, save best model
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_data.py
jupyter notebook notebooks/01_eda.ipynb
```

## What the notebook shows
- Shape, column types and missing values
- Summary statistics (count, mean, std, min, quartiles, max, skewness) and date range
- Line chart of daily price over time, plus a monthly-average chart

## Baseline model results (Task 2)
Chronological 80/20 split (no shuffling), feature: days since start.

| Metric | Value |
|--------|-------|
| MAE | 0.0265 |
| RMSE | 0.0307 |
| R2 | -0.2796 |

**Test RMSE: 0.0307**

**Reproduce it:** `python scripts/train_baseline.py` loads the cleaned CSV, splits it 80/20
with `train_test_split`, trains a `LinearRegression`, prints the scores above, and saves
the model with joblib to `models/model_baseline.pkl`. The split uses `shuffle=False`
(oldest 80% of days train, newest 20% test) with a fixed `random_state=42`, so it is
identical on every run. It is not shuffled because this is time-series data: a random
split would let the model learn from future prices.

The baseline beats a naive "predict the training mean" model on MAE (0.0265 vs
0.0527), but R2 is negative: a straight line cannot capture the yearly
seasonality in the data. Task 3 adds time-series features to fix this.

## Feature engineering results (Task 3)
`python scripts/feature_engineering.py` adds four columns to the data, trains a new
`LinearRegression` and saves it with joblib to `models/model_fe.pkl`.

| New column | Meaning |
|------------|---------|
| `lag_1` | price on the previous day |
| `ma_7` | average price of the previous 7 days |
| `day_of_week` | 0 = Monday ... 6 = Sunday |
| `month` | month number 1-12 |

`ma_7` is computed from the *previous* 7 days only, so today's price is never used to
predict today's price. The first 7 rows have no history and are dropped (1765 -> 1758
rows). The enriched data is saved to `data/milk_prices_fe.csv`. Same 80/20
`train_test_split` as Task 2 (`shuffle=False`, `random_state=42`); both models are scored
on the **same** 352 test rows.

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| Baseline (`day_num` only) | 0.0265 | 0.0308 | -0.2962 |
| **Feature model (`model_fe.pkl`)** | **0.0088** | **0.0111** | **0.8303** |

**Test RMSE fell from 0.0308 to 0.0111 (about 64% lower).** (Task 2 reported 0.0307
because dropping the first 7 rows changes the test set by one day.) The biggest gain
comes from `ma_7`, which gives the model the recent price level, so it can follow the
yearly seasonal wave that a straight line cannot. Scores are one-step-ahead forecasts
(real previous prices are used as inputs) on synthetic data.

**Further exploration** (`notebooks/03_time_series_features.ipynb`) tried other feature
sets on the same data: adding smooth seasonality terms (sin/cos of the day of year) gave
an RMSE of about 0.0102, and extra lag and rolling features did not improve on that. The
enriched file from that notebook (`data/milk_features.csv`) is used in Task 4.

## Model comparison (Task 4)
Selection by 5-fold `TimeSeriesSplit` cross-validation on the training period only;
test scores reported once afterwards.

| Model | CV RMSE | Test MAE | Test RMSE | Test R2 |
|-------|---------|----------|-----------|---------|
| **Ridge (selected)** | 0.0136 | 0.0081 | 0.0103 | 0.8552 |
| Random Forest | 0.0177 | 0.0127 | 0.0158 | 0.6575 |
| Gradient Boosting | 0.0190 | 0.0135 | 0.0168 | 0.6139 |
| Linear Regression | 0.0201 | 0.0082 | 0.0103 | 0.8541 |
| HistGradientBoosting | 0.0226 | 0.0128 | 0.0160 | 0.6474 |

Ridge was selected on cross-validation; on the test period it is essentially tied
with plain Linear Regression. Tree-based models did worse because they cannot
extrapolate the upward price trend beyond the range seen in training. Results
come from synthetic data and should be re-checked on real data.

## API (Task 5)

Run locally:
```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for the interactive docs.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service and model status |
| `POST /predict` | Forecast the price for one date |

`POST /predict` body:
```json
{
  "date": "2026-11-01",
  "recent_prices": [0.62, 0.63, "... at least 30 daily prices, oldest first, ending the day before `date`"]
}
```
Response:
```json
{"date": "2026-11-01", "predicted_price": 0.6205, "model": "Ridge"}
```
The model uses lag and rolling-average features, so the caller supplies the
last 30 daily prices; the service builds the same features as the training
notebook. Fewer than 30 prices returns HTTP 422.

Example client: `python scripts/example_request.py [base_url]`. Tests: `pytest -q`.

**Live URL:** https://pure-drop-milk-price.onrender.com (interactive docs at `/docs`)

The free Render plan puts the service to sleep after a period of inactivity, so the
first request after a quiet spell can take 30-60 seconds.

## Data note (important)
The UCI "Milk Production" page linked in the brief is an HTML page, not a
downloadable CSV, and the well-known dataset of that name holds monthly milk
*production*, not price. So `scripts/download_data.py`:

1. tries to download the real data, and
2. if that fails, generates a **synthetic daily price series**
   (trend + yearly seasonality + noise, fixed seed) for 2022-01-01 to 2026-10-31.

**The data in `data/milk_prices.csv` is synthetic.** It is a stand-in so the
pipeline runs end to end, and can be replaced with PureDrop's real sales
history (same two columns: `date`, `price`) without changing the notebook.

## Author
Malik Abbas Ali. Ezitech internship, PureDrop Dairy project (Tasks 1-6).
