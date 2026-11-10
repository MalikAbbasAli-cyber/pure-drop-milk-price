# PureDrop Dairy: Milk Price Prediction

Machine learning project for PureDrop Dairy: forecasting milk selling prices
end to end (data, model, FastAPI deployment).

## Status
**Task 5 of 6:** FastAPI service and deployment. (Tasks 1-4 done)

| # | Task | Status |
|---|------|--------|
| 1 | Repo, data, initial notebook | Done |
| 2 | Baseline linear regression | Done |
| 3 | Time-series feature engineering | Done |
| 4 | Compare advanced regressors | Done |
| 5 | FastAPI service + free deployment | Done (live URL below) |
| 6 | Model card and hand-over | Planned |

## Project structure
```
pure-drop-milk-price/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── milk_prices.csv        # clean copy: columns date, price
├── app/
│   └── main.py                # FastAPI service (/health, /predict)
├── tests/
│   └── test_api.py            # API tests
├── requirements-api.txt       # minimal deps for deployment
├── render.yaml                # Render deployment config
├── models/
│   ├── best_model.pkl         # selected model + feature list
│   └── model_comparison.csv   # CV and test scores for all models
├── scripts/
│   └── download_data.py       # fetches data and saves the CSV
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

The baseline beats a naive "predict the training mean" model on MAE (0.0265 vs
0.0527), but R2 is negative: a straight line cannot capture the yearly
seasonality in the data. Task 3 adds time-series features to fix this.

## Feature engineering results (Task 3)
Same linear regression, same chronological 80/20 split; only the features change.

| Feature set | MAE | RMSE | R2 |
|-------------|-----|------|----|
| Baseline (day_num) | 0.0267 | 0.0309 | -0.3161 |
| + Calendar | 0.0180 | 0.0214 | 0.3693 |
| + Seasonality (sin/cos) | 0.0081 | 0.0102 | 0.8570 |
| + Lags | 0.0081 | 0.0102 | 0.8568 |
| All features | 0.0082 | 0.0103 | 0.8541 |

Seasonality features gave almost all of the improvement. Lag and rolling
features did not help on this synthetic data (noise is independent day to
day); they should be re-tested on real data. Engineered data is saved to
`data/milk_features.csv` for Task 4.

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

**Live URL:** _add your Render URL here after deploying_

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
Malik Abbas Ali. Ezitech internship, Task 1.
