# PureDrop Dairy: Milk Price Prediction

Machine learning project for PureDrop Dairy: forecasting milk selling prices
end to end (data, model, FastAPI deployment).

## Status
**Task 1 of 6:** repo setup, data acquisition and exploratory analysis. (done)

| # | Task | Status |
|---|------|--------|
| 1 | Repo, data, initial notebook | Done |
| 2 | Baseline linear regression | Planned |
| 3 | Time-series feature engineering | Planned |
| 4 | Compare advanced regressors | Planned |
| 5 | FastAPI service + free deployment | Planned |
| 6 | Model card and hand-over | Planned |

## Project structure
```
pure-drop-milk-price/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── milk_prices.csv        # clean copy: columns date, price
├── scripts/
│   └── download_data.py       # fetches data and saves the CSV
└── notebooks/
    └── 01_eda.ipynb           # summary stats + price-over-time plots
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
