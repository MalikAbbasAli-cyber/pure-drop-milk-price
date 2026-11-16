"""Task 3: add time-series features, train a new model, compare it to the baseline.

Run from the project root:  python scripts/feature_engineering.py

Steps
1. Load data/milk_prices.csv and sort by date
2. Add four new columns: lag_1, ma_7, day_of_week, month
3. Drop the rows made incomplete by the lag / moving average (NaN)
4. Save the enriched dataset to data/milk_prices_fe.csv
5. 80/20 split with train_test_split (same settings as the baseline script)
6. Train a LinearRegression on the new features, save it to models/model_fe.pkl (joblib)
7. Score the baseline (days-since-start only) on the SAME test rows and print both RMSEs
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "milk_prices.csv"
FE_DATA = ROOT / "data" / "milk_prices_fe.csv"
MODEL_PATH = ROOT / "models" / "model_fe.pkl"
RANDOM_STATE = 42
NEW_FEATURES = ["lag_1", "ma_7", "day_of_week", "month"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with four new columns. Only past prices are used (no leakage)."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["lag_1"] = df["price"].shift(1)                         # previous day's price
    df["ma_7"] = df["price"].shift(1).rolling(7).mean()        # mean of the previous 7 days
    df["day_of_week"] = df["date"].dt.dayofweek                # 0 = Monday
    df["month"] = df["date"].dt.month
    return df


def metrics(y_true, pred):
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": r2_score(y_true, pred),
    }


def main():
    raw = pd.read_csv(DATA, parse_dates=["date"])
    df = add_features(raw)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)      # first rows have no history
    print(f"Rows: {rows_before} -> {len(df)} after dropping NaN from lag / moving average")

    # baseline feature (days since the first record), for a like-for-like comparison
    df["day_num"] = (df["date"] - raw["date"].min()).dt.days
    df.drop(columns="day_num").to_csv(FE_DATA, index=False)
    print(f"Saved enriched dataset ({len(NEW_FEATURES)} new columns) to {FE_DATA}")

    y = df["price"]
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, shuffle=False, random_state=RANDOM_STATE
    )

    # baseline: day_num only
    base = LinearRegression().fit(df.loc[idx_train, ["day_num"]], y[idx_train])
    base_m = metrics(y[idx_test], base.predict(df.loc[idx_test, ["day_num"]]))

    # new model: engineered features
    fe = LinearRegression().fit(df.loc[idx_train, NEW_FEATURES], y[idx_train])
    fe_m = metrics(y[idx_test], fe.predict(df.loc[idx_test, NEW_FEATURES]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fe, MODEL_PATH)

    print(f"Train rows: {len(idx_train)} | Test rows: {len(idx_test)}")
    print(f"New features: {', '.join(NEW_FEATURES)}")
    print(f"{'':22s}{'MAE':>8s}{'RMSE':>9s}{'R2':>9s}")
    print(f"{'Baseline (day_num)':22s}{base_m['MAE']:8.4f}{base_m['RMSE']:9.4f}{base_m['R2']:9.4f}")
    print(f"{'Feature model':22s}{fe_m['MAE']:8.4f}{fe_m['RMSE']:9.4f}{fe_m['R2']:9.4f}")
    change = (base_m["RMSE"] - fe_m["RMSE"]) / base_m["RMSE"] * 100
    print(f"RMSE change vs baseline: {-change:+.1f}%  ({'lower' if change > 0 else 'HIGHER'} than baseline)")
    print(f"Saved feature model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
