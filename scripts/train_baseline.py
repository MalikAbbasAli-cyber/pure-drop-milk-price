"""Task 2: train the baseline linear regression and save it.

Run from the project root:  python scripts/train_baseline.py

- Loads the cleaned data (data/milk_prices.csv)
- 80/20 train/test split with scikit-learn's train_test_split
- Fits a LinearRegression on one feature: days since the first record
- Prints MAE / RMSE / R2 on the test set
- Saves the trained model with joblib to models/model_baseline.pkl
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
MODEL_PATH = ROOT / "models" / "model_baseline.pkl"
RANDOM_STATE = 42  # fixed seed; the split below uses no randomness anyway (shuffle=False)


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["day_num"] = (df["date"] - df["date"].min()).dt.days

    X, y = df[["day_num"]], df["price"]

    # 80/20 split. shuffle=False keeps the order in time: the older 80% of days
    # train the model and the newest 20% test it, so the model never sees the
    # future. With no shuffling the split is identical on every run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE
    )

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train: {df.loc[X_train.index, 'date'].min().date()} to {df.loc[X_train.index, 'date'].max().date()}")
    print(f"Test : {df.loc[X_test.index, 'date'].min().date()} to {df.loc[X_test.index, 'date'].max().date()}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    print(f"Saved baseline model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
