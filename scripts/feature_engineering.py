"""Task 3: feature engineering + a new LinearRegression model (model_fe.pkl).

Run from the project root:  python scripts/feature_engineering.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

# 1. Load the original CSV and sort by date
df = pd.read_csv(ROOT / "data" / "milk_prices.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Add four new feature columns (only past prices are used, so no leakage)
df["lag_1"] = df["price"].shift(1)                    # previous day's price
df["ma_7"] = df["price"].shift(1).rolling(7).mean()   # mean of the previous 7 days
df["day_of_week"] = df["date"].dt.day_of_week         # 0 = Monday
df["month"] = df["date"].dt.month
FEATURES = ["lag_1", "ma_7", "day_of_week", "month"]

# 3. Drop the rows with NaN created by the lag / moving average
rows_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Rows: {rows_before} -> {len(df)} after dropping NaN")
df.to_csv(ROOT / "data" / "milk_prices_fe.csv", index=False)   # enriched dataset

# 4. Same 80/20 train/test split as the baseline (no shuffling, fixed random_state)
train, test = train_test_split(df, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)

# 4a. Baseline for comparison: LinearRegression on days-since-start only, same rows
df_day = (df["date"] - df["date"].min()).dt.days
base_model = LinearRegression().fit(df_day.loc[train.index].to_frame(), train["price"])
base_rmse = np.sqrt(mean_squared_error(test["price"], base_model.predict(df_day.loc[test.index].to_frame())))

# 4b. Train a NEW LinearRegression on the engineered features and evaluate it
model_fe = LinearRegression().fit(train[FEATURES], train["price"])
fe_rmse = np.sqrt(mean_squared_error(test["price"], model_fe.predict(test[FEATURES])))

# 5. Save the new model as models/model_fe.pkl
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model_fe, ROOT / "models" / "model_fe.pkl")

print(f"New features: {', '.join(FEATURES)}")
print(f"Baseline RMSE      : {base_rmse:.4f}")
print(f"Feature model RMSE : {fe_rmse:.4f}")
print(f"RMSE is {'LOWER' if fe_rmse < base_rmse else 'HIGHER'} than the baseline "
      f"({(fe_rmse - base_rmse) / base_rmse * 100:+.1f}%)")
print("Saved model to models/model_fe.pkl")