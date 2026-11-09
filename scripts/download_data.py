"""Fetch milk data and save a clean copy to data/milk_prices.csv.

1. Tries to download the real dataset from the URL below.
2. If the URL does not serve a CSV (the UCI page is an HTML page, not data),
   falls back to a clearly labelled SYNTHETIC daily price series so the
   pipeline always runs end to end.
"""
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/ml/datasets/Milk+Production"
OUT = Path(__file__).resolve().parents[1] / "data" / "milk_prices.csv"


def try_download(url: str):
    """Return a DataFrame if the URL serves a CSV, otherwise None."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            return None  # got a web page, not data
        df = pd.read_csv(io.StringIO(r.text))
        if not {"date", "price"}.issubset(df.columns):
            return None  # wrong schema (e.g. production, not price)
        return df
    except Exception as exc:
        print(f"Download failed: {exc}")
        return None


def synthetic_daily_prices(start="2022-01-01", end="2026-10-31", seed=42):
    """Stand-in daily price series: trend + yearly seasonality + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))
    price = (
        0.55
        + 0.00006 * t
        + 0.04 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 0.01, len(dates))
    )
    return pd.DataFrame({"date": dates, "price": price.round(3)})


def main():
    df = try_download(URL)
    if df is None:
        print("Using SYNTHETIC data (real dataset unavailable).")
        df = synthetic_daily_prices()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().drop_duplicates("date").sort_values("date")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
