"""Call the API. Usage: python scripts/example_request.py [base_url]"""
import sys
from pathlib import Path

import pandas as pd
import requests

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
prices = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "milk_prices.csv")
recent = prices["price"].tail(30).tolist()

r = requests.post(f"{base}/predict", json={"date": "2026-11-01", "recent_prices": recent}, timeout=60)
print(r.status_code, r.json())
