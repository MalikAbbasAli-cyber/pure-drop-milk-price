# Model Card: PureDrop Milk Price Forecaster

## Model details
| | |
|---|---|
| Name | PureDrop milk price forecaster |
| Type | Ridge regression (scikit-learn 1.9.1), with standardised inputs |
| File | `models/best_model.pkl` (a dict holding the model, feature list, test metrics and training end date) |
| Task | Regression: predict the selling price for one delivery date |
| Author | Malik Abbas Ali (Ezitech internship) |
| Trained through | 2025-11-18 (the training split; the last 20% of the data was held out) |

## Intended use
- **Intended:** a next-day forecast of the selling price, to help PureDrop plan inventory and pricing. The caller supplies the forecast date and the previous 30 daily prices.
- **Not intended:** forecasts several days ahead by feeding the model its own predictions (not tested, accuracy will degrade), or any use that treats the numbers as validated on real PureDrop sales.

## Data
- **The training data is synthetic.** The UCI link in the project brief returns a 404, and the dataset of that name holds monthly milk production, not price. No PureDrop sales history was provided.
- The stand-in is a generated daily series for 2022-01-01 to 2026-10-31 (1,765 rows): a slow upward trend, a yearly seasonal wave, and random noise, with a fixed random seed (`scripts/download_data.py`).
- Because the generator is a trend plus a clean seasonal curve, all scores below are **optimistic** compared with what real, messier data would give.

## Features
Built from the date and past prices only (no leakage; see `notebooks/03_time_series_features.ipynb`):
- `day_num` (days since 2022-01-01), `month`, `dayofweek`, `dayofyear`
- `sin_year`, `cos_year` (smooth yearly cycle)
- `lag_1`, `lag_7`, `lag_30` (price 1, 7 and 30 days before)
- `roll_mean_7`, `roll_mean_30` (average of the previous 7 and 30 days)

## Evaluation
Chronological 80/20 split with no shuffling: training on older data, testing on the newest 353 days. Model choice used 5-fold `TimeSeriesSplit` cross-validation on the training period only; the test set was scored once afterwards.

| Metric (test period) | Ridge (selected) | Baseline (straight line on time) |
|---|---|---|
| MAE | 0.0081 | 0.0265 |
| RMSE | 0.0103 | 0.0307 |
| R2 | 0.855 | -0.280 |

Models compared: Linear Regression, Ridge, Random Forest, Gradient Boosting, HistGradientBoosting (full table in `models/model_comparison.csv` and the README).

What the comparison showed:
- Seasonal features (sin/cos of day of year) produced almost all of the improvement over the baseline.
- Lag and rolling features did not improve the score on this data, because the noise is independent from day to day.
- Ridge and plain Linear Regression scored almost the same on the test period; Ridge was chosen because its cross-validation error was lower and more stable.
- The tree-based models scored worse (test R2 about 0.61 to 0.66) because they cannot extrapolate the upward trend beyond the range seen in training.

Prices are in the units of the dataset (synthetic, no currency).

## Limitations and risks
- **Not validated on real data.** Re-run notebooks 03 and 04 on PureDrop's real history before relying on it.
- **The model was trained only on data up to 2025-11-18** and has not been retrained on the newest data. A trend model drifts as time passes; retrain regularly.
- **The trend is linear.** `day_num` extends the trend in a straight line, so forecasts far beyond the training period become less reliable.
- **No handling of shocks** (holidays, supply disruptions, promotions); none of these were in the data.
- **Inputs are trusted.** The API checks that at least 30 positive prices are supplied but cannot tell whether they are correct or in date order.
- **Pickle file:** only load `best_model.pkl` from a source you trust, and use the pinned scikit-learn version (1.9.1) to avoid load errors.

## Maintenance
See `docs/USAGE_GUIDE.md` for how to retrain. The date `2022-01-01` used for `day_num` is set in `app/main.py` (`BASE_DATE`) and must match the first date in the training data if the dataset changes.
