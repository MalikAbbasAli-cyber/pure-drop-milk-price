# Hand-over: PureDrop Milk Price Prediction

## What is delivered
| Item | Location |
|---|---|
| Source code and README | GitHub: https://github.com/MalikAbbasAli-cyber/pure-drop-milk-price |
| Data ingestion script | `scripts/download_data.py` |
| Exploratory notebook | `notebooks/01_eda.ipynb` |
| Modelling notebooks (baseline, features, comparison) | `notebooks/02_*.ipynb` to `04_*.ipynb` |
| Trained model (pickle) | `models/best_model.pkl` |
| FastAPI service with `/predict` | `app/main.py` |
| Live deployment | https://pure-drop-milk-price.onrender.com |
| Model card | `docs/MODEL_CARD.md` |
| Usage guide | `docs/USAGE_GUIDE.md` |

## How it works, in short
The notebooks build features from the date and recent prices, compare five regression models with time-aware cross-validation, and save the best (Ridge regression). The API loads that file, rebuilds the same features from the request, and returns a predicted price.

## Important: read before relying on it
1. **The data is synthetic.** No PureDrop sales history was provided and the dataset named in the brief was not usable. The scores (test R2 about 0.86, MAE about 0.008) describe performance on generated data and will be worse on real data.
2. **Retrain on real data first.** Follow section 4 of the usage guide. Until then, treat the service as a working demonstration of the pipeline, not a validated forecast.
3. **Callers must send the last 30 daily prices** with each request.
4. **Free hosting limits:** the service sleeps when idle (slow first request) and is not suitable for production traffic. A paid instance or another host is the next step.
5. **No authentication, logging or monitoring** was added; the brief asked for a simple API.

## Suggested next steps for PureDrop
- Supply real daily sales/price history and retrain (usage guide, section 4).
- Add API authentication and request logging before exposing the service widely.
- Schedule periodic retraining and compare new errors with those in the model card.
- Consider extra inputs that affect price (holidays, promotions, supply), which the current model does not use.

## Ownership
Built by Malik Abbas Ali as an Ezitech internship project for PureDrop Dairy.
