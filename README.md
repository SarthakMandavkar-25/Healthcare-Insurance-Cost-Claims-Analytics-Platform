# 💰 Healthcare Insurance Cost & Claims Analytics Platform

An end-to-end analytics platform that predicts expected health-insurance
claim costs, flags anomalous/suspicious claims for review, and surfaces
both through an interactive Streamlit dashboard plus a Power BI-ready export.

**Live demo:** _add your deployed Streamlit/Render link here_
**Power BI report:** _add your published Power BI Service link here_

---

## Why this project

Health insurers and claims-management teams constantly balance two goals:
predicting expected cost for pricing/reserving, and catching abnormal claims
before they're paid out. This project builds both capabilities as separate,
composable models on top of one shared feature pipeline — mirroring how a
real actuarial/claims-analytics team would structure the work.

## Architecture

```
healthcare-insurance-analytics/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── generate_data.py       # Synthetic claims generator (swap for real source)
│   │   └── preprocessing.py       # Cleaning & imputation
│   ├── features/
│   │   └── feature_engineering.py # BMI category, smoker/obesity flag, etc.
│   ├── models/
│   │   ├── train.py               # Cost regressor + residual-based anomaly detector
│   │   └── predict.py             # Inference wrapper (schema-safe)
│   ├── visualization/
│   │   └── charts.py
│   └── utils/logger.py
├── app/streamlit_app.py           # 3-tab dashboard
├── tests/                         # Pytest unit tests
├── powerbi/                       # Power BI export + setup guide
├── deployment/{Dockerfile, docker-compose.yml}
├── .github/workflows/ci.yml
├── render.yaml
├── run_pipeline.py
└── requirements.txt
```

### Two-model design

1. **Cost model** — `RandomForestRegressor` predicts expected claim amount
   from patient/claim attributes (age, BMI, smoker status, claim type, network).
2. **Anomaly model** — `IsolationForest` runs on the **residual** between
   actual and predicted cost (not raw features). This is the same core idea
   real fraud/claims-review teams use: a claim isn't suspicious because of
   its raw dollar amount, it's suspicious because it's far higher than the
   cost model expected for that patient profile. This gets ~90%+ precision
   at catching the injected anomalies, vs. ~15% when naively run on raw features.

## Tech stack

- **Python** — pandas, scikit-learn, joblib
- **Modeling** — RandomForest (regression) + IsolationForest (anomaly detection)
- **Dashboard** — Streamlit + Plotly
- **BI reporting** — Power BI (via CSV export)
- **Deployment** — Docker, docker-compose, Render.com blueprint
- **CI** — GitHub Actions

## Getting started

```bash
git clone <your-repo-url>
cd healthcare-insurance-analytics
pip install -r requirements.txt

python run_pipeline.py
streamlit run app/streamlit_app.py
```

## Run with Docker

```bash
docker compose -f deployment/docker-compose.yml up --build
# Dashboard available at http://localhost:8502
```

## Deploy to Render (or Streamlit Community Cloud)

- **Render:** push to GitHub → "New → Blueprint" → point at this repo (`render.yaml` handles the rest)
- **Streamlit Cloud:** connect the repo, set main file to `app/streamlit_app.py`

## Power BI report

See [`powerbi/README.md`](powerbi/README.md) for the step-by-step guide.

## Model performance

| Model | Metric | Score |
|---|---|---|
| Cost Regressor | MAE | ~$1,700 |
| Cost Regressor | R² | ~0.42 |
| Anomaly Detector | Precision | ~0.90 |
| Anomaly Detector | Recall | ~0.95 |

The cost model's R² is intentionally moderate — ~3.5% of claims are
synthetically injected as cost outliers (simulating fraud/billing errors),
which the *cost* model isn't meant to predict. Those are exactly what the
*anomaly* model is designed to catch instead, which is why it scores so
much higher on precision/recall.

## Testing

```bash
pytest tests/ -v
```

## Notes on the data

Ships with a **synthetic but realistic** dataset structured after the public
"Medical Cost Personal Datasets", extended with claims-operations fields
(claim type, network, approval status) and injected anomalies. Swap
`src/data/generate_data.py`'s output for a real claims source and keep the
same schema — nothing downstream needs to change.

## License

MIT — see [LICENSE](LICENSE).
