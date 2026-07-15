# Power BI Dashboard — Setup Guide

The pipeline exports a flattened, BI-ready dataset to:

```
powerbi/claims_dashboard_data.csv
```

This includes raw claim fields **plus** the model's `predicted_cost`,
`cost_variance` (actual − predicted), and `is_anomaly_pred` — so Power BI
can be used purely for reporting while Python owns the ML logic.

## Steps to build the report

1. Run `python run_pipeline.py` to generate `claims_dashboard_data.csv`.
2. Open Power BI Desktop → **Get Data → Text/CSV** → select the file above.
3. Suggested visuals:
   - Card visuals: total claims, total claim value, average claim, flagged claims
   - Bar chart: average claim amount by `region`, `claim_type`, `smoker`
   - Line chart: total claim cost by month (`claim_date`)
   - Scatter: `age` vs `claim_amount`, colored by `is_anomaly_pred`
   - Table: top flagged claims sorted by `cost_variance`
4. Set up a **scheduled refresh** by pointing Power BI's data source at the
   same CSV path (or swap in a database connection in `src/config.py`).
5. Publish to the Power BI Service and share the workspace link on your
   resume/LinkedIn next to the live Streamlit app link.

> Tip: keep a screenshot of the finished report in `docs/screenshots/` and
> embed it in the main `README.md`.
