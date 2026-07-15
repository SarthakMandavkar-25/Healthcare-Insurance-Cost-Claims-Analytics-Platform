"""
Healthcare Insurance Cost & Claims Analytics — Streamlit Dashboard
Run locally:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_PATH, METRICS_PATH, COST_MODEL_PATH, ANOMALY_MODEL_PATH, POWERBI_EXPORT_PATH
from src.models.predict import ClaimsPredictor
from src.visualization.charts import (
    avg_cost_by_group, cost_trend_over_time, cost_distribution,
    anomaly_scatter, feature_importance_chart,
)

st.set_page_config(page_title="Insurance Claims Analytics", layout="wide", page_icon="💰")


@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["claim_date"])


@st.cache_resource
def load_predictor():
    return ClaimsPredictor(COST_MODEL_PATH, ANOMALY_MODEL_PATH)


if not PROCESSED_DATA_PATH.exists() or not COST_MODEL_PATH.exists():
    st.title("💰 Healthcare Insurance Cost & Claims Analytics")
    st.warning("No processed data found. Run the pipeline first:\n\n```\npython run_pipeline.py\n```")
    st.stop()

df = load_data()
predictor = load_predictor()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

st.title("💰 Healthcare Insurance Cost & Claims Analytics")
st.caption("Claim cost trends, anomaly/fraud-pattern detection, and a live cost estimator")

tab_overview, tab_anomaly, tab_estimate = st.tabs(
    ["📊 Claims Overview", "🚩 Anomaly Detection", "🧮 Cost Estimator"]
)

# ---------------- Overview ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(df):,}")
    c2.metric("Total Claim Value", f"${df['claim_amount'].sum():,.0f}")
    c3.metric("Avg Claim", f"${df['claim_amount'].mean():,.0f}")
    c4.metric("Cost Model R²", f"{metrics['cost_model']['r2_score']:.2f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        group = st.selectbox(
            "Average cost by:", ["region", "claim_type", "smoker", "provider_network", "sex"]
        )
        st.plotly_chart(avg_cost_by_group(df, group), use_container_width=True)
    with col2:
        st.plotly_chart(cost_trend_over_time(df), use_container_width=True)

    st.plotly_chart(cost_distribution(df), use_container_width=True)
    st.subheader("Claims sample")
    st.dataframe(df.head(200), use_container_width=True, height=300)

# ---------------- Anomaly Detection ----------------
with tab_anomaly:
    st.subheader("Suspicious claim detection (unsupervised IsolationForest)")
    df_scored = df.copy()
    df_scored["is_anomaly_pred"] = predictor.predict_anomaly(df_scored)

    c1, c2 = st.columns(2)
    c1.metric("Flagged Claims", f"{df_scored['is_anomaly_pred'].sum():,}")
    c2.metric("Detection Precision", f"{metrics['anomaly_model']['precision']*100:.1f}%")

    st.plotly_chart(anomaly_scatter(df_scored), use_container_width=True)

    st.subheader("Flagged claims")
    st.dataframe(
        df_scored[df_scored["is_anomaly_pred"] == 1]
        .sort_values("claim_amount", ascending=False)
        .head(50),
        use_container_width=True, height=300,
    )

    with st.expander("Model evaluation metrics"):
        st.json(metrics)

# ---------------- Cost Estimator ----------------
with tab_estimate:
    st.subheader("Estimate expected claim cost")
    with st.form("cost_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 85, 40)
        sex = c1.selectbox("Sex", df["sex"].unique())
        bmi = c1.slider("BMI", 15.0, 55.0, 27.0)

        children = c2.slider("Children", 0, 5, 0)
        smoker = c2.selectbox("Smoker", df["smoker"].unique())
        region = c2.selectbox("Region", df["region"].dropna().unique())

        claim_type = c3.selectbox("Claim type", df["claim_type"].unique())
        provider_network = c3.selectbox("Provider network", df["provider_network"].unique())

        submitted = st.form_submit_button("Estimate cost", type="primary")

    if submitted:
        record = pd.DataFrame([{
            "age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker,
            "region": region, "claim_type": claim_type, "provider_network": provider_network,
        }])
        predicted_cost = predictor.predict_cost(record).iloc[0]
        st.metric("Predicted claim amount", f"${predicted_cost:,.2f}")

st.sidebar.header("About")
st.sidebar.info(
    "Modular ML pipeline for insurance claims analytics: cost prediction (RandomForest "
    "Regressor) + anomaly detection (IsolationForest). "
    f"Power BI export available at `{POWERBI_EXPORT_PATH.name}`."
)
