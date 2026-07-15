"""Reusable Plotly chart builders shared by the Streamlit app and notebooks."""
import pandas as pd
import plotly.express as px


def avg_cost_by_group(df: pd.DataFrame, group_col: str, cost_col: str = "claim_amount"):
    agg = (
        df.groupby(group_col)[cost_col]
        .mean().round(2).reset_index(name="avg_claim_amount")
        .sort_values("avg_claim_amount", ascending=False)
    )
    fig = px.bar(
        agg, x=group_col, y="avg_claim_amount",
        title=f"Average Claim Amount by {group_col.replace('_', ' ').title()}",
        color="avg_claim_amount", color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False)
    return fig


def cost_trend_over_time(df: pd.DataFrame):
    df = df.copy()
    df["claim_month"] = pd.to_datetime(df["claim_date"]).dt.to_period("M").astype(str)
    trend = df.groupby("claim_month")["claim_amount"].sum().reset_index()
    fig = px.line(trend, x="claim_month", y="claim_amount", markers=True,
                  title="Total Claims Cost Over Time")
    return fig


def cost_distribution(df: pd.DataFrame):
    fig = px.histogram(df, x="claim_amount", nbins=50, title="Claim Amount Distribution")
    return fig


def anomaly_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df, x="age", y="claim_amount", color=df["is_anomaly_pred"].map({0: "Normal", 1: "Flagged"}),
        title="Claims by Age vs. Amount (Flagged Anomalies Highlighted)",
        labels={"color": "Status"}, opacity=0.6,
    )
    return fig


def feature_importance_chart(model, feature_columns, top_n: int = 15):
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top = importances.sort_values(ascending=True).tail(top_n)
    fig = px.bar(
        top, orientation="h", title=f"Top {top_n} Drivers of Claim Cost",
        labels={"value": "Importance", "index": "Feature"},
    )
    fig.update_layout(showlegend=False)
    return fig
