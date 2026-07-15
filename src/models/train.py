"""Train two models:
1. Cost regressor  — predicts expected claim_amount from patient/claim profile.
2. Anomaly detector — unsupervised IsolationForest to flag suspicious claims
   (evaluated against a held-out synthetic ground-truth flag for reporting).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score

from src.config import (
    FEATURES_DATA_PATH, COST_MODEL_PATH, ANOMALY_MODEL_PATH,
    METRICS_PATH, COST_TARGET_COLUMN, RANDOM_STATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

LEAKAGE_COLS = ["is_anomalous_flag"]  # not available at prediction time in production


def train_cost_model(df: pd.DataFrame):
    X = df.drop(columns=[COST_TARGET_COLUMN] + LEAKAGE_COLS)
    y = df[COST_TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    logger.info("Training RandomForestRegressor for claim cost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    logger.info(f"Cost model metrics: {metrics}")

    joblib.dump({"model": model, "feature_columns": list(X.columns)}, COST_MODEL_PATH)
    return model, metrics, list(X.columns)


def train_anomaly_model(df: pd.DataFrame, cost_model, cost_feature_columns):
    X_cost = df[cost_feature_columns]
    predicted_cost = cost_model.predict(X_cost)

    residual = df[COST_TARGET_COLUMN] - predicted_cost
    residual_ratio = residual / (predicted_cost + 1)

    X = pd.DataFrame({
        "cost_residual": residual,
        "cost_residual_ratio": residual_ratio,
        "claim_amount": df[COST_TARGET_COLUMN],
    })
    y_true = df["is_anomalous_flag"]

    model = IsolationForest(
        n_estimators=300, contamination=0.035, random_state=RANDOM_STATE, n_jobs=-1
    )
    logger.info("Training IsolationForest on cost-model residuals for anomaly detection...")
    model.fit(X)

    raw_pred = model.predict(X)  # -1 = anomaly, 1 = normal
    y_pred = (raw_pred == -1).astype(int)

    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "flagged_count": int(y_pred.sum()),
    }
    logger.info(f"Anomaly model metrics: {metrics}")

    joblib.dump(
        {"model": model, "feature_columns": list(X.columns), "cost_feature_columns": cost_feature_columns},
        ANOMALY_MODEL_PATH,
    )
    return model, metrics, list(X.columns)


def train_all():
    df = pd.read_csv(FEATURES_DATA_PATH)
    cost_model, cost_metrics, cost_feature_columns = train_cost_model(df)
    anomaly_model, anomaly_metrics, _ = train_anomaly_model(df, cost_model, cost_feature_columns)

    all_metrics = {"cost_model": cost_metrics, "anomaly_model": anomaly_metrics}
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Models saved to {COST_MODEL_PATH} and {ANOMALY_MODEL_PATH}")
    return all_metrics


if __name__ == "__main__":
    train_all()
