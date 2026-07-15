"""Load trained models and score new/unseen claims."""
import joblib
import pandas as pd

from src.config import COST_MODEL_PATH, ANOMALY_MODEL_PATH, COST_TARGET_COLUMN
from src.features.feature_engineering import add_engineered_features, encode_features


class ClaimsPredictor:
    """Wraps the cost regressor + residual-based anomaly detector behind one interface."""

    def __init__(self, cost_model_path=COST_MODEL_PATH, anomaly_model_path=ANOMALY_MODEL_PATH):
        cost_bundle = joblib.load(cost_model_path)
        self.cost_model = cost_bundle["model"]
        self.cost_feature_columns = cost_bundle["feature_columns"]

        anomaly_bundle = joblib.load(anomaly_model_path)
        self.anomaly_model = anomaly_bundle["model"]
        self.anomaly_feature_columns = anomaly_bundle["feature_columns"]

    def _prepare(self, df_raw: pd.DataFrame, feature_columns) -> pd.DataFrame:
        df = add_engineered_features(df_raw)
        df = encode_features(df)
        df = df.reindex(columns=feature_columns, fill_value=0)
        return df

    def predict_cost(self, df_raw: pd.DataFrame) -> pd.Series:
        X = self._prepare(df_raw, self.cost_feature_columns)
        return pd.Series(self.cost_model.predict(X), index=df_raw.index, name="predicted_cost")

    def predict_anomaly(self, df_raw: pd.DataFrame) -> pd.Series:
        predicted_cost = self.predict_cost(df_raw)
        actual = df_raw[COST_TARGET_COLUMN] if COST_TARGET_COLUMN in df_raw.columns else predicted_cost
        residual = actual - predicted_cost
        residual_ratio = residual / (predicted_cost + 1)
        X = pd.DataFrame({
            "cost_residual": residual,
            "cost_residual_ratio": residual_ratio,
            "claim_amount": actual,
        }, index=df_raw.index)
        raw = self.anomaly_model.predict(X)
        return pd.Series((raw == -1).astype(int), index=df_raw.index, name="is_anomaly_pred")
