"""Feature engineering for claim cost prediction & anomaly detection."""
import pandas as pd

from src.config import PROCESSED_DATA_PATH, FEATURES_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["sex", "smoker", "region", "claim_type", "provider_network"]

DROP_COLS = ["claim_id", "claim_date", "approval_status"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype(str)
    df["is_smoker_and_obese"] = ((df["smoker"] == "yes") & (df["bmi"] >= 30)).astype(int)
    df["has_dependents"] = (df["children"] > 0).astype(int)
    df["out_of_network"] = (df["provider_network"] == "Out-of-Network").astype(int)
    return df


def encode_features(df: pd.DataFrame, extra_categorical=("bmi_category",)) -> pd.DataFrame:
    cat_cols = list(CATEGORICAL_COLS) + list(extra_categorical)
    cat_cols = [c for c in cat_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    drop_cols = [c for c in DROP_COLS if c in df_encoded.columns]
    df_encoded = df_encoded.drop(columns=drop_cols)
    return df_encoded


def build_feature_set() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = add_engineered_features(df)
    df_encoded = encode_features(df)
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(FEATURES_DATA_PATH, index=False)
    logger.info(f"Saved feature set ({df_encoded.shape[1]} columns) to {FEATURES_DATA_PATH}")
    return df_encoded


if __name__ == "__main__":
    build_feature_set()
