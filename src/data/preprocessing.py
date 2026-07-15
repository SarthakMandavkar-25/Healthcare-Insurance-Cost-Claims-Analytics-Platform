"""Data cleaning and preprocessing for insurance claims records."""
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python -m src.data.generate_data` first "
            f"(or point this at a real claims data source in production)."
        )
    return pd.read_csv(path, parse_dates=["claim_date"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)

    df = df.drop_duplicates(subset="claim_id")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    df["bmi"] = df["bmi"].clip(lower=12, upper=60)
    df["claim_amount"] = df["claim_amount"].clip(lower=0)

    logger.info(f"Cleaned data: {initial_rows} -> {len(df)} rows")
    return df


def run_preprocessing() -> pd.DataFrame:
    df = load_raw_data()
    df = clean_data(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved cleaned data to {PROCESSED_DATA_PATH}")
    return df


if __name__ == "__main__":
    run_preprocessing()
