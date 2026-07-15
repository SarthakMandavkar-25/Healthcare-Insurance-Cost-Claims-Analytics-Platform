"""
End-to-end pipeline entry point:
    generate synthetic data -> clean -> engineer features -> train models -> export for Power BI

Usage:
    python run_pipeline.py
"""
import pandas as pd

from src.data.generate_data import main as generate_data
from src.data.preprocessing import run_preprocessing
from src.features.feature_engineering import build_feature_set
from src.models.train import train_all
from src.config import PROCESSED_DATA_PATH, POWERBI_EXPORT_PATH, COST_MODEL_PATH, ANOMALY_MODEL_PATH
from src.models.predict import ClaimsPredictor
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def export_for_powerbi():
    """Produce a flat, BI-tool-friendly CSV with raw fields + model outputs."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    predictor = ClaimsPredictor(COST_MODEL_PATH, ANOMALY_MODEL_PATH)
    df["predicted_cost"] = predictor.predict_cost(df).round(2)
    df["is_anomaly_pred"] = predictor.predict_anomaly(df)
    df["cost_variance"] = (df["claim_amount"] - df["predicted_cost"]).round(2)
    df.to_csv(POWERBI_EXPORT_PATH, index=False)
    logger.info(f"Power BI-ready export written to {POWERBI_EXPORT_PATH}")


def main():
    logger.info("STEP 1/5: Generating synthetic data")
    generate_data()

    logger.info("STEP 2/5: Preprocessing")
    run_preprocessing()

    logger.info("STEP 3/5: Feature engineering")
    build_feature_set()

    logger.info("STEP 4/5: Model training (cost regressor + anomaly detector)")
    train_all()

    logger.info("STEP 5/5: Exporting Power BI dataset")
    export_for_powerbi()

    logger.info("Pipeline complete. Launch the dashboard with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
