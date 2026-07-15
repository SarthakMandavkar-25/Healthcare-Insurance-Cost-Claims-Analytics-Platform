"""Central configuration for the Insurance Cost & Claims Analytics pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
POWERBI_DIR = ROOT_DIR / "powerbi"

RAW_DATA_PATH = DATA_RAW_DIR / "insurance_claims.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "claims_clean.csv"
FEATURES_DATA_PATH = DATA_PROCESSED_DIR / "claims_features.csv"
POWERBI_EXPORT_PATH = POWERBI_DIR / "claims_dashboard_data.csv"

COST_MODEL_PATH = MODEL_DIR / "cost_model.pkl"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
N_SYNTHETIC_RECORDS = 10000

COST_TARGET_COLUMN = "claim_amount"

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, POWERBI_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
