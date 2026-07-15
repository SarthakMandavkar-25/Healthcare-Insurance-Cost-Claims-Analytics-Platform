"""
Synthetic health insurance claims data generator.

Structurally similar to the well-known public "Medical Cost Personal Datasets"
(age, bmi, smoker, region, charges) extended with claims-operations fields
(claim type, provider, approval status, fraud flag) so the project supports
both a cost-prediction model and a claims-anomaly-detection model.
Swap `load_raw_data()` in preprocessing.py for a real data source in production.
"""
import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, N_SYNTHETIC_RECORDS, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_claims(n: int = N_SYNTHETIC_RECORDS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 85, size=n)
    sex = rng.choice(["male", "female"], size=n)
    bmi = np.clip(rng.normal(28, 6, size=n), 15, 55).round(1)
    children = rng.integers(0, 5, size=n)
    smoker = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], size=n)

    claim_type = rng.choice(
        ["Outpatient", "Inpatient", "Emergency", "Preventive", "Dental", "Pharmacy"],
        size=n, p=[0.28, 0.15, 0.12, 0.18, 0.12, 0.15],
    )
    provider_network = rng.choice(["In-Network", "Out-of-Network"], size=n, p=[0.8, 0.2])

    # --- Cost model: clinically-plausible drivers of medical charges ---
    base_cost = (
        250
        + age * 45
        + (bmi - 25).clip(min=0) * 90
        + children * 150
        + (smoker == "yes") * 9500
        + (claim_type == "Inpatient") * 4200
        + (claim_type == "Emergency") * 2100
        + (provider_network == "Out-of-Network") * 1300
    )
    noise = rng.normal(0, 500, size=n)
    claim_amount = np.clip(base_cost + noise, 100, None).round(2)

    # --- Anomaly / fraud-like signal: small % of claims with abnormal cost-to-profile ratio ---
    is_anomalous = rng.random(n) < 0.035
    claim_amount = np.where(
        is_anomalous, claim_amount * rng.uniform(3.5, 6.0, size=n), claim_amount
    ).round(2)

    approval_status = np.where(
        is_anomalous, rng.choice(["Flagged", "Denied"], size=n, p=[0.7, 0.3]), "Approved"
    )

    claim_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 545, size=n), unit="D"
    )

    df = pd.DataFrame({
        "claim_id": [f"CLM{500000+i}" for i in range(n)],
        "claim_date": claim_date,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "claim_type": claim_type,
        "provider_network": provider_network,
        "claim_amount": claim_amount,
        "approval_status": approval_status,
        "is_anomalous_flag": is_anomalous.astype(int),
    })

    # Inject small amount of realistic missingness
    for col in ["bmi", "region"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


def main():
    logger.info("Generating synthetic insurance claims dataset...")
    df = generate_claims()
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Saved {len(df):,} records to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
