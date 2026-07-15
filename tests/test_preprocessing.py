import pandas as pd
from src.data.preprocessing import clean_data


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "claim_id": ["C1", "C1", "C2"],
        "bmi": [25.0, 25.0, 30.0],
        "claim_amount": [1000.0, 1000.0, -50.0],
    })
    cleaned = clean_data(df)
    assert cleaned["claim_id"].nunique() == 2
    assert cleaned["claim_amount"].min() >= 0


def test_clean_data_fills_missing():
    df = pd.DataFrame({
        "claim_id": ["C1", "C2"],
        "bmi": [25.0, None],
        "claim_amount": [1000.0, 2000.0],
    })
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
