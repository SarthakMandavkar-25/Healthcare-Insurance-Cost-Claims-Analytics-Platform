import pandas as pd
from src.features.feature_engineering import add_engineered_features


def test_smoker_obese_flag():
    df = pd.DataFrame({
        "bmi": [22.0, 32.0],
        "smoker": ["no", "yes"],
        "children": [0, 2],
        "provider_network": ["In-Network", "Out-of-Network"],
    })
    out = add_engineered_features(df)
    assert out.loc[0, "is_smoker_and_obese"] == 0
    assert out.loc[1, "is_smoker_and_obese"] == 1
    assert out.loc[1, "has_dependents"] == 1
    assert out.loc[1, "out_of_network"] == 1
