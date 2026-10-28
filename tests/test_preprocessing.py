
import pandas as pd
import pytest
from src.preprocessing import prepare_features

def test_prepare_features_extracts_calendar_columns():
    df = pd.DataFrame([{"date":"2026-10-10","meal_type":"lunch","menu_category":"South Indian","is_holiday":0}])
    out = prepare_features(df)
    assert list(out.columns) == ["meal_type","menu_category","weekday","month","is_weekend","is_holiday"]
    assert out.iloc[0]["is_weekend"] == 1

def test_invalid_meal_is_rejected():
    df = pd.DataFrame([{"date":"2026-10-10","meal_type":"snack","menu_category":"Snack","is_holiday":0}])
    with pytest.raises(ValueError):
        prepare_features(df)
