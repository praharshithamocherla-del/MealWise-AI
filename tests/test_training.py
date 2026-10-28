
import pandas as pd
from src.data_validation import validate_dataset

def test_required_columns_are_checked():
    errors = validate_dataset(pd.DataFrame({"date":["2026-01-01"]}))
    assert any("Missing required columns" in e for e in errors)
