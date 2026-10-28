
"""Feature preparation shared by training and prediction."""
import pandas as pd

TARGET = "attendance"
CATEGORICAL_FEATURES = ["meal_type", "menu_category"]
NUMERIC_FEATURES = ["weekday", "month", "is_weekend", "is_holiday"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "meal_type", "menu_category", "is_holiday"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Please provide a valid date.")
    out["meal_type"] = out["meal_type"].astype(str).str.strip().str.lower()
    if not out["meal_type"].isin(["breakfast", "lunch", "dinner"]).all():
        raise ValueError("Meal type must be breakfast, lunch, or dinner.")
    out["menu_category"] = out["menu_category"].astype(str).str.strip()
    if (out["menu_category"] == "").any():
        raise ValueError("Menu category cannot be blank.")
    out["is_holiday"] = pd.to_numeric(out["is_holiday"], errors="coerce")
    if out["is_holiday"].isna().any() or not out["is_holiday"].isin([0, 1]).all():
        raise ValueError("Holiday must be 0 (No) or 1 (Yes).")
    out["weekday"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    return out[["meal_type", "menu_category", "weekday", "month", "is_weekend", "is_holiday"]]
