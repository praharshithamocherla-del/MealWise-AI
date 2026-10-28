
"""Validation helpers for MealWise AI datasets."""
import pandas as pd

REQUIRED_COLUMNS = ["date", "meal_type", "menu_category", "attendance", "is_holiday"]
VALID_MEALS = {"breakfast", "lunch", "dinner"}

def validate_dataset(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return errors
    if df.empty:
        errors.append("Dataset is empty.")
        return errors
    if pd.to_datetime(df["date"], errors="coerce").isna().any():
        errors.append("Some date values are invalid.")
    if (~df["meal_type"].astype(str).str.lower().isin(VALID_MEALS)).any():
        errors.append("meal_type must contain breakfast, lunch, or dinner.")
    attendance = pd.to_numeric(df["attendance"], errors="coerce")
    if attendance.isna().any():
        errors.append("attendance contains missing or non-numeric values.")
    elif (attendance < 0).any():
        errors.append("attendance cannot be negative.")
    if df["is_holiday"].isna().any():
        errors.append("is_holiday contains missing values.")
    return errors
