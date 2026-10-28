
"""Train and compare baseline and regression models on a chronological holdout."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing import prepare_features
from src.data_validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "meal_attendance_synthetic.csv"
MODEL_PATH = ROOT / "models" / "mealwise_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

def build_pipeline(model):
    categorical = ["meal_type", "menu_category"]
    numeric = ["weekday", "month", "is_weekend", "is_holiday"]
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric)
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def train_and_evaluate(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    errors = validate_dataset(df)
    if errors:
        raise ValueError("Dataset validation failed: " + " | ".join(errors))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    features = prepare_features(df)
    y = pd.to_numeric(df["attendance"]).astype(float)
    split = max(1, int(len(df) * 0.8))
    if split >= len(df):
        raise ValueError("Need at least two records to create a train/test split.")
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    candidates = {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=160, max_depth=12, random_state=42, n_jobs=-1)
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, pipe.predict(X_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results.append({"model": name, "MAE": float(mean_absolute_error(y_test, pred)),
                        "RMSE": rmse, "R2": float(r2_score(y_test, pred))})
        fitted[name] = pipe
    # Select the lowest MAE among ML models, while retaining the baseline for comparison.
    ml_results = [r for r in results if r["model"] != "Dummy baseline"]
    best_name = min(ml_results, key=lambda r: r["MAE"])["model"]
    best_model = fitted[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {"selected_model": best_name, "selection_rule": "Lowest held-out MAE among the three ML models",
                "dataset_type": "synthetic demonstration data", "test_records": int(len(y_test)),
                "train_records": int(len(y_train)), "metrics": results,
                "warning": "These results describe synthetic data only and are not evidence of real cafeteria performance."}
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    train_and_evaluate()
