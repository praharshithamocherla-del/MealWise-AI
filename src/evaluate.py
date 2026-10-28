
"""Print evaluation metrics and save actual-vs-predicted and residual charts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing import prepare_features
from src.train import DATA_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
def evaluate():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    test = df.iloc[split:]
    if test.empty:
        raise ValueError("Test set is empty; add more records.")
    model = joblib.load(MODEL_PATH)
    actual = test["attendance"].astype(float).to_numpy()
    pred = np.maximum(0, model.predict(prepare_features(test)))
    metrics = {"MAE": float(mean_absolute_error(actual, pred)),
               "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
               "R2": float(r2_score(actual, pred))}
    (ROOT / "models" / "selected_model_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    plt.figure(figsize=(7, 5)); plt.scatter(actual, pred, alpha=.6)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--"); plt.xlabel("Actual attendance"); plt.ylabel("Predicted attendance")
    plt.title("Actual vs Predicted (synthetic test data)"); plt.tight_layout()
    plt.savefig(ROOT / "models" / "actual_vs_predicted.png", dpi=140); plt.close()
    residuals = actual - pred
    plt.figure(figsize=(7, 5)); plt.scatter(pred, residuals, alpha=.6); plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted attendance"); plt.ylabel("Residual (actual - predicted)")
    plt.title("Residual Plot (synthetic test data)"); plt.tight_layout()
    plt.savefig(ROOT / "models" / "residual_plot.png", dpi=140); plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics
if __name__ == "__main__":
    evaluate()
