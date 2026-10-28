
"""Load the trained pipeline and predict meal attendance."""
from pathlib import Path
import joblib
import pandas as pd
from src.preprocessing import prepare_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mealwise_model.joblib"

def predict_attendance(input_df: pd.DataFrame, model_path: Path = MODEL_PATH) -> list[float]:
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.train")
    model = joblib.load(model_path)
    features = prepare_features(input_df)
    predictions = model.predict(features)
    return [max(0.0, float(x)) for x in predictions]
