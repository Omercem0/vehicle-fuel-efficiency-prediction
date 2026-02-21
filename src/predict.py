"""
Prediction module for Vehicle Fuel Efficiency.
Loads saved model & scaler, preprocesses input, and returns MPG predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


def load_artifacts():
    """Load model, scaler, and metadata from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def prepare_input(
    cylinders: int,
    displacement: float,
    horsepower: float,
    weight: float,
    acceleration: float,
    model_year: int,
    origin: int,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame matching the training feature layout.
    Handles one-hot encoding and column alignment.
    """
    raw = pd.DataFrame(
        {
            "Displacement": [displacement],
            "Horsepower": [horsepower],
            "Weight": [weight],
            "Acceleration": [acceleration],
            "Model Year": [model_year],
            "Cylinders": [str(cylinders)],
            "Origin": [str(origin)],
        }
    )

    raw = pd.get_dummies(raw)

    # Align columns with training set (fill missing dummy cols with 0)
    raw = raw.reindex(columns=feature_columns, fill_value=0)
    return raw


def predict_mpg(
    cylinders: int,
    displacement: float,
    horsepower: float,
    weight: float,
    acceleration: float,
    model_year: int,
    origin: int,
) -> float:
    """
    End-to-end prediction: input → preprocess → scale → predict → inverse-transform.
    Returns estimated MPG value.
    """
    model, scaler, metadata = load_artifacts()
    feature_columns = metadata["feature_columns"]

    X = prepare_input(
        cylinders, displacement, horsepower, weight,
        acceleration, model_year, origin, feature_columns,
    )
    X_scaled = scaler.transform(X)
    y_log = model.predict(X_scaled)

    # Inverse log1p to get real MPG
    mpg = np.expm1(y_log[0])
    return float(mpg)


# ---------- CLI ----------
if __name__ == "__main__":
    print("Vehicle Fuel Efficiency — Prediction")
    print("-" * 40)

    # Example prediction
    result = predict_mpg(
        cylinders=4,
        displacement=120.0,
        horsepower=80.0,
        weight=2500.0,
        acceleration=15.0,
        model_year=80,
        origin=1,
    )
    print(f"Predicted MPG: {result:.2f}")
