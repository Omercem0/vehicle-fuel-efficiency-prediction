"""
Data preprocessing module for Vehicle Fuel Efficiency prediction.
Handles data loading, cleaning, outlier removal, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Column names for the auto-mpg dataset
COLUMN_NAMES = [
    "MPG", "Cylinders", "Displacement", "Horsepower",
    "Weight", "Acceleration", "Model Year", "Origin"
]

# Outlier threshold multiplier for IQR method
OUTLIER_THRESHOLD = 2

# Test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw auto-mpg data from whitespace-separated file."""
    data = pd.read_csv(
        filepath,
        names=COLUMN_NAMES,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )
    data = data.rename(columns={"MPG": "target"})
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Horsepower values with column mean."""
    df = data.copy()
    df["Horsepower"] = df["Horsepower"].fillna(df["Horsepower"].mean())
    return df


def remove_outliers(data: pd.DataFrame, column: str, threshold: float = OUTLIER_THRESHOLD) -> pd.DataFrame:
    """Remove outliers from a column using IQR method."""
    df = data.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = (df[column] > lower) & (df[column] < upper)
    return df[mask]


def apply_log_transform(data: pd.DataFrame, column: str = "target") -> pd.DataFrame:
    """Apply log1p transformation to reduce skewness."""
    df = data.copy()
    df[column] = np.log1p(df[column])
    return df


def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Cylinders and Origin columns."""
    df = data.copy()
    df["Cylinders"] = df["Cylinders"].astype(str)
    df["Origin"] = df["Origin"].astype(str)
    df = pd.get_dummies(df)
    return df


def split_data(data: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split data into train/test sets."""
    X = data.drop("target", axis=1)
    y = data["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test, scaler_path: str = None):
    """
    Scale features using RobustScaler.
    Optionally saves the scaler to disk for inference.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def run_full_pipeline(raw_data_path: str, scaler_save_path: str = None):
    """
    Execute the complete preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, scaler, and feature_columns.
    """
    # 1. Load
    data = load_data(raw_data_path)

    # 2. Missing values
    data = handle_missing_values(data)

    # 3. Outlier removal
    data = remove_outliers(data, "Horsepower")
    data = remove_outliers(data, "Acceleration")

    # 4. Log-transform target
    data = apply_log_transform(data, "target")

    # 5. One-hot encoding
    data = encode_categoricals(data)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = split_data(data)

    # Save feature columns for prediction-time alignment
    feature_columns = X_train.columns.tolist()

    # 7. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_path=scaler_save_path
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
