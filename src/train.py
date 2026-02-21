"""
Model training module for Vehicle Fuel Efficiency prediction.
Trains multiple regression models, compares them, and saves the best one.
"""

import os
import sys
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import xgboost as xgb

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import run_full_pipeline

# ---------- Paths ----------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "auto-mpg.data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

N_FOLDS = 5
ALPHAS = np.logspace(-4, -0.5, 30)


# ---------- Averaging Ensemble ----------
class AveragingModels:
    """Simple averaging ensemble of regression models."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(m) for m in self.models]
        for m in self.models_:
            m.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return np.mean(preds, axis=1)


# ---------- Individual trainers ----------
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train):
    ridge = Ridge(random_state=42, max_iter=10000)
    params = {"alpha": ALPHAS}
    clf = GridSearchCV(ridge, [params], cv=N_FOLDS,
                       scoring="neg_mean_squared_error", refit=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def train_lasso(X_train, y_train):
    lasso = Lasso(random_state=42, max_iter=10000)
    params = {"alpha": ALPHAS}
    clf = GridSearchCV(lasso, [params], cv=N_FOLDS,
                       scoring="neg_mean_squared_error", refit=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def train_elasticnet(X_train, y_train):
    enet = ElasticNet(random_state=42, max_iter=10000)
    params = {"alpha": ALPHAS, "l1_ratio": np.arange(0, 1.0, 0.05)}
    clf = GridSearchCV(enet, params, cv=N_FOLDS,
                       scoring="neg_mean_squared_error", refit=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def train_xgboost(X_train, y_train):
    params = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.03, 0.05, 0.07],
        "max_depth": [5, 6, 7],
        "min_child_weight": [4],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
    }
    model = xgb.XGBRegressor(objective="reg:squarederror", nthread=4, verbosity=0)
    clf = GridSearchCV(model, params, cv=N_FOLDS,
                       scoring="neg_mean_squared_error", refit=True,
                       n_jobs=5, verbose=1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_


# ---------- Evaluation ----------
def evaluate(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name:20s} | MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")
    return {"name": name, "mse": mse, "mae": mae, "r2": r2}


# ---------- Main ----------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("  Vehicle Fuel Efficiency — Model Training Pipeline")
    print("=" * 65)

    # 1. Preprocessing
    print("\n[1/4] Running preprocessing pipeline …")
    X_train, X_test, y_train, y_test, scaler, feature_columns = run_full_pipeline(
        DATA_PATH, scaler_save_path=SCALER_PATH
    )
    print(f"      Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"      Features  : {X_train.shape[1]}")

    # 2. Train models
    print("\n[2/4] Training models …")
    results = []

    lr = train_linear_regression(X_train, y_train)
    results.append(evaluate(lr, X_test, y_test, "Linear Regression"))

    ridge = train_ridge(X_train, y_train)
    results.append(evaluate(ridge, X_test, y_test, "Ridge"))

    lasso = train_lasso(X_train, y_train)
    results.append(evaluate(lasso, X_test, y_test, "Lasso"))

    enet = train_elasticnet(X_train, y_train)
    results.append(evaluate(enet, X_test, y_test, "ElasticNet"))

    xgb_model = train_xgboost(X_train, y_train)
    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost"))

    # 3. Averaging ensemble (XGBoost + Lasso)
    print("\n[3/4] Training ensemble (XGBoost + Lasso) …")
    avg_model = AveragingModels(models=(xgb_model, lasso))
    avg_model.fit(X_train, y_train)
    results.append(evaluate(avg_model, X_test, y_test, "Averaged (XGB+Lasso)"))

    # 4. Select & save best model
    best = min(results, key=lambda r: r["mse"])
    print(f"\n[4/4] Best model: {best['name']}  (MSE={best['mse']:.6f})")

    # Map name → model object
    model_map = {
        "Linear Regression": lr,
        "Ridge": ridge,
        "Lasso": lasso,
        "ElasticNet": enet,
        "XGBoost": xgb_model,
        "Averaged (XGB+Lasso)": avg_model,
    }
    best_model = model_map[best["name"]]
    joblib.dump(best_model, MODEL_PATH)
    print(f"      Model saved → {MODEL_PATH}")

    # Save metadata
    metadata = {
        "best_model": best["name"],
        "mse": best["mse"],
        "mae": best["mae"],
        "r2": best["r2"],
        "feature_columns": feature_columns,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"      Metadata saved → {METADATA_PATH}")

    print("\n" + "=" * 65)
    print("  Training complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
