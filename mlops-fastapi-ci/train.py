#!/usr/bin/env python3
"""Train a tiny regression model, log it with MLflow, and export model.pkl for the API."""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "model.pkl"


def main() -> None:
    rng = np.random.default_rng(42)
    x_train = np.linspace(1, 20, 40).reshape(-1, 1)
    y_train = (2.0 * x_train.ravel()) + rng.normal(0, 0.1, size=x_train.shape[0])
    x_test = np.array([[21.0], [22.0], [25.0]])
    y_test = 2.0 * x_test.ravel()

    mlflow.set_experiment("mlops-fastapi-ci")
    n_estimators = 100

    with mlflow.start_run(run_name="random-forest-baseline"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForestRegressor")

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        mse = float(mean_squared_error(y_test, preds))
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, artifact_path="model")
        with MODEL_PATH.open("wb") as f:
            pickle.dump(model, f)

        print(f"MSE={mse:.6f}")
        print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
