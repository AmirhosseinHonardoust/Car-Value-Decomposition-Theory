from __future__ import annotations

import json

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .model_quality import describe_r2
from .split_utils import load_train_test_split


def evaluate_model() -> None:
    _X_train, X_test, _y_train, y_test = load_train_test_split()

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")

    model = joblib.load(config.MODEL_PATH)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(mse**0.5)
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    config.REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_METRICS_DIR / "regression_metrics_eval.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:", metrics)
    print(describe_r2(r2))


if __name__ == "__main__":
    evaluate_model()
