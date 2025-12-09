from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config, data_prep, features


def compute_baseline_stats(X) -> dict:
    """Compute baseline feature values for value decomposition.

    - Numeric features: mean
    - Categorical features: mode
    """
    import pandas as pd

    X_df = pd.DataFrame(X, columns=config.FEATURE_COLUMNS)

    baseline = {}
    for col in config.NUMERIC_FEATURES:
        baseline[col] = float(X_df[col].mean())

    for col in config.CATEGORICAL_FEATURES:
        mode_val = X_df[col].mode(dropna=True)
        baseline[col] = str(mode_val.iloc[0]) if not mode_val.empty else "Unknown"

    return baseline


def train_price_model() -> None:
    df = data_prep.load_clean()
    X, y = features.get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    preprocessor = features.get_preprocessor()

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # Save model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    # Save baseline stats for value decomposition
    baseline_stats = {
        "baseline_features": compute_baseline_stats(X_train),
        "groups": config.GROUP_DEFINITIONS,
    }
    with config.BASELINE_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    # Save metrics
    config.REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_METRICS_DIR / "regression_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {config.MODEL_PATH}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    train_price_model()
