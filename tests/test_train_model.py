from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config, data_prep, features, train_model


def test_compute_baseline_stats_numeric_and_categorical(tiny_raw_df: pd.DataFrame) -> None:
    cleaned = data_prep.clean_data(tiny_raw_df)
    X, _y = features.get_feature_matrix(cleaned)
    baseline = train_model.compute_baseline_stats(X)

    for col in config.NUMERIC_FEATURES:
        assert isinstance(baseline[col], float)
    for col in config.CATEGORICAL_FEATURES:
        assert isinstance(baseline[col], str)


def test_train_price_model_writes_expected_files(raw_csv: Path) -> None:
    train_model.train_price_model()

    assert config.MODEL_PATH.exists()
    assert config.BASELINE_STATS_PATH.exists()
    metrics_path = config.REPORTS_METRICS_DIR / "regression_metrics.json"
    assert metrics_path.exists()

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())


def test_train_price_model_baseline_stats_has_all_groups(raw_csv: Path) -> None:
    train_model.train_price_model()
    with config.BASELINE_STATS_PATH.open(encoding="utf-8") as f:
        baseline_stats = json.load(f)

    assert set(baseline_stats["groups"]) == set(config.GROUP_DEFINITIONS)
    assert set(baseline_stats["baseline_features"]) == set(config.FEATURE_COLUMNS)
