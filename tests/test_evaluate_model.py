from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config, evaluate_model, train_model


def test_evaluate_model_raises_when_no_model(raw_csv: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_model.evaluate_model()


def test_evaluate_model_writes_metrics_after_training(raw_csv: Path) -> None:
    train_model.train_price_model()
    evaluate_model.evaluate_model()

    metrics_path = config.REPORTS_METRICS_DIR / "regression_metrics_eval.json"
    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    assert set(metrics) == {"mae", "rmse", "r2"}
