from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config


@pytest.fixture
def tiny_raw_df() -> pd.DataFrame:
    """A small, hand-built dataset with the same schema as the real CSV.

    Deliberately tiny (kept out of the real 2500-row dataset) so the full
    train/evaluate/decompose pipeline runs in well under a second in CI,
    while still covering: multiple brands/fuel types/transmissions/
    conditions, a whitespace-padded string, and a Year == reference_year
    row (car_age clamps to 1 instead of 0).
    """
    return pd.DataFrame(
        {
            "Car ID": list(range(1, 25)),
            "Brand": (["Tesla", "BMW", "Audi", "Toyota"] * 6),
            "Year": [2020, 2015, 2018, 2010, 2022, 2012] * 4,
            "Engine Size": [1.6, 2.0, 2.5, 3.0, 1.8, 4.4] * 4,
            "Fuel Type": (["Petrol", "Diesel", "Electric", "Hybrid"] * 6),
            "Transmission": (["Manual", "Automatic"] * 12),
            "Mileage": [10000, 50000, 90000, 150000, 30000, 200000] * 4,
            "Condition": (["New", "Used", "Like New"] * 8),
            "Price": [
                20000.0,
                35000.0,
                42000.0,
                18000.0,
                55000.0,
                12000.0,
            ]
            * 4,
            "Model": (["Model X ", "5 Series", "A4", "Corolla"] * 6),
        }
    )


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all of config's filesystem paths under a throwaway tmp dir.

    Keeps every test fully isolated from the real repo's data/models/reports
    directories -- no test ever reads or writes real project files.
    """
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path / "data" / "raw" / "car_price_prediction.csv")
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "REPORTS_METRICS_DIR", tmp_path / "reports" / "metrics")
    monkeypatch.setattr(config, "REPORTS_FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "price_model.joblib")
    monkeypatch.setattr(config, "BASELINE_STATS_PATH", tmp_path / "models" / "baseline_stats.json")
    return tmp_path


@pytest.fixture
def raw_csv(isolated_paths: Path, tiny_raw_df: pd.DataFrame) -> Path:
    """Write tiny_raw_df to the (isolated) location load_raw() reads from."""
    config.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    tiny_raw_df.to_csv(config.DATA_RAW, index=False)
    return config.DATA_RAW
