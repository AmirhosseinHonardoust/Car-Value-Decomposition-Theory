from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config, data_prep


def test_load_raw_missing_file_raises(isolated_paths: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data_prep.load_raw()


def test_load_raw_reads_csv(raw_csv: Path, tiny_raw_df: pd.DataFrame) -> None:
    df = data_prep.load_raw()
    assert len(df) == len(tiny_raw_df)
    assert list(df.columns) == list(tiny_raw_df.columns)


def test_clean_data_adds_engineered_columns(tiny_raw_df: pd.DataFrame) -> None:
    cleaned = data_prep.clean_data(tiny_raw_df)
    assert config.COL_CAR_AGE in cleaned.columns
    assert config.COL_KM_PER_YEAR in cleaned.columns
    assert not cleaned[config.FEATURE_COLUMNS].isna().any().any()


def test_clean_data_strips_whitespace(tiny_raw_df: pd.DataFrame) -> None:
    cleaned = data_prep.clean_data(tiny_raw_df)
    assert (cleaned[config.COL_CAR_NAME] == cleaned[config.COL_CAR_NAME].str.strip()).all()


def test_clean_data_clamps_nonpositive_car_age(tiny_raw_df: pd.DataFrame) -> None:
    # tiny_raw_df includes Year == 2020, the hardcoded reference year,
    # which would otherwise produce car_age == 0.
    cleaned = data_prep.clean_data(tiny_raw_df)
    assert (cleaned[config.COL_CAR_AGE] >= 1).all()


def test_clean_data_drops_rows_missing_required_columns(tiny_raw_df: pd.DataFrame) -> None:
    broken = tiny_raw_df.copy()
    broken.loc[0, config.COL_FUEL_TYPE] = None
    cleaned = data_prep.clean_data(broken)
    assert len(cleaned) == len(tiny_raw_df) - 1


def test_load_clean_caches_to_parquet(raw_csv: Path) -> None:
    assert not (config.DATA_PROCESSED_DIR / "car_price_clean.parquet").exists()
    df_first = data_prep.load_clean()
    assert (config.DATA_PROCESSED_DIR / "car_price_clean.parquet").exists()

    df_cached = data_prep.load_clean()
    pd.testing.assert_frame_equal(df_first.reset_index(drop=True), df_cached.reset_index(drop=True))
