from __future__ import annotations

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src import config, data_prep, features


def test_add_brand_column_noop_when_present(tiny_raw_df: pd.DataFrame) -> None:
    result = features.add_brand_column(tiny_raw_df)
    pd.testing.assert_series_equal(result[config.COL_BRAND], tiny_raw_df[config.COL_BRAND])


def test_add_brand_column_derives_from_car_name() -> None:
    df = pd.DataFrame({config.COL_CAR_NAME: ["Model X", "5 Series"]})
    result = features.add_brand_column(df)
    assert list(result[config.COL_BRAND]) == ["Model", "5"]


def test_get_feature_matrix_shape(tiny_raw_df: pd.DataFrame) -> None:
    cleaned = data_prep.clean_data(tiny_raw_df)
    X, y = features.get_feature_matrix(cleaned)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert len(X) == len(y) == len(cleaned)


def test_get_feature_matrix_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"unrelated_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        features.get_feature_matrix(df)


def test_get_preprocessor_returns_column_transformer() -> None:
    assert isinstance(features.get_preprocessor(), ColumnTransformer)


def test_preprocessor_fits_and_transforms_tiny_data(tiny_raw_df: pd.DataFrame) -> None:
    cleaned = data_prep.clean_data(tiny_raw_df)
    X, _y = features.get_feature_matrix(cleaned)
    preprocessor = features.get_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
