from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def add_brand_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Brand' column extracted from the car name, if not already present."""
    df = df.copy()
    if config.COL_BRAND in df.columns:
        return df
    if config.COL_CAR_NAME in df.columns:
        df[config.COL_BRAND] = df[config.COL_CAR_NAME].astype(str).str.split().str[0]
    else:
        df[config.COL_BRAND] = "Unknown"
    return df


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y for modeling."""
    df = add_brand_column(df)

    missing_cols = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.COL_SELLING_PRICE].copy()
    return X, y


def get_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer to preprocess numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor
