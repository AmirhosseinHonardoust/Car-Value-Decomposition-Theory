from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config, data_prep, features


def _predict_full_dataset() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load the clean dataset and model, and return (df, y, y_pred).

    Shared by group_price_bias and multi_group_bias_audit so the model is
    only loaded and scored once, even when auditing several group columns.
    """
    df = data_prep.load_clean()
    X, y = features.get_feature_matrix(df)

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    model = joblib.load(config.MODEL_PATH)

    y_pred = model.predict(X)
    return df, y, y_pred


def _bias_table(
    df: pd.DataFrame, group_col: str, y: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col} not found in dataframe.")

    tmp = pd.DataFrame(
        {
            group_col: df[group_col].values,
            "y_true": y.values,
            "y_pred": y_pred,
        }
    )

    grouped = (
        tmp.groupby(group_col)
        .agg(
            avg_true=("y_true", "mean"),
            avg_pred=("y_pred", "mean"),
            count=("y_true", "size"),
        )
        .reset_index()
    )
    grouped["avg_error"] = grouped["avg_pred"] - grouped["avg_true"]
    return grouped


def group_price_bias(group_col: str = config.COL_FUEL_TYPE) -> pd.DataFrame:
    """Compare average predicted vs actual price per group (fuel, seller type, etc.)."""
    df, y, y_pred = _predict_full_dataset()
    return _bias_table(df, group_col, y, y_pred)


def multi_group_bias_audit(
    group_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run group_price_bias across several categorical columns at once.

    Defaults to config.CATEGORICAL_FEATURES (brand, fuel, seller/condition,
    transmission) -- the same columns the model is trained on -- so a single
    call surfaces average predicted-vs-actual price bias across every
    modeled category, not just one at a time. This does not replace a real
    fairness audit (no intersectional analysis, no protected-attribute
    framing), but it removes the need to call group_price_bias once per
    column by hand.

    Returns a dict keyed by group column name, each value the same
    DataFrame shape group_price_bias returns for that column.
    """
    cols = group_cols if group_cols is not None else list(config.CATEGORICAL_FEATURES)

    df, y, y_pred = _predict_full_dataset()
    return {col: _bias_table(df, col, y, y_pred) for col in cols}

