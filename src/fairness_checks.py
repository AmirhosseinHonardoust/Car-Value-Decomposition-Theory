from __future__ import annotations

import joblib
import pandas as pd

from . import config, data_prep, features


def group_price_bias(group_col: str = "Fuel_Type") -> pd.DataFrame:
    """Compare average predicted vs actual price per group (fuel, seller type, etc.)."""
    df = data_prep.load_clean()
    X, y = features.get_feature_matrix(df)

    if group_col not in df.columns:
        raise ValueError(f"Column {group_col} not found in dataframe.")

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    model = joblib.load(config.MODEL_PATH)

    y_pred = model.predict(X)

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
