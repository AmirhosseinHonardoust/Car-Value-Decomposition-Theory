from __future__ import annotations

import json
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from . import config


def load_model_and_baseline():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    if not config.BASELINE_STATS_PATH.exists():
        raise FileNotFoundError(f"Baseline stats not found at {config.BASELINE_STATS_PATH}")

    model = joblib.load(config.MODEL_PATH)
    with config.BASELINE_STATS_PATH.open("r", encoding="utf-8") as f:
        baseline_stats = json.load(f)

    baseline_features: Dict[str, object] = baseline_stats["baseline_features"]
    groups: Dict[str, List[str]] = baseline_stats["groups"]
    return model, baseline_features, groups


def build_feature_row_from_baseline(
    baseline_features: Dict[str, object]
) -> pd.DataFrame:
    """Create a one-row DataFrame from baseline feature values."""
    data = {col: baseline_features.get(col) for col in config.FEATURE_COLUMNS}
    return pd.DataFrame([data])


def build_feature_row_from_sample(
    sample: pd.Series, baseline_features: Dict[str, object]
) -> pd.DataFrame:
    """Create a one-row DataFrame from a sample row.

    If a feature is missing or NaN in the sample, we fall back to the
    baseline value so that the encoder never sees NaN categories.
    """
    import pandas as pd  # ensure pd.isna is available

    data = {}
    for col in config.FEATURE_COLUMNS:
        if col in sample.index and not pd.isna(sample[col]):
            data[col] = sample[col]
        else:
            data[col] = baseline_features.get(col)
    return pd.DataFrame([data])



def decompose_value_for_row(row: pd.Series) -> Dict[str, float]:
    """Decompose the predicted price for a given car into value components.

    The algorithm:
      - Start from a baseline feature row.
      - For each group (brand, age, mileage, ...), swap that group's
        features from the row into the baseline, measure the delta.
      - The sum of contributions + base value ~= predicted price.
    """
    model, baseline_features, groups = load_model_and_baseline()

    baseline_df = build_feature_row_from_baseline(baseline_features)
    sample_df = build_feature_row_from_sample(row, baseline_features)

    # Ensure we respect the same column ordering
    baseline_df = baseline_df[config.FEATURE_COLUMNS]
    sample_df = sample_df[config.FEATURE_COLUMNS]

    # Base prediction from baseline
    base_value = float(model.predict(baseline_df)[0])

    current_df = baseline_df.copy()
    contributions: Dict[str, float] = {}
    current_pred = base_value

    for group_name, cols in groups.items():
        # If any of these columns are not in our feature set, skip
        valid_cols = [c for c in cols if c in config.FEATURE_COLUMNS]
        if not valid_cols:
            continue

        next_df = current_df.copy()
        for c in valid_cols:
            next_df[c] = sample_df[c].values[0]

        next_pred = float(model.predict(next_df)[0])
        contrib = next_pred - current_pred
        contributions[group_name] = contrib

        current_df = next_df
        current_pred = next_pred

    final_pred = float(model.predict(sample_df)[0])

    # Sanity check difference between reconstructed and actual
    reconstructed = base_value + sum(contributions.values())
    reconstruction_error = final_pred - reconstructed

    result = {
        "base_value": base_value,
        "final_prediction": final_pred,
        "reconstruction_error": reconstruction_error,
    }
    for k, v in contributions.items():
        result[f"contrib_{k}"] = v

    return result
