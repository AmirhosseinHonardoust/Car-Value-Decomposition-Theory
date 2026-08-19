from __future__ import annotations

import json
import random

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config


def load_model_and_baseline() -> tuple[Pipeline, dict[str, object], dict[str, list[str]]]:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    if not config.BASELINE_STATS_PATH.exists():
        raise FileNotFoundError(f"Baseline stats not found at {config.BASELINE_STATS_PATH}")

    model = joblib.load(config.MODEL_PATH)
    with config.BASELINE_STATS_PATH.open("r", encoding="utf-8") as f:
        baseline_stats = json.load(f)

    baseline_features: dict[str, object] = baseline_stats["baseline_features"]
    groups: dict[str, list[str]] = baseline_stats["groups"]
    return model, baseline_features, groups


def build_feature_row_from_baseline(baseline_features: dict[str, object]) -> pd.DataFrame:
    """Create a one-row DataFrame from baseline feature values."""
    data = {col: baseline_features.get(col) for col in config.FEATURE_COLUMNS}
    return pd.DataFrame([data])


def build_feature_row_from_sample(
    sample: pd.Series, baseline_features: dict[str, object]
) -> pd.DataFrame:
    """Create a one-row DataFrame from a sample row.

    If a feature is missing or NaN in the sample, we fall back to the
    baseline value so that the encoder never sees NaN categories.
    """
    data = {}
    for col in config.FEATURE_COLUMNS:
        if col in sample.index and not pd.isna(sample[col]):
            data[col] = sample[col]
        else:
            data[col] = baseline_features.get(col)
    return pd.DataFrame([data])


def _swap_along_order(
    model: Pipeline,
    baseline_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    base_value: float,
    groups: dict[str, list[str]],
    order: list[str],
) -> dict[str, float]:
    """Run one sequential group-swap pass along a given group order.

    Returns per-group contributions for that single ordering. The sum of
    the returned contributions always equals final_prediction - base_value
    (telescoping construction), regardless of the order used.
    """
    current_df = baseline_df.copy()
    contributions: dict[str, float] = {}
    current_pred = base_value

    for group_name in order:
        cols = groups[group_name]
        valid_cols = [c for c in cols if c in config.FEATURE_COLUMNS]
        if not valid_cols:
            continue

        next_df = current_df.copy()
        for c in valid_cols:
            next_df[c] = sample_df[c].values[0]

        next_pred = float(model.predict(next_df)[0])
        contributions[group_name] = next_pred - current_pred

        current_df = next_df
        current_pred = next_pred

    return contributions


def decompose_value_for_row(
    row: pd.Series,
    *,
    n_orderings: int = 1,
    random_state: int | None = None,
) -> dict[str, float]:
    """Decompose the predicted price for a given car into value components.

    The default algorithm (n_orderings=1, unchanged from before):
      - Start from a baseline feature row.
      - For each group (brand, age, mileage, ...), swap that group's
        features from the row into the baseline, measure the delta, using
        whatever group order baseline_stats.json happens to store.
      - The sum of contributions + base value == the final prediction
        exactly, but the per-group split is order-dependent (see README
        and tests/test_value_decomposition.py).

    Passing n_orderings > 1 instead averages the per-group contributions
    over that many random group orderings -- a cheap Monte Carlo
    approximation of a true Shapley-value attribution, which averages over
    *every* possible ordering. This is opt-in and does not change the
    default single-ordering behavior or its output.
    """
    model, baseline_features, groups = load_model_and_baseline()

    baseline_df = build_feature_row_from_baseline(baseline_features)
    sample_df = build_feature_row_from_sample(row, baseline_features)

    # Ensure we respect the same column ordering
    baseline_df = baseline_df[config.FEATURE_COLUMNS]
    sample_df = sample_df[config.FEATURE_COLUMNS]

    # Base prediction from baseline
    base_value = float(model.predict(baseline_df)[0])

    if n_orderings <= 1:
        orderings = [list(groups.keys())]
    else:
        rng = random.Random(random_state if random_state is not None else config.RANDOM_STATE)
        group_names = list(groups.keys())
        orderings = []
        for _ in range(n_orderings):
            shuffled = group_names.copy()
            rng.shuffle(shuffled)
            orderings.append(shuffled)

    per_ordering_contributions = [
        _swap_along_order(model, baseline_df, sample_df, base_value, groups, order)
        for order in orderings
    ]

    contributions: dict[str, float] = {}
    for group_name in groups:
        values = [c[group_name] for c in per_ordering_contributions if group_name in c]
        if values:
            contributions[group_name] = sum(values) / len(values)

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
