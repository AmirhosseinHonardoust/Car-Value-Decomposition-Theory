from __future__ import annotations

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config
from .split_utils import load_train_test_split


def global_permutation_importance(n_repeats: int = 10) -> pd.DataFrame:
    """Compute permutation feature importance on the held-out test set."""
    _X_train, X_test, _y_train, y_test = load_train_test_split()

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
    model = joblib.load(config.MODEL_PATH)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": config.FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df
