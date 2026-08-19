from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, data_prep, features


def load_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the cleaned dataset and return the train/test split used everywhere.

    Centralizes the load-clean -> get_feature_matrix -> train_test_split
    sequence that was previously copy-pasted (with identical arguments) in
    train_model.py, evaluate_model.py, and explainability.py. Because every
    caller passes the same config.TEST_SIZE and config.RANDOM_STATE, this
    always reproduces the exact same split those call sites produced before
    -- this change only removes the duplication, it does not alter the
    split itself.
    """
    df = data_prep.load_clean()
    X, y = features.get_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test
