from __future__ import annotations

from pathlib import Path

import pytest

from src import config, explainability, train_model


@pytest.fixture
def trained(raw_csv: Path) -> Path:
    train_model.train_price_model()
    return raw_csv


def test_global_permutation_importance_covers_all_features(trained: Path) -> None:
    importance_df = explainability.global_permutation_importance(n_repeats=2)
    assert set(importance_df["feature"]) == set(config.FEATURE_COLUMNS)
    assert {"feature", "importance_mean", "importance_std"} <= set(importance_df.columns)


def test_global_permutation_importance_is_sorted_descending(trained: Path) -> None:
    importance_df = explainability.global_permutation_importance(n_repeats=2)
    means = importance_df["importance_mean"].tolist()
    assert means == sorted(means, reverse=True)


def test_global_permutation_importance_raises_when_no_model(raw_csv: Path) -> None:
    with pytest.raises(FileNotFoundError):
        explainability.global_permutation_importance(n_repeats=2)
