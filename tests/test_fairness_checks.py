from __future__ import annotations

from pathlib import Path

import pytest

from src import config, fairness_checks, train_model


@pytest.fixture
def trained(raw_csv: Path) -> Path:
    train_model.train_price_model()
    return raw_csv


def test_group_price_bias_default_arg_matches_a_real_column(trained: Path) -> None:
    """Regression test for a bug found during the repo audit.

    group_price_bias's default group_col used to be the literal string
    "Fuel_Type" (underscore), while the dataset's actual column -- defined
    once in config.COL_FUEL_TYPE -- is "Fuel Type" (space). Calling
    group_price_bias() with no arguments therefore always raised
    ValueError. The default now references config.COL_FUEL_TYPE directly
    so it can't drift out of sync with the real column name again.
    """
    result = fairness_checks.group_price_bias()
    assert config.COL_FUEL_TYPE in result.columns
    assert {"avg_true", "avg_pred", "count", "avg_error"} <= set(result.columns)


def test_group_price_bias_raises_on_unknown_column(trained: Path) -> None:
    with pytest.raises(ValueError, match="not found in dataframe"):
        fairness_checks.group_price_bias(group_col="not_a_real_column")


def test_group_price_bias_raises_when_no_model(raw_csv: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fairness_checks.group_price_bias()


def test_group_price_bias_groups_cover_all_categories(trained: Path) -> None:
    from src import data_prep

    df = data_prep.load_clean()
    result = fairness_checks.group_price_bias(group_col=config.COL_TRANSMISSION)
    assert set(result[config.COL_TRANSMISSION]) == set(df[config.COL_TRANSMISSION].unique())
