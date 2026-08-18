from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config, data_prep, train_model, value_decomposition


@pytest.fixture
def trained(raw_csv: Path) -> Path:
    train_model.train_price_model()
    return raw_csv


def test_decompose_value_reconstructs_final_prediction(trained: Path) -> None:
    df = data_prep.load_clean()
    row = df.iloc[0]
    result = value_decomposition.decompose_value_for_row(row)

    assert set(result) >= {"base_value", "final_prediction", "reconstruction_error"}
    # The sequential-swap construction guarantees this holds by
    # construction (base + sum(deltas) telescopes to final_prediction);
    # it's a mathematical identity of the method, not evidence the
    # explanation is a good one -- see the order-dependence test below.
    assert result["reconstruction_error"] == pytest.approx(0.0, abs=1e-6)


def test_decompose_value_has_one_contribution_per_group(trained: Path) -> None:
    df = data_prep.load_clean()
    row = df.iloc[0]
    result = value_decomposition.decompose_value_for_row(row)

    contrib_keys = {k for k in result if k.startswith("contrib_")}
    assert contrib_keys == {f"contrib_{name}" for name in config.GROUP_DEFINITIONS}


def test_contributions_are_order_dependent_known_limitation(trained: Path) -> None:
    """Documents a known limitation surfaced during the repo audit.

    decompose_value_for_row attributes each group's contribution along a
    single fixed path -- the order the "groups" dict was serialized in
    inside baseline_stats.json at train time -- not averaged over
    orderings like Shapley values. That makes contributions order-
    dependent even though the README describes the method as
    deterministic and stable. This test doesn't assert a "correct" value
    -- there isn't one under this method -- it pins down that the
    order-dependence exists, so it doesn't silently disappear (or get
    silently "fixed" by an unrelated change) without anyone noticing.
    Fixing this properly (e.g. averaging over several group orderings) is
    tracked as a follow-up PR.
    """
    df = data_prep.load_clean()
    row = df.iloc[0]

    forward = value_decomposition.decompose_value_for_row(row)

    # Rewrite baseline_stats.json with the groups in reversed order --
    # this is what decompose_value_for_row actually reads its path from
    # (see load_model_and_baseline), not config.GROUP_DEFINITIONS directly.
    with config.BASELINE_STATS_PATH.open(encoding="utf-8") as f:
        baseline_stats = json.load(f)
    baseline_stats["groups"] = dict(reversed(list(baseline_stats["groups"].items())))
    with config.BASELINE_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline_stats, f)

    reversed_result = value_decomposition.decompose_value_for_row(row)

    contrib_keys = [k for k in forward if k.startswith("contrib_")]
    differing = [
        k for k in contrib_keys if forward[k] != pytest.approx(reversed_result[k], abs=1e-6)
    ]
    assert differing, (
        "Expected at least one contribution to change under a different group "
        "order -- if this now fails, the order-dependence issue documented in "
        "the audit may have been fixed; update this test to reflect that."
    )
