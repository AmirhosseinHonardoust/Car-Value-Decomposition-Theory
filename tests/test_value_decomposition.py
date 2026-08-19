from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config, data_prep, train_model, value_decomposition


def assert_decompositions_close(a: dict[str, float], b: dict[str, float]) -> None:
    """Compare two decomposition results allowing float tolerance.

    RandomForestRegressor is fit with n_jobs=-1, so repeated calls to
    model.predict() can sum per-tree outputs in a different thread order
    and differ by a few ULPs even for identical inputs -- exact `==` on
    these dicts is flaky. This is float-precision noise, not a
    determinism bug in decompose_value_for_row's own seeding.
    """
    assert set(a) == set(b)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-6), f"mismatch on {key}"


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


def test_decompose_value_default_is_unchanged_single_ordering(trained: Path) -> None:
    """n_orderings defaults to 1, which must reproduce the exact prior behavior."""
    df = data_prep.load_clean()
    row = df.iloc[0]

    default_call = value_decomposition.decompose_value_for_row(row)
    explicit_call = value_decomposition.decompose_value_for_row(row, n_orderings=1)

    assert_decompositions_close(default_call, explicit_call)


def test_decompose_value_multi_ordering_still_reconstructs_exactly(trained: Path) -> None:
    """Averaging over several random orderings must still telescope to the
    same final_prediction, since each individual ordering's contributions
    sum to the same base->final delta regardless of order."""
    df = data_prep.load_clean()
    row = df.iloc[0]

    result = value_decomposition.decompose_value_for_row(row, n_orderings=8, random_state=1)

    assert result["reconstruction_error"] == pytest.approx(0.0, abs=1e-6)


def test_decompose_value_multi_ordering_covers_all_groups(trained: Path) -> None:
    df = data_prep.load_clean()
    row = df.iloc[0]

    result = value_decomposition.decompose_value_for_row(row, n_orderings=5, random_state=2)

    contrib_keys = {k for k in result if k.startswith("contrib_")}
    assert contrib_keys == {f"contrib_{name}" for name in config.GROUP_DEFINITIONS}


def test_decompose_value_multi_ordering_is_reproducible_with_same_seed(trained: Path) -> None:
    df = data_prep.load_clean()
    row = df.iloc[0]

    result_a = value_decomposition.decompose_value_for_row(row, n_orderings=6, random_state=7)
    result_b = value_decomposition.decompose_value_for_row(row, n_orderings=6, random_state=7)

    assert_decompositions_close(result_a, result_b)


def test_decompose_value_multi_ordering_smooths_the_order_dependence(trained: Path) -> None:
    """Averaging over many orderings should land closer to a stable value
    than any single fixed ordering does -- that's the whole point of the
    approximation. We check it produces a different (and finite) answer
    than the single-ordering default, without asserting a specific value."""
    df = data_prep.load_clean()
    row = df.iloc[0]

    single = value_decomposition.decompose_value_for_row(row)
    averaged = value_decomposition.decompose_value_for_row(row, n_orderings=25, random_state=3)

    contrib_keys = [k for k in single if k.startswith("contrib_")]
    assert all(averaged[k] == pytest.approx(averaged[k]) for k in contrib_keys)  # finite, no NaN
    assert any(
        single[k] != pytest.approx(averaged[k], abs=1e-6) for k in contrib_keys
    ), "Expected averaging over orderings to change at least one contribution"


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
