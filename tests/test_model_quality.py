from __future__ import annotations

from src.model_quality import describe_r2


def test_describe_r2_meaningful_signal() -> None:
    msg = describe_r2(0.82)
    assert "0.820" in msg
    assert "meaningful share" in msg


def test_describe_r2_weak_signal() -> None:
    msg = describe_r2(0.2)
    assert "0.200" in msg
    assert "very little price variance" in msg
    assert "WORSE" not in msg


def test_describe_r2_boundary_is_meaningful_not_weak() -> None:
    # 0.5 is the boundary; the function documents >= 0.5 as "meaningful".
    msg = describe_r2(0.5)
    assert "meaningful share" in msg


def test_describe_r2_zero_is_weak_not_negative() -> None:
    # 0.0 is the boundary; the function documents >= 0.0 as "weak", not
    # the harsher negative-R^2 message.
    msg = describe_r2(0.0)
    assert "very little price variance" in msg
    assert "WORSE" not in msg


def test_describe_r2_negative_signal() -> None:
    msg = describe_r2(-0.065)
    assert "-0.065" in msg
    assert "WORSE" in msg
    assert "demonstration of the method" in msg
