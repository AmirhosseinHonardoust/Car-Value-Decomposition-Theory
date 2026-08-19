from __future__ import annotations

import importlib

import pytest

pytest.importorskip("streamlit")


def test_app_module_imports_and_exposes_expected_functions() -> None:
    """Smoke test for app/app.py.

    Only checks that the module imports cleanly and exposes the functions
    the dashboard relies on -- it does not spin up a real Streamlit server
    or exercise widget callbacks, which would need a much heavier test
    harness (e.g. Streamlit's AppTest). This closes the "no automated
    coverage of app.py" gap noted in the README's Limitations section
    without adding a flaky, slow, or environment-fragile test.
    """
    app_module = importlib.import_module("app.app")

    for name in (
        "main",
        "load_data",
        "load_model",
        "load_r2_warning",
        "render_single_car_tab",
        "render_compare_tab",
        "render_market_tab",
    ):
        assert hasattr(app_module, name), f"app.app is missing expected attribute: {name}"
        assert callable(getattr(app_module, name))
