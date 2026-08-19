from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest
else:
    # Runtime-only import gated behind importorskip so this file skips
    # cleanly (instead of an ImportError) if streamlit isn't installed --
    # mirrors the `pytest.importorskip("streamlit")` guard already used in
    # test_app_smoke.py.
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture
def app(raw_csv: Path) -> AppTest:
    """Run app/app.py end-to-end against the isolated tiny fixture dataset.

    Reuses the same isolated_paths/raw_csv fixtures the rest of the suite
    uses, so this never touches the real project's data/models/reports
    directories, and trains a real (tiny, fast) model the same way the app
    would on a fresh checkout -- exercising main(), load_data(),
    load_model(), and load_r2_warning() for real instead of only checking
    they exist, which is what the older test_app_smoke.py does.
    """
    # st.cache_data/st.cache_resource caches are process-global, keyed by
    # function identity + args -- not by which AppTest session is running.
    # Without clearing them, a later test's (args-free) load_model() call
    # can silently return an earlier test's cached Pipeline instead of
    # training + writing one to *this* test's isolated MODEL_PATH, which
    # then breaks decompose_value_for_row() (it reads straight from disk,
    # bypassing the cache) with a spurious "model not found". Clearing
    # before each run keeps every test's app session fully independent.
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_loads_without_exception(app: AppTest) -> None:
    assert not app.exception
    assert [t.value for t in app.title] == ["Car-Value-Decoding-Engine"]
    assert len(app.tabs) == 3


def test_app_shows_r2_warning_banner(app: AppTest) -> None:
    """load_r2_warning() + describe_r2() output should render as a warning banner."""
    assert len(app.warning) == 1
    assert "explainability-method demo" in app.warning[0].value
    assert "R^2" in app.warning[0].value


def test_single_car_decoder_tab_runs_and_reconstructs(app: AppTest) -> None:
    """Clicking 'Decode Car Value' should render a metric and a JSON decomposition."""
    decode_button = next(b for b in app.button if b.label == "Decode Car Value")
    result = decode_button.click().run()

    assert not result.exception
    assert len(result.metric) == 1
    # st.metric shows the formatted final_prediction; just check it parses as a number.
    float(result.metric[0].value)

    assert len(result.json) == 1
    # st.json() elements expose their rendered content as a JSON string.
    dec = json.loads(result.json[0].value)
    assert dec["reconstruction_error"] == pytest.approx(0.0, abs=1e-6)
    contrib_keys = [k for k in dec if k.startswith("contrib_")]
    assert len(contrib_keys) == 7  # one per config.GROUP_DEFINITIONS entry


def test_compare_tab_runs_and_shows_both_cars(app: AppTest) -> None:
    """Clicking 'Compare' should render specs + decomposition JSON for both cars."""
    compare_button = next(b for b in app.button if b.label == "Compare")
    result = compare_button.click().run()

    assert not result.exception
    # 2 spec dicts + 2 decomposition dicts = 4 st.json() calls.
    assert len(result.json) == 4


def test_market_tab_renders_without_exception(app: AppTest) -> None:
    """Market Explorer tab has no button; just verify its static content renders."""
    assert not app.exception
    market_tab = app.tabs[2]
    assert "Market Explorer" in [s.value for s in market_tab.subheader]
