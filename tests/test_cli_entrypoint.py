from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import entry_points

import pytest


def test_console_script_is_registered_to_cli_main() -> None:
    """The `car-value` console script (see pyproject.toml [project.scripts])
    must resolve to src.cli:main.

    CI only ever exercised `src.cli` via `python -m src.cli ...` /
    `pythonpath = ["."]`, which does not catch a broken or missing
    packaging entry point. This checks the entry point metadata directly,
    so a typo or removal in [project.scripts] fails fast instead of only
    surfacing when a user runs `pip install -e .` and types `car-value`.
    """
    scripts = entry_points(group="console_scripts")
    matches = [ep for ep in scripts if ep.name == "car-value"]
    assert matches, "no 'car-value' console script registered"
    assert matches[0].value == "src.cli:main"


def test_console_script_runs_and_shows_help() -> None:
    """End-to-end: invoke the installed `car-value` executable itself.

    Skips cleanly (rather than failing) if the package hasn't been
    installed in this environment (e.g. `pip install -e .`), since some
    dev workflows only install requirements-dev.txt without the package
    itself.
    """
    car_value_path = shutil.which("car-value")
    if car_value_path is None:
        pytest.skip("car-value entry point not installed in this environment")

    result = subprocess.run(
        [car_value_path, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Car-Value-Decoding-Engine CLI" in result.stdout
