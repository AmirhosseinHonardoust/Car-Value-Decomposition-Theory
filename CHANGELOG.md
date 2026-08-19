# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- `src/data_prep.py`: silenced a pandas 4 `Pandas4Warning` by passing
  `include=["object", "str"]` to `select_dtypes` explicitly, instead of
  relying on pandas 4's deprecated object-includes-str backward-compat
  shim. No behavior change on any supported pandas version.
- `app/app.py`: the `sys.path` import fallback now only runs if `from src
  import ...` fails, instead of unconditionally mutating `sys.path` before
  every import. Supports both `pip install -e .` and running the script
  directly (`streamlit run app/app.py`) without an install step.

### Added
- `tests/test_app_render.py`: real interaction tests for `app/app.py`
  using Streamlit's `AppTest` harness -- clicks the single-car decoder and
  compare buttons, asserts on rendered metrics/JSON, and checks the R²
  warning banner. Previously only `test_app_smoke.py`'s import check
  covered this file.
- `Dockerfile` and `.dockerignore` for running the Streamlit dashboard as
  a container (`docker build -t car-value . && docker run -p 8501:8501
  car-value`).
- `src/split_utils.py`: `load_train_test_split()`, a single shared
  implementation of the load-clean → get-feature-matrix → train_test_split
  sequence.
- A code comment in `src/train_model.py` documenting the `joblib`/NumPy
  2.5 `DeprecationWarning` seen during `joblib.dump`, so it's a tracked,
  known issue rather than a silent surprise in a future upgrade.
- This changelog.

### Changed
- `src/train_model.py`, `src/evaluate_model.py`, `src/explainability.py`
  now call `split_utils.load_train_test_split()` instead of each
  duplicating an identical `train_test_split(...)` call. No change to the
  actual split produced (same `config.TEST_SIZE` / `config.RANDOM_STATE`
  as before).
