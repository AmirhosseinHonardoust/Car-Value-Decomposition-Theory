<div align="center">

# Car-Value-Decoding-Engine
<img width="1262" height="472" alt="Car Value Decoding Engine" src="https://github.com/user-attachments/assets/6dc677e5-437d-43c0-9f8b-099ca1ba179e" />

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForestRegressor-orange)
![Explainability](https://img.shields.io/badge/Explainability-Value%20Decomposition-9C27B0)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Car-Value-Decomposition-Theory/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Car-Value-Decomposition-Theory/actions/workflows/ci.yml)

</div>

A machine learning project that predicts car prices with a **Random Forest Regressor** and then *explains* each prediction by decomposing it into per-group contributions (brand, age, mileage, fuel type, transmission, condition, base spec), with **command-line inference**, a **Streamlit dashboard**, permutation importance, a group-level bias check, and an automated test suite.

> **Important:** This project is an **educational explainability demo**, not a real-world valuation or appraisal tool.
>
> The bundled dataset's `Price` column shows essentially no measurable correlation with any of its own numeric features (verified below). The model's job here is to demonstrate an honest, auditable decomposition *method* — not to produce prices anyone should act on. See [Limitations](#limitations) before drawing any conclusions from its output.

> **Naming note:** the app and this README use the display name "Car-Value-Decoding-Engine"; the GitHub repository itself is named `Car-Value-Decomposition-Theory`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Value Decomposition Method and Known Limitation](#value-decomposition-method-and-known-limitation)
- [Example Decomposition Output](#example-decomposition-output)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Most price-prediction demos stop at a single number. This project instead treats explainability as the deliverable: every prediction is broken down into how much each feature group added or subtracted relative to a baseline "average" car, in a way that always reconstructs exactly back to the final prediction.

The system covers the full loop: data cleaning and feature engineering, model training, a sequential group-swap decomposition method, permutation-importance and group-bias diagnostics, a CLI, a Streamlit dashboard, and an automated test suite that pins down both the intended behavior *and* a known limitation of the decomposition method (see below).

---

## What This Project Does

This project can:

- Clean and engineer features from a raw car-listing CSV (car age, mileage-per-year, brand extraction)
- Train a `RandomForestRegressor` price model and report MAE, RMSE, and R²
- Decompose any single prediction into per-group contributions (brand, age, mileage, fuel, transmission, seller/condition, base spec) that sum exactly to the final prediction
- Compare two cars' specs and decompositions side by side
- Compute global permutation feature importance
- Compare average predicted vs. actual price across a categorical group (a basic bias/fairness check)
- Automatically print and display a plain-language interpretation of R² every time the model is trained, evaluated, or queried — including a clear "worse than predicting the mean" warning when R² is negative — so the model's real predictive quality can't be missed
- Provide a four-command CLI (`prepare-data`, `train`, `evaluate`, `decode-car`)
- Provide a three-tab Streamlit dashboard (single-car decoder, car comparison, market explorer)
- Run an isolated, fixture-based pytest suite in CI (ruff, black, mypy, pytest)

---

## What This Project Does Not Do

This project does **not**:

- Produce prices suitable for real buying, selling, or appraisal decisions
- Guarantee that per-group contributions are stable — they depend on a fixed swap order, not a Shapley-style average over orderings (see below)
- Perform any real-world market validation; it is trained and evaluated only on one bundled Kaggle CSV
- Audit for fairness beyond a single group-vs-group average-error comparison
- Retrain, monitor, or serve the model in production

---

## Key Features

- **Random Forest price model** (`RandomForestRegressor`, 300 trees) trained via a `ColumnTransformer` + `Pipeline` (scaling for numerics, one-hot encoding for categoricals)
- **Sequential group-swap decomposition** (`value_decomposition.py`) — swaps each feature group from a baseline row into the sample row one group at a time and measures the prediction delta, so contributions always sum exactly to the final prediction
- **Documented order-dependence** — a dedicated test (`tests/test_value_decomposition.py`) proves contributions change if the group order changes, so this known limitation can't silently regress
- **Opt-in Shapley approximation** — `decompose_value_for_row(row, n_orderings=N)` averages contributions over `N` random group orderings instead of one fixed order, while still reconstructing exactly to the final prediction
- **Permutation importance** (`explainability.py`) for global feature ranking
- **Group-level bias check** (`fairness_checks.py`) comparing average predicted vs. actual price per category
- **Automatic R² interpretation** (`model_quality.py`) — a shared, tested function that `train`, `evaluate`, `decode-car`, and the Streamlit app all call to print/display an honest, plain-language read of model quality, instead of leaving a bare R² number for the reader to interpret
- **CLI** with four subcommands and clear error handling (missing file, missing model, out-of-range index)
- **Streamlit dashboard** with three tabs: single-car decoder, two-car comparison, market explorer
- **Isolated pytest fixtures** (`tests/conftest.py`) that redirect every filesystem path to a throwaway temp directory, so tests never touch real project data
- **Ruff, Black, and mypy** configured in `pyproject.toml`, enforced in CI
- **GitHub Actions CI** running lint, format-check, type-check, and the full test suite

---

## System Workflow

```text
Raw CSV (data/raw/car_price_prediction.csv)
        ↓
Cleaning + feature engineering (data_prep.py)
        ↓
Preprocessing: scaling + one-hot encoding (features.py)
        ↓
RandomForestRegressor training (train_model.py)
        ↓
Saved model + baseline feature stats (models/)
        ↓
Sequential group-swap decomposition (value_decomposition.py)
        ↓
Per-group contributions + reconstruction check
        ↓
CLI (src/cli.py) and Streamlit dashboard (app/app.py)
```

---

## Project Structure

```text
Car-Value-Decomposition-Theory/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── car_price_prediction.csv
│
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data_prep.py
│   ├── evaluate_model.py
│   ├── explainability.py
│   ├── fairness_checks.py
│   ├── features.py
│   ├── model_quality.py
│   ├── train_model.py
│   └── value_decomposition.py
│
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_data_prep.py
│   ├── test_evaluate_model.py
│   ├── test_explainability.py
│   ├── test_fairness_checks.py
│   ├── test_features.py
│   ├── test_model_quality.py
│   ├── test_train_model.py
│   └── test_value_decomposition.py
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

`data/processed/`, `models/`, and `reports/metrics/` are generated locally by the CLI and excluded via `.gitignore`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Car-Value-Decomposition-Theory.git
cd Car-Value-Decomposition-Theory
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt` / `requirements-dev.txt` for reproducible installs.

For development tools (pytest, ruff, black, mypy):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Clean and cache the dataset:

```bash
python -m src.cli prepare-data
```

Train the model:

```bash
python -m src.cli train
```

Evaluate the trained model:

```bash
python -m src.cli evaluate
```

Decode a single car's price into value components:

```bash
python -m src.cli decode-car --index 0
```

Launch the dashboard:

```bash
streamlit run app/app.py
```

---

## Training and Evaluation

`train` loads the cleaned dataset, splits it (80/20, `random_state=42`), fits the preprocessing + Random Forest pipeline, and writes:

```text
models/price_model.joblib
models/baseline_stats.json
reports/metrics/regression_metrics.json
```

`baseline_stats.json` stores the mean of each numeric feature and the mode of each categorical feature across the training split — this baseline row is what every decomposition starts from. `evaluate` reloads the saved model and re-scores it on a freshly drawn test split, writing `reports/metrics/regression_metrics_eval.json`.

---

## Value Decomposition Method and Known Limitation

`value_decomposition.py` explains a prediction by starting from the baseline row and swapping in one feature group at a time (brand, then age, then mileage, and so on), measuring how much the prediction moves at each step. By construction, `base_value + sum(contributions) == final_prediction` exactly — this project's own tests verify a reconstruction error of `0.0`.

This is **not** the same guarantee Shapley values give. Shapley values average the contribution of each group over *every possible ordering*; this method uses one fixed ordering (whatever order the groups happen to be stored in inside `baseline_stats.json`). That means:

- The reconstruction identity always holds — it's a property of the arithmetic, not evidence the attribution is "correct"
- Per-group numbers can change if the group order changes, for the same car and the same model

`tests/test_value_decomposition.py::test_contributions_are_order_dependent_known_limitation` proves this by reversing the group order and confirming at least one contribution changes. This test exists specifically so the limitation can't silently disappear or get "fixed" without anyone noticing.

`decompose_value_for_row` also accepts an opt-in `n_orderings` parameter. The default, `n_orderings=1`, is the exact behavior described above and is unchanged. Passing `n_orderings > 1` (plus an optional `random_state`) instead averages the per-group contributions over that many random group orderings — a cheap Monte Carlo approximation of a true Shapley-value average over *every* ordering. The reconstruction identity still holds exactly for the averaged result, because each individual ordering's contributions already sum to the same base-to-final delta regardless of order. This is not wired into the CLI or Streamlit app by default; call it directly (`decompose_value_for_row(row, n_orderings=25)`) if you want smoothed-out contributions instead of the fixed-order ones.

---

## Example Decomposition Output

Run against the bundled dataset (row 0, a 2016 Tesla, Petrol, Manual, "New" condition), verified in this environment:

```json
{
  "base_value": 43029.64,
  "final_prediction": 35690.13,
  "reconstruction_error": 0.0,
  "contrib_base_spec": 10636.23,
  "contrib_brand": 1371.73,
  "contrib_age": -851.42,
  "contrib_mileage": -10765.85,
  "contrib_fuel": -3881.96,
  "contrib_transmission": 0.0,
  "contrib_seller": -3848.24
}
```

`base_value + sum(contrib_*) == final_prediction` exactly (43029.64 + 1371.73 − 851.42 − 10765.85 − 3881.96 + 0.0 − 3848.24 + 10636.23 = 35690.13). Given the near-zero correlation between `Price` and the underlying features on this dataset (see [Limitations](#limitations)), treat these specific numbers as a demonstration of the *mechanism*, not a real pricing signal.

---

## Streamlit Dashboard

Launch the app:

```bash
streamlit run app/app.py
```

The dashboard has three tabs:

- **Single Car Decoder** — configure a car's specs and see its predicted price broken into contributions, as a bar chart and raw JSON
- **Compare Two Cars** — pick two rows from the dataset and view their specs and decompositions side by side
- **Market Explorer** — a scatter plot of price vs. car age over a sample of the dataset

<div align="center">
<img width="1262" height="472" alt="Single Car Decoder" src="https://github.com/user-attachments/assets/6dc677e5-437d-43c0-9f8b-099ca1ba179e" />
</div>

---

## Evaluation Metrics

<div align="center">

| Metric | Why it matters |
|---|---|
| MAE | Average absolute prediction error, in the same units as price |
| RMSE | Penalizes large errors more heavily than MAE |
| R² | Fraction of price variance explained by the model; 0 = no better than predicting the mean, negative = worse |

</div>

Verified results on the full bundled dataset (2,500 rows, 80/20 split, `random_state=42`):

<div align="center">

| Metric | Value |
|---|---:|
| MAE | 24,227.46 |
| RMSE | 28,407.25 |
| R² | −0.065 |

</div>

> **A negative R² means this model performs worse than simply predicting the mean price for every car.** Checking the underlying data explains why: the correlation between `Price` and every numeric feature (Engine Size, Mileage, car age, km-per-year) is below 0.06 in magnitude across the board. On this particular bundled CSV, price appears to carry no real signal tied to the other columns. This is the most important caveat in this README — read the decomposition output as a demonstration of the *method*, not as evidence the model has learned anything predictive about car pricing.

You don't have to rely on this README to catch that, though: `train`, `evaluate`, `decode-car`, and the Streamlit dashboard all print or display this same interpretation automatically (`src/model_quality.py`), every time you run them.

---

## Visual Reports

<div align="center">

| Raw Decomposition | Decomposition Chart |
|---|---|
| <img width="409" alt="Raw decomposition JSON" src="https://github.com/user-attachments/assets/b693d382-e3e7-42ef-849d-fd5530741415" /> | <img width="1262" alt="Decomposition bar chart" src="https://github.com/user-attachments/assets/56a7fb41-81fe-4db9-adce-dc8779abc8bd" /> |
| **What it shows:** the machine-readable decomposition — base value, per-group contributions, final prediction, and reconstruction error. | **What it shows:** the same contributions as a bar chart, making positive and negative pushes on price easy to scan at a glance. |

</div>

<details>
<summary>Compare Two Cars and Market Explorer tabs</summary>

<div align="center">

| Compare Two Cars | Market Explorer |
|---|---|
| <img width="657" alt="Compare two cars" src="https://github.com/user-attachments/assets/116c45e4-3ce2-486a-acf6-5d9324573fa3" /> | <img width="1099" alt="Market explorer scatter plot" src="https://github.com/user-attachments/assets/0456f0d2-ecd3-476e-90e0-1b15a0e23e8b" /> |

Compare Two Cars places two rows' specs and decompositions side by side. Market Explorer plots price against car age over a sample of the dataset.

</div>

</details>

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Lint, format-check, and type-check:

```bash
ruff check .
black --check .
mypy .
```

The GitHub Actions workflow runs, in order:

- dependency installation
- `ruff check .`
- `black --check .`
- `mypy .`
- `pytest -q`

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

<div align="center">

| Module | Purpose |
|---|---|
| `src/config.py` | Central column mapping, feature/group definitions, and path constants |
| `src/data_prep.py` | Loads and cleans the raw CSV; engineers `car_age` and `km_per_year`; caches to parquet |
| `src/features.py` | Builds the feature matrix and the scaling/one-hot preprocessing pipeline |
| `src/train_model.py` | Trains the Random Forest, saves the model, baseline stats, and metrics |
| `src/evaluate_model.py` | Reloads the saved model and re-scores it on a fresh split |
| `src/value_decomposition.py` | Sequential group-swap decomposition of a prediction |
| `src/explainability.py` | Global permutation feature importance |
| `src/fairness_checks.py` | Average predicted-vs-actual price by category |
| `src/model_quality.py` | Turns a raw R² number into an honest, plain-language interpretation, shared by the CLI and the app |
| `src/cli.py` | Argparse CLI wiring the four subcommands together |
| `app/app.py` | Streamlit dashboard |

</div>

Tooling is configured through `pyproject.toml` (ruff, black, mypy, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- **The bundled dataset shows no meaningful correlation between `Price` and any of its own features** (all |r| < 0.06), which drives the model's negative R² — treat all example output as a demonstration of the method, not a real valuation
- The decomposition method is order-dependent, not a true Shapley-value average — documented and tested, but still a real limitation of the current implementation
- `reference_year` for computing car age is a hardcoded constant, not derived from the current date
- The bias check compares only average predicted vs. actual price per single category; it isn't a full fairness audit
- Trained and evaluated on one bundled Kaggle CSV only — no external or out-of-sample validation
- The Streamlit app has no automated test coverage of its own beyond the pure logic it shares with `src/`

---

## Responsible Use

This repository is intended for:

- machine learning education and portfolio demonstration
- practicing explainable-ML workflow design (baseline construction, contribution decomposition, reconstruction checks)
- exploring the difference between an additive decomposition method and true Shapley values
- CLI and Streamlit app-building practice

It should not be used as-is for:

- real car pricing, appraisal, or negotiation decisions
- any production pricing, insurance, or credit system
- fairness or compliance claims about a real deployed model

A real deployment would need a dataset with genuine price signal, external validation, a properly averaged (Shapley or similar) attribution method, and human review.

---

## Future Improvements

- Wire the opt-in multi-ordering (`n_orderings`) averaging into the CLI and Streamlit app as a user-facing toggle, instead of requiring a direct function call
- Validate against a dataset with real, verifiable price signal
- Expand the bias check into a fuller fairness audit across multiple groups at once
- Derive `reference_year` from the current date instead of a hardcoded constant
- Add a lightweight smoke test for the Streamlit app
- Explore gradient-boosted alternatives to the Random Forest baseline

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- pyarrow
- Streamlit
- pytest
- ruff
- black
- mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

MIT License. See [LICENSE](LICENSE) for the full text.
