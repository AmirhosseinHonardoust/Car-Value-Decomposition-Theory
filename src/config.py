from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "car_price_prediction.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# === COLUMN MAPPING FOR YOUR DATASET ===
# Your CSV has:
# ['Car ID', 'Brand', 'Year', 'Engine Size', 'Fuel Type',
#  'Transmission', 'Mileage', 'Condition', 'Price', 'Model']

# We map them to the internal names the project expects:

COL_CAR_NAME = "Model"          # used only for extracting a "Brand" if needed
COL_YEAR = "Year"
COL_SELLING_PRICE = "Price"     # target we are predicting
COL_PRESENT_PRICE = "Engine Size"  # treat engine size as base spec
COL_KMS_DRIVEN = "Mileage"      # mileage column in your dataset
COL_FUEL_TYPE = "Fuel Type"
COL_SELLER_TYPE = "Condition"   # like New, Used, Like New → "condition profile"
COL_TRANSMISSION = "Transmission"
COL_OWNER = "Car ID"            # we keep it for completeness but do NOT model it

# Engineered feature names
COL_BRAND = "Brand"             # your dataset already has Brand
COL_CAR_AGE = "car_age"
COL_KM_PER_YEAR = "km_per_year"

# Numeric and categorical features used for modeling
NUMERIC_FEATURES = [
    COL_CAR_AGE,
    COL_PRESENT_PRICE,   # Engine Size
    COL_KMS_DRIVEN,      # Mileage
    COL_KM_PER_YEAR,
]

CATEGORICAL_FEATURES = [
    COL_BRAND,
    COL_FUEL_TYPE,
    COL_SELLER_TYPE,     # Condition
    COL_TRANSMISSION,
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Group definitions for value decomposition
GROUP_DEFINITIONS = {
    "base_spec": [COL_PRESENT_PRICE],                    # Engine Size
    "brand": [COL_BRAND],
    "age": [COL_CAR_AGE],
    "mileage": [COL_KMS_DRIVEN, COL_KM_PER_YEAR],
    "fuel": [COL_FUEL_TYPE],
    "transmission": [COL_TRANSMISSION],
    "seller": [COL_SELLER_TYPE],                         # Condition (New / Used / Like New)
}

BASELINE_STATS_PATH = MODELS_DIR / "baseline_stats.json"
MODEL_PATH = MODELS_DIR / "price_model.joblib"
