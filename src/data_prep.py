from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Load the raw car price dataset from CSV."""
    path = config.DATA_RAW
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")
    df = pd.read_csv(path)
    return df


def compute_car_age(year: int, reference_year: int | None = None) -> int:
    """Compute a car's age from its model year, clamped to a minimum of 1.

    Shared by clean_data() (applied per-row over the full dataset) and
    app.py's single-car input builder, so the "year >= reference_year ->
    age clamped to 1" rule can't drift between the two the way it could
    when each one reimplemented the formula separately.
    """
    ref = config.REFERENCE_YEAR if reference_year is None else reference_year
    age = ref - year
    return age if age > 0 else 1


def compute_km_per_year(kms: float, car_age: float) -> float:
    """Compute km driven per year of car age, clamped to a minimum of 0.

    Assumes car_age > 0, which compute_car_age() already guarantees.
    """
    value = kms / car_age
    return value if value > 0 else 0.0


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace in object/string columns. Pandas 4 warns that
    # include="object" implicitly also matches the new "str" dtype for
    # backward compatibility; passing both explicitly silences that
    # warning while keeping identical behavior on pandas <4 (which has
    # no separate "str" dtype to match).
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Ensure numeric columns
    numeric_cols = [
        config.COL_YEAR,
        config.COL_SELLING_PRICE,
        config.COL_PRESENT_PRICE,
        config.COL_KMS_DRIVEN,
        config.COL_OWNER,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key columns
    required = [
        config.COL_YEAR,
        config.COL_SELLING_PRICE,
        config.COL_PRESENT_PRICE,
        config.COL_KMS_DRIVEN,
        config.COL_FUEL_TYPE,
        config.COL_SELLER_TYPE,
        config.COL_TRANSMISSION,
        config.COL_OWNER,
    ]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Fill missing categoricals so OneHotEncoder never sees NaN.
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Compute car age and km/year via the shared helpers above (also used
    # by app.py's single-car input builder), so the two can't drift.
    df[config.COL_CAR_AGE] = df[config.COL_YEAR].apply(compute_car_age)
    df[config.COL_KM_PER_YEAR] = df.apply(
        lambda r: compute_km_per_year(r[config.COL_KMS_DRIVEN], r[config.COL_CAR_AGE]),
        axis=1,
    )

    return df


def load_clean() -> pd.DataFrame:
    """Load cleaned data, computing and caching it if needed."""
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = config.DATA_PROCESSED_DIR / "car_price_clean.parquet"

    if clean_path.exists():
        return pd.read_parquet(clean_path)

    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    df_clean.to_parquet(clean_path, index=False)
    return df_clean
