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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace in object columns
    for col in df.select_dtypes(include="object").columns:
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

    # 🔹 NEW: fill missing categoricals so OneHotEncoder never sees NaN
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Compute car age using a reference year
    reference_year = 2020
    df[config.COL_CAR_AGE] = reference_year - df[config.COL_YEAR]
    df.loc[df[config.COL_CAR_AGE] <= 0, config.COL_CAR_AGE] = 1

    # Compute km_per_year
    df[config.COL_KM_PER_YEAR] = df[config.COL_KMS_DRIVEN] / df[config.COL_CAR_AGE]
    df[config.COL_KM_PER_YEAR] = df[config.COL_KM_PER_YEAR].clip(lower=0)

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
