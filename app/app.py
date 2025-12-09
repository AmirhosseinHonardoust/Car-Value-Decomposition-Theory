import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------------------------------
# Make src importable
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config, data_prep, features
from src.value_decomposition import decompose_value_for_row
from src.train_model import train_price_model


# --------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    return data_prep.load_clean()


@st.cache_resource
def load_model():
    """Load the trained model, training once if it doesn't exist."""
    if not config.MODEL_PATH.exists():
        train_price_model()
    return joblib.load(config.MODEL_PATH)


# --------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Car Value Decoding Engine", layout="wide")
    st.title("Car-Value-Decoding-Engine")

    df = load_data()
    model = load_model()  # noqa: F841  # kept for future extensions

    tab_single, tab_compare, tab_market = st.tabs(
        ["Single Car Decoder", "Compare Two Cars", "Market Explorer"]
    )

    # ----------------------------------------------------------------
    # Tab 1: Single Car Decoder
    # ----------------------------------------------------------------
    with tab_single:
        st.subheader("Decode the value of a single car")

        # Use a random row as a reasonable default
        example_row = df.sample(1, random_state=config.RANDOM_STATE).iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input(
                "Year",
                min_value=int(df[config.COL_YEAR].min()),
                max_value=int(df[config.COL_YEAR].max()),
                value=int(example_row[config.COL_YEAR]),
            )
            kms = st.number_input(
                "Kms Driven",
                min_value=0,
                max_value=int(df[config.COL_KMS_DRIVEN].max()),
                value=int(example_row[config.COL_KMS_DRIVEN]),
            )

        with col2:
            fuel = st.selectbox(
                "Fuel Type",
                sorted(df[config.COL_FUEL_TYPE].unique().tolist()),
                index=0,
            )
            transmission = st.selectbox(
                "Transmission",
                sorted(df[config.COL_TRANSMISSION].unique().tolist()),
                index=0,
            )

        with col3:
            seller = st.selectbox(
                "Seller Type (Condition)",
                sorted(df[config.COL_SELLER_TYPE].unique().tolist()),
                index=0,
            )
            owner = st.number_input(
                "Number of previous owners (Car ID is not used as ID here, just numeric)",
                min_value=int(df[config.COL_OWNER].min()),
                max_value=int(df[config.COL_OWNER].max()),
                value=int(example_row[config.COL_OWNER]),
            )

        # Present price (in this project we treat this as a spec proxy, e.g. engine size)
        present_price = st.number_input(
            "Present Price / Engine Size (spec proxy)",
            min_value=float(df[config.COL_PRESENT_PRICE].min()),
            max_value=float(df[config.COL_PRESENT_PRICE].max()),
            value=float(example_row[config.COL_PRESENT_PRICE]),
            step=0.1,
        )

        # Brand + Model
        brands = sorted(df[config.COL_BRAND].unique().tolist())
        default_brand = str(example_row.get(config.COL_BRAND, brands[0]))
        brand_index = brands.index(default_brand) if default_brand in brands else 0

        brand = st.selectbox(
            "Brand",
            brands,
            index=brand_index,
        )

        model_name = st.text_input(
            "Model Name",
            value=str(example_row.get(config.COL_CAR_NAME, "Model")),
        )

        if st.button("Decode Car Value"):
            # Compute engineered features consistent with data_prep
            reference_year = 2020
            car_age = reference_year - year
            if car_age <= 0:
                car_age = 1
            km_per_year = kms / car_age if car_age > 0 else kms

            # Build one-row dataframe
            row_dict = {
                config.COL_CAR_NAME: model_name,
                config.COL_BRAND: brand,
                config.COL_YEAR: year,
                config.COL_SELLING_PRICE: np.nan,  # unknown at prediction time
                config.COL_PRESENT_PRICE: present_price,
                config.COL_KMS_DRIVEN: kms,
                config.COL_FUEL_TYPE: fuel,
                config.COL_SELLER_TYPE: seller,
                config.COL_TRANSMISSION: transmission,
                config.COL_OWNER: owner,
                config.COL_CAR_AGE: car_age,
                config.COL_KM_PER_YEAR: km_per_year,
            }
            df_input = pd.DataFrame([row_dict])

            # Decode value
            dec = decompose_value_for_row(df_input.iloc[0])

            st.markdown("### Predicted Price")
            st.metric("Predicted selling price", f"{dec['final_prediction']:.2f}")

            st.markdown("### Value Decomposition")
            contrib_keys = [k for k in dec.keys() if k.startswith("contrib_")]
            if contrib_keys:
                contrib_data = {
                    "component": [k.replace("contrib_", "") for k in contrib_keys],
                    "contribution": [dec[k] for k in contrib_keys],
                }
                contrib_df = pd.DataFrame(contrib_data)
                st.bar_chart(contrib_df.set_index("component"))

            st.markdown("### Raw decomposition values")
            st.json(dec)

    # ----------------------------------------------------------------
    # Tab 2: Compare Two Cars
    # ----------------------------------------------------------------
    with tab_compare:
        st.subheader("Compare two cars from the dataset")

        idx1 = st.number_input(
            "Index of first car", min_value=0, max_value=len(df) - 1, value=0
        )
        idx2 = st.number_input(
            "Index of second car", min_value=0, max_value=len(df) - 1, value=1
        )

        if st.button("Compare"):
            row1 = df.iloc[int(idx1)]
            row2 = df.iloc[int(idx2)]

            dec1 = decompose_value_for_row(row1)
            dec2 = decompose_value_for_row(row2)

            st.markdown("### Car 1 Specs")
            st.json(row1.to_dict())
            st.markdown("### Car 1 Decomposition")
            st.json(dec1)

            st.markdown("### Car 2 Specs")
            st.json(row2.to_dict())
            st.markdown("### Car 2 Decomposition")
            st.json(dec2)

    # ----------------------------------------------------------------
    # Tab 3: Market Explorer
    # ----------------------------------------------------------------
    with tab_market:
        st.subheader("Market Explorer")

        df_plot = features.add_brand_column(df)

        st.write("Price vs car age (sample of the dataset).")
        sample = df_plot.sample(min(500, len(df_plot)), random_state=config.RANDOM_STATE)
        st.scatter_chart(sample, x=config.COL_CAR_AGE, y=config.COL_SELLING_PRICE)

        st.write(
            "You can extend this tab with richer visualizations like brand premiums, "
            "fuel-type effects, or condition-based price distributions."
        )


if __name__ == "__main__":
    main()
