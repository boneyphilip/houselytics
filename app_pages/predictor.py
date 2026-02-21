import streamlit as st
import pandas as pd
import joblib

from src.preprocess import build_feature_frame, preprocess_inherited


@st.cache_resource
def load_model():
    """Load trained model once (cached)."""
    return joblib.load("src/house_price_model.pkl")


@st.cache_data
def load_training_data():
    """Load training data columns once (cached) to prevent feature mismatch."""
    df = pd.read_csv("data/processed/clean_train.csv")
    feature_cols = df.drop("SalePrice", axis=1).columns
    train_features = df.drop("SalePrice", axis=1)
    return feature_cols, train_features


def render() -> None:
    st.header("Price Predictor")

    # Load model + training feature columns (cached)
    model = load_model()
    feature_columns, train_features_df = load_training_data()

    tab1, tab2 = st.tabs(["Quick Estimate", "Pro Estimate"])

    # -------------------------
    # QUICK TAB
    # -------------------------
    with tab1:
        st.subheader("Quick Estimate")
        st.write("Simple inputs for a fast estimate.")

        quality_map = {
            "Basic": 3,
            "Standard": 5,
            "Good": 7,
            "Premium": 9,
        }

        quality_label = st.selectbox(
            "Construction quality",
            list(quality_map.keys()),
            key="quick_quality_label",
        )
        overall_qual = quality_map[quality_label]

        gr_liv_area = st.number_input(
            "Total living area (sq ft)",
            300,
            6000,
            1500,
            step=50,
            key="quick_gr_liv_area",
        )

        garage_area = st.number_input(
            "Garage area (sq ft)",
            0,
            1500,
            400,
            step=25,
            key="quick_garage_area",
        )

        year_built = st.number_input(
            "Year built",
            1800,
            2025,
            2000,
            key="quick_year_built",
        )

        if st.button("Estimate price (Quick)", key="btn_quick"):
            user_values = {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageArea": garage_area,
                "YearBuilt": year_built,
            }

            input_df = build_feature_frame(
                user_values, feature_columns, train_features_df
            )
            pred = model.predict(input_df)[0]

            st.metric("Estimated Market Value", f"${pred:,.0f}")
            st.caption(
                "Note: This is an ML estimate based on historical "
                "Ames housing data."
            )

    # -------------------------
    # PRO TAB
    # -------------------------
    with tab2:
        st.subheader("Pro Estimate")
        st.write("More inputs = more detailed estimate.")

        col1, col2, col3 = st.columns(3)

        with col1:
            overall_qual = st.slider(
                "Overall quality (1–10)",
                1,
                10,
                5,
                key="pro_overall_qual",
            )
            overall_cond = st.slider(
                "Overall condition (1–9)",
                1,
                9,
                5,
                key="pro_overall_cond",
            )
            year_built = st.number_input(
                "Year built",
                1800,
                2025,
                2000,
                key="pro_year_built",
            )
            year_remod = st.number_input(
                "Remodel year",
                1800,
                2025,
                2005,
                key="pro_year_remod",
            )

        with col2:
            gr_liv_area = st.number_input(
                "Living area (sq ft)",
                300,
                6000,
                1500,
                step=50,
                key="pro_gr_liv_area",
            )
            bedroom = st.number_input(
                "Bedrooms (above ground)",
                0,
                10,
                3,
                key="pro_bedroom",
            )
            first_flr = st.number_input(
                "1st floor area (sq ft)",
                300,
                5000,
                1100,
                step=50,
                key="pro_1stflr",
            )
            second_flr = st.number_input(
                "2nd floor area (sq ft)",
                0,
                3000,
                300,
                step=50,
                key="pro_2ndflr",
            )

        with col3:
            lot_area = st.number_input(
                "Lot area (sq ft)",
                1000,
                250000,
                9000,
                step=250,
                key="pro_lot_area",
            )
            total_bsmt = st.number_input(
                "Total basement area (sq ft)",
                0,
                7000,
                900,
                step=50,
                key="pro_total_bsmt",
            )
            garage_area = st.number_input(
                "Garage area (sq ft)",
                0,
                1500,
                400,
                step=25,
                key="pro_garage_area",
            )
            open_porch = st.number_input(
                "Open porch (sq ft)",
                0,
                600,
                20,
                step=10,
                key="pro_open_porch",
            )

        if st.button("Estimate price (Pro)", key="btn_pro"):
            user_values = {
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "GrLivArea": gr_liv_area,
                "BedroomAbvGr": bedroom,
                "1stFlrSF": first_flr,
                "2ndFlrSF": second_flr,
                "LotArea": lot_area,
                "TotalBsmtSF": total_bsmt,
                "GarageArea": garage_area,
                "OpenPorchSF": open_porch,
            }

            input_df = build_feature_frame(
                user_values, feature_columns, train_features_df
            )
            pred = model.predict(input_df)[0]

            st.metric("Estimated Market Value", f"${pred:,.0f}")
            st.caption("Note: Pro estimate uses more property attributes.")

    st.divider()
    st.subheader("Lydia’s 4 inherited houses (auto prediction)")

    if st.button("Predict inherited houses", key="btn_inherited"):
        inherited_df = pd.read_csv("data/raw/inherited_houses.csv")
        inherited_ready = preprocess_inherited(
            inherited_df, feature_columns, train_features_df
        )

        preds = model.predict(inherited_ready)
        out = inherited_df.copy()
        out["PredictedSalePrice"] = preds.round(0).astype(int)

        st.dataframe(out, width="stretch")
