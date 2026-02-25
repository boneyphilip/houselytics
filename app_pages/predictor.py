"""
Price Predictor Page.

Goal:
- Provide a clean, professional property valuation experience.
- Support both quick estimates and more detailed estimates.
- Show batch predictions for Lydia's 4 inherited houses.
"""

from __future__ import annotations
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.preprocess import build_feature_frame, preprocess_inherited


# ------------------------------------------------------------
# 1) Load the trained model once (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the saved ML model.
    Caching prevents repeated re-loading from disk.
    """
    return joblib.load("src/house_price_model.pkl")


# ------------------------------------------------------------
# 2) Load training metadata once (cached)
# ------------------------------------------------------------
@st.cache_data
def load_training_data():
    """
    Load processed training data to recover original feature columns.
    Ensures input frames match model requirements.
    """
    df = pd.read_csv("data/processed/clean_train.csv")
    feature_cols = df.drop("SalePrice", axis=1).columns
    train_features = df.drop("SalePrice", axis=1)
    return feature_cols, train_features


# ------------------------------------------------------------
# 3) Plotly helper (future-safe width handling)
# ------------------------------------------------------------
def _plotly_chart(fig) -> None:
    """
    Render Plotly charts in a responsive way.
    """
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------
# 4) DataFrame helper (future-safe width handling)
# ------------------------------------------------------------
def _safe_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Render dataframes with responsive width.
    """
    try:
        st.dataframe(dataframe, use_container_width=True)
    except TypeError:
        st.dataframe(dataframe, width="stretch")


# ------------------------------------------------------------
# 5) Small helper for result display
# ------------------------------------------------------------
def _show_prediction_result(
    title: str,
    predicted_value: float,
    note: str,
) -> None:
    """
    Display one prediction result in a clean format.
    """
    st.markdown("---")
    st.subheader(title)
    st.metric("Estimated Market Value", f"${predicted_value:,.0f}")
    st.caption(note)
    st.info(
        "This value is a machine learning estimate based on historical "
        "housing data from Ames, Iowa. Real market prices may differ."
    )


# ------------------------------------------------------------
# 6) Main page renderer
# ------------------------------------------------------------
def render() -> None:
    """
    Main function to display the Price Predictor page.
    """
    st.title("Property Valuation Tool")
    st.markdown(
        "Ames, Iowa House Price Prediction Service. "
        "Enter property specifications below to generate a market estimate."
    )

    # Helpful user instructions
    with st.expander("How to use this page"):
        st.write(
            "1. Use **Quick Estimate** for a fast result.\n"
            "2. Use **Detailed Analysis** for a professional estimate.\n"
            "3. Use **Inherited Appraisal** for Lydia's portfolio.\n"
            "4. Treat all results as estimates, not guaranteed prices."
        )

    # Definitions for confusing terms
    with st.expander("Field definitions"):
        st.write(
            "- **Construction Quality**: overall build and finish level.\n"
            "- **Condition**: general wear and maintenance.\n"
            "- **Living Area**: indoor space used for living.\n"
            "- **Lot Size**: total land area of the property."
        )

    model = load_model()
    feature_columns, train_features_df = load_training_data()

    tab1, tab2 = st.tabs(["Quick Estimate", "Detailed Analysis"])

    # --- TAB 1: QUICK ESTIMATE ---
    with tab1:
        st.subheader("Property Basics")
        st.caption("Fast estimate using high-impact features.")

        quality_map = {
            "Basic (Low)": 3,
            "Standard (Average)": 5,
            "Good (Above Average)": 7,
            "Premium (High)": 9,
        }

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            quality_label = st.selectbox(
                "Construction Quality",
                options=list(quality_map.keys()),
                key="quick_qual",
            )
            gr_liv_area = st.number_input(
                "Total Living Area (sq ft)",
                300, 6000, 1500, step=50,
                key="quick_area",
            )
        with col_q2:
            year_built = st.number_input(
                "Year Built", 1800, 2026, 2000,
                key="quick_yr",
            )
            garage_area = st.number_input(
                "Garage Size (sq ft)", 0, 1500, 400, step=25,
                key="quick_gar",
            )

        if st.button("Generate Valuation", key="btn_quick"):
            user_values = {
                "OverallQual": quality_map[quality_label],
                "GrLivArea": gr_liv_area,
                "GarageArea": garage_area,
                "YearBuilt": year_built,
            }
            input_df = build_feature_frame(
                user_values, feature_columns, train_features_df
            )
            pred = model.predict(input_df)[0]
            _show_prediction_result(
                "Quick Estimate Result",
                pred,
                "Quick Estimate uses a smaller set of inputs for speed."
            )

    # --- TAB 2: DETAILED ANALYSIS ---
    with tab2:
        st.subheader("Comprehensive Attribute Input")
        st.write("Complete more fields for a richer estimate.")

        c1, c2, c3 = st.columns(3)
        with c1:
            ov_qual = st.slider("Quality (1-10)", 1, 10, 5)
            ov_cond = st.slider("Condition (1-9)", 1, 9, 5)
            yr_blt = st.number_input("Year Built", 1800, 2026, 2000)
        with c2:
            liv_area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
            lot_size = st.number_input("Lot Size (sq ft)", 1000, 50000, 9000)
            bed = st.number_input("Bedrooms", 0, 10, 3)
        with c3:
            bsmt = st.number_input("Basement (sq ft)", 0, 5000, 900)
            gar = st.number_input("Garage (sq ft)", 0, 1500, 400)
            yr_rem = st.number_input("Remodel Year", 1800, 2026, 2005)

        if st.button("Run Detailed Valuation", type="primary"):
            user_values = {
                "OverallQual": ov_qual, "OverallCond": ov_cond,
                "YearBuilt": yr_blt, "YearRemodAdd": yr_rem,
                "GrLivArea": liv_area, "BedroomAbvGr": bed,
                "LotArea": lot_size, "TotalBsmtSF": bsmt,
                "GarageArea": gar
            }
            input_df = build_feature_frame(
                user_values, feature_columns, train_features_df
            )
            pred = model.predict(input_df)[0]
            _show_prediction_result(
                "Detailed Analysis Result",
                pred,
                "Detailed Analysis uses more property attributes."
            )

    # --- INHERITED PORTFOLIO ---
    st.divider()
    st.subheader("Inherited Property Appraisal")

    if st.button("Calculate Portfolio Value", key="btn_inherited"):
        inherited_df = pd.read_csv("data/raw/inherited_houses.csv")
        ready_df = preprocess_inherited(
            inherited_df, feature_columns, train_features_df
        )
        preds = model.predict(ready_df)

        st.metric("Aggregate Portfolio Value", f"${preds.sum():,.0f}")

        # Summary Chart
        chart_df = pd.DataFrame({
            "Property": [f"House {i+1}" for i in range(len(preds))],
            "Value": preds.round(0)
        })
        fig = px.bar(
            chart_df, x="Property", y="Value", color="Value",
            color_continuous_scale="Viridis", template="plotly_dark"
        )
        _plotly_chart(fig)

        # Property Cards
        cols = st.columns(2)
        for i, price in enumerate(preds):
            with cols[i % 2]:
                st.info(f"**Property Identification: {i + 1}**")
                st.markdown(f"**Appraisal:** ${price:,.0f}")

        with st.expander("Detailed Metadata"):
            out = inherited_df.copy()
            out["Predicted_Value"] = preds.round(0).astype(int)
            _safe_dataframe(out)
