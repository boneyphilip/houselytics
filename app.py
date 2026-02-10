import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Houselytics",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("src/house_price_model.pkl")

train_df = pd.read_csv("data/processed/clean_train.csv")
feature_columns = train_df.drop("SalePrice", axis=1).columns


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>Houselytics</h1>
    <p style='text-align: center; color: grey; font-size:18px;'>
        AI Powered Property Valuation System
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# -------------------------------------------------
# PROPERTY INPUTS
# -------------------------------------------------
st.markdown("### Property Details")


quality_map = {
    "Basic Construction": 3,
    "Standard Construction": 5,
    "High Quality Construction": 7,
    "Luxury Construction": 9
}

quality_selected = st.selectbox(
    "Construction Quality",
    list(quality_map.keys())
)

overall_qual = quality_map[quality_selected]


col1, col2 = st.columns(2)

with col1:

    gr_liv_area = st.number_input(
        "Living Area (Square Feet)",
        min_value=300,
        max_value=6000,
        value=1500
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2025,
        value=2005
    )

with col2:

    garage_area = st.number_input(
        "Garage Area (Square Feet)",
        min_value=0,
        max_value=1500,
        value=400
    )


# -------------------------------------------------
# BUTTON (DEFINE BEFORE USING)
# -------------------------------------------------
predict = st.button("Estimate Property Value")


# -------------------------------------------------
# CREATE INPUT DATA (DEFINE BEFORE PREDICTION)
# -------------------------------------------------
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0

input_data.at[0, "OverallQual"] = overall_qual
input_data.at[0, "GrLivArea"] = gr_liv_area
input_data.at[0, "GarageArea"] = garage_area
input_data.at[0, "YearBuilt"] = year_built


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if predict:

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Estimated Market Value")

    st.metric(
        label="Predicted House Price",
        value=f"${prediction:,.0f}"
    )

    st.caption(
        "This valuation is generated using a Machine Learning model "
        "trained on historical housing data."
    )
