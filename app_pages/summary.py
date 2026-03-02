import streamlit as st


def render() -> None:
    """Project Summary page."""

    st.title("Project Summary")

    # --- Introduction Section ---
    st.markdown(
        "Houselytics is a predictive analytics dashboard built to support "
        "a property valuation use-case using the **Ames, Iowa housing "
        "dataset**. It helps the client understand **which house features "
        "influence sale price** and provides **machine learning-based "
        "price estimates** for both:"
    )

    st.markdown(
        "- A known set of inherited houses.\n"
        "- New user-entered property details."
    )

    st.success(
        "This dashboard directly supports both project requirements by "
        "combining explanatory data analysis with a working regression-"
        "based prediction system."
    )

    st.divider()

    # --- Context Section ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Client and Context")
        st.markdown(
            "**Client:** Lydia Doe  \n"
            "**Client Context:** Belgium-based property owner  \n"
            "**Property Market:** Ames, Iowa, USA  \n"
            "**Scenario:** Lydia inherited four houses and needs support "
            "understanding value drivers and estimating realistic market "
            "value."
        )

    with col2:
        st.subheader("Stakeholder Goals")
        st.markdown(
            "1. **Identify** the key drivers of sale price.\n"
            "2. **Estimate** sale price for inherited houses.\n"
            "3. **Predict** sale price for new user inputs.\n"
            "4. **Support** better pricing decisions using data."
        )

    st.divider()

    # --- Business Requirements ---
    st.subheader("Business Requirements")
    st.markdown("The solution is designed to satisfy two core requirements:")

    st.info(
        "**Business Requirement 1: Data Analysis**\n\n"
        "The client wants to understand how house attributes correlate "
        "with `SalePrice`. The dashboard therefore provides visual "
        "analysis of the strongest relationships between features and "
        "house value."
    )

    st.success(
        "**Business Requirement 2: Predictive System**\n\n"
        "The client, Lydia Doe, wants to estimate the sale price of her "
        "four inherited houses and predict the sale price of any other "
        "house in Ames, Iowa using a trained regression model."
    )

    st.divider()

    # --- Dashboard Features ---
    st.subheader("Dashboard Guide")

    tabs = st.tabs(
        ["Recommended Flow", "Prediction Modes", "Trust and Limits"]
    )

    with tabs[0]:
        st.markdown(
            "**Recommended workflow:**\n\n"
            "1. Start with **Project Hypothesis** to see what the project "
            "tests.\n"
            "2. Move to **Data Insights** to understand the key price "
            "drivers.\n"
            "3. Use **Price Predictor** to estimate values.\n"
            "4. Review **Model Performance** to assess model quality."
        )

    with tabs[1]:
        st.markdown(
            "**Available prediction modes:**\n\n"
            "- **Quick Estimate:** Fewer inputs for a fast result.\n"
            "- **Detailed Analysis:** More inputs for a richer estimate.\n"
            "- **Inherited Houses:** Batch prediction for Lydia's four "
            "properties."
        )

    with tabs[2]:
        st.markdown(
            "**Model transparency:**\n\n"
            "The **Model Performance** page provides evaluation metrics "
            "such as R², MAE, and RMSE, along with visual checks like "
            "Actual vs Predicted, residual analysis, and feature "
            "importance."
        )

    st.divider()

    # --- Scope & Limitations ---
    with st.expander("Project scope and limitations"):
        st.write(
            "Predictions are based on historical Ames, Iowa housing data. "
            "The model provides a statistical estimate, not a guaranteed "
            "sale price."
        )
        st.write(
            "External factors such as local demand, renovation quality, "
            "interest rates, and timing of sale can still affect the "
            "final market value."
        )
