import streamlit as st


def render() -> None:
    """Project Summary page."""

    st.title("Project Summary")

    # --- Introduction Section ---
    st.markdown(
        "This dashboard supports a property valuation use-case using the "
        "**Ames, Iowa housing dataset**. It is designed to help a client "
        "understand **which property features influence sale price** and "
        "to provide **machine learning price estimates** for both:"
    )

    st.markdown(
        "- A known set of inherited houses.\n"
        "- New user-entered property details."
    )

    st.divider()

    # --- Context Section ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Client and Context")
        st.markdown(
            "**Client:** Lydia Doe  \n"
            "**Location:** Ames, Iowa (USA)  \n"
            "**Scenario:** Lydia inherited four houses and needs support "
            "understanding value drivers and estimating market value."
        )

    with col2:
        st.subheader("Objectives")
        st.markdown(
            "1. **Identify** key drivers of sale price.\n"
            "2. **Estimate** sale price for inherited houses.\n"
            "3. **Predict** sale price for new user inputs."
        )

    st.divider()

    # --- Business Requirements ---
    st.subheader("Business Requirements")
    st.markdown("The solution is designed to satisfy two core pillars:")

    st.info(
        "**1. Data Insights (Business Understanding)**\n\n"
        "Visualize relationships between attributes and `SalePrice` "
        "to highlight key drivers such as size, quality, and condition."
    )

    st.success(
        "**2. Predictive System (Regression Model)**\n\n"
        "Generate sale price predictions for Lydia's inherited houses and "
        "new user-defined property inputs."
    )

    st.divider()

    # --- Dashboard Features ---
    st.subheader("Dashboard Features")

    tabs = st.tabs(["Navigation", "Predictor Modes", "Transparency"])

    with tabs[0]:
        st.markdown(
            "**Recommended workflow:**\n\n"
            "Start with **Data Insights** to understand the data, then use "
            "**Price Predictor** to estimate prices, and finally review "
            "**Model Performance** for evaluation metrics."
        )

    with tabs[1]:
        st.markdown(
            "**Available modes:**\n\n"
            "- **Quick Estimate:** Fewer inputs for a fast estimate.\n"
            "- **Pro Estimate:** More inputs for a richer experience.\n"
            "- **Inherited Houses:** Batch prediction for Lydia’s 4 houses."
        )

    with tabs[2]:
        st.markdown(
            "**Model transparency:**\n\n"
            "The **Model Performance** page provides evaluation metrics "
            "(e.g., R² and error measures) to support trust in the "
            "model outputs."
        )

    st.divider()

    # --- Scope & Limitations ---
    with st.expander("Project scope and limitations"):
        st.write(
            "Predictions are based on historical Ames, Iowa data. "
            "The model provides a statistical estimate, not a "
            "guaranteed sale price. "
        )
        st.write(
            "External factors such as local market demand, renovations, "
            "and interest rates can impact real-world value."
        )
