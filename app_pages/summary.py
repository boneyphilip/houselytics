import streamlit as st


def render() -> None:
    st.header("Project Summary")

    st.write(
        "Client: Lydia Doe inherited four houses in Ames, Iowa, USA. "
        "She needs help understanding which features affect price and "
        "a system to predict sale price for her houses and future properties."
    )

    st.subheader("Business Requirements")
    st.markdown(
        "1. **Understand correlations** between house attributes and "
        "SalePrice (data visualisations).\n"
        "2. **Predict SalePrice** for Lydia’s 4 inherited houses and "
        "new user inputs (ML regression)."
    )

    st.subheader("What this dashboard provides")
    st.markdown(
        """
        - A data insights area to explore relationships with SalePrice
        - A price predictor (Quick + Pro)
        - A model performance page showing evaluation results
        """
    )
