"""
Project Hypothesis Page.

Purpose:
- Present 3 clear project hypotheses
- Demonstrate data-driven validation using charts and correlation
- Support Merit/Distinction assessment criteria
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Load the cleaned dataset once (cached)
# ------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed training dataset.

    Why cache?
    - Streamlit reruns scripts often
    - Caching prevents repeated CSV loading
    - This makes page switching faster
    """
    return pd.read_csv("data/processed/clean_train.csv")


# ------------------------------------------------------------
# Plotly helper for responsive charts
# ------------------------------------------------------------
def _plotly_chart(fig) -> None:
    """
    Render Plotly charts in a responsive way.

    Streamlit is moving from use_container_width -> width.
    This helper supports both old and new versions safely.
    """
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Correlation helper
# ------------------------------------------------------------
def _get_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> float:
    """
    Return the correlation value between one feature and the target.

    A positive value means both tend to increase together.
    A negative value means one tends to decrease as the other increases.
    """
    return df[[feature, target]].corr(numeric_only=True).iloc[0, 1]


# ------------------------------------------------------------
# Main page renderer
# ------------------------------------------------------------
def render() -> None:
    """
    Build the Project Hypothesis page UI.
    """
    # Page title
    st.title("Project Hypothesis & Validation")

    # Short page explanation
    st.markdown(
        "This page presents the key project hypotheses and evaluates "
        "whether they are supported by the dataset using visual analysis "
        "and correlation."
    )

    # Simple explanation box for non-technical users
    st.info(
        "A hypothesis is a data-based assumption. "
        "This page tests each assumption using charts and a correlation "
        "score (a mathematical value between -1 and 1)."
    )

    # Load dataset
    df = load_data()
    target = "SalePrice"

    # Safety check: stop if target is missing
    if target not in df.columns:
        st.error(f"Target column '{target}' is missing from the dataset.")
        return

    # ============================================================
    # HYPOTHESIS 1: CONSTRUCTION QUALITY
    # ============================================================
    st.divider()
    st.subheader("Hypothesis 1: Construction Quality")
    st.markdown(
        "**H1:** Houses with higher quality scores (OverallQual) "
        "tend to have higher sale prices."
    )

    # Only test if the required column exists
    if "OverallQual" in df.columns:
        # Box plot is useful here because OverallQual is a score/category
        # It shows median, spread, and outliers for each quality level
        fig_h1 = px.box(
            df,
            x="OverallQual",
            y=target,
            template="plotly_dark",
            labels={
                "OverallQual": "Quality Score",
                target: "Sale Price ($)",
            },
        )
        fig_h1.update_layout(height=450)
        _plotly_chart(fig_h1)

        # Calculate correlation between quality and sale price
        corr_h1 = _get_correlation(df, "OverallQual", target)
        st.caption(f"Correlation score: {corr_h1:.3f}")

        # Conclusion based on correlation strength
        if corr_h1 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Construction quality "
                "shows a strong positive relationship with sale price."
            )
        elif corr_h1 > 0.30:
            st.success(
                "Conclusion: Supported. Construction quality has a clear "
                "positive relationship with sale price."
            )
        else:
            st.warning(
                "Conclusion: The relationship is weaker than expected in "
                "this dataset."
            )
    else:
        st.warning("OverallQual column is missing, so H1 cannot be tested.")

    # ============================================================
    # HYPOTHESIS 2: LIVING AREA
    # ============================================================
    st.divider()
    st.subheader("Hypothesis 2: Living Area Size")
    st.markdown(
        "**H2:** As total living area (GrLivArea) increases, "
        "sale price generally tends to increase."
    )

    if "GrLivArea" in df.columns:
        # Scatter plot is ideal for comparing two continuous variables
        # It helps reveal upward/downward trends in the data
        fig_h2 = px.scatter(
            df,
            x="GrLivArea",
            y=target,
            opacity=0.5,
            template="plotly_dark",
            labels={
                "GrLivArea": "Living Area (sq ft)",
                target: "Sale Price ($)",
            },
        )
        fig_h2.update_layout(height=450)
        _plotly_chart(fig_h2)

        # Calculate correlation between living area and sale price
        corr_h2 = _get_correlation(df, "GrLivArea", target)
        st.caption(f"Correlation score: {corr_h2:.3f}")

        # Dynamic conclusion based on the observed correlation
        if corr_h2 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Larger homes tend to "
                "sell for higher prices."
            )
        elif corr_h2 > 0.30:
            st.success(
                "Conclusion: Supported. Living area shows a positive "
                "relationship with sale price."
            )
        else:
            st.warning(
                "Conclusion: The relationship is weaker than expected in "
                "this dataset."
            )
    else:
        st.warning("GrLivArea column is missing, so H2 cannot be tested.")

    # ============================================================
    # HYPOTHESIS 3: YEAR BUILT
    # ============================================================
    st.divider()
    st.subheader("Hypothesis 3: Year Built")
    st.markdown(
        "**H3:** Newer houses (YearBuilt) tend to sell for higher prices "
        "than older houses."
    )

    if "YearBuilt" in df.columns:
        # Another scatter plot works well here because YearBuilt is numeric
        # This helps show whether newer build years align with higher prices
        fig_h3 = px.scatter(
            df,
            x="YearBuilt",
            y=target,
            opacity=0.5,
            template="plotly_dark",
            labels={
                "YearBuilt": "Year Built",
                target: "Sale Price ($)",
            },
        )
        fig_h3.update_layout(height=450)
        _plotly_chart(fig_h3)

        # Calculate correlation between build year and sale price
        corr_h3 = _get_correlation(df, "YearBuilt", target)
        st.caption(f"Correlation score: {corr_h3:.3f}")

        # Dynamic conclusion based on correlation
        if corr_h3 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Newer construction is "
                "strongly linked with higher sale prices."
            )
        elif corr_h3 > 0.30:
            st.success(
                "Conclusion: Supported. Newer houses tend to show "
                "higher market value."
            )
        else:
            st.warning(
                "Conclusion: The relationship is weaker than expected in "
                "this dataset."
            )
    else:
        st.warning("YearBuilt column is missing, so H3 cannot be tested.")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    st.divider()
    st.subheader("Executive Summary")
    st.write(
        "This analysis indicates that construction quality, living area, "
        "and build year all show positive relationships with sale price. "
        "These findings support the business case that quality, size, and "
        "property age are important market value drivers."
    )
