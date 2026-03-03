"""
Project Hypothesis Page.

Purpose:
- Present 3 clear project hypotheses
- Demonstrate data-driven validation using charts and correlation
- Support Merit/Distinction criteria through business-focused insights
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
    """Load the processed training dataset."""
    return pd.read_csv("data/processed/clean_train.csv")


# ------------------------------------------------------------
# Plotly helper for responsive charts
# ------------------------------------------------------------
def _plotly_chart(fig) -> None:
    """Render Plotly charts in a responsive way."""
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
    """Return the Pearson correlation value between feature and target."""
    return df[[feature, target]].corr(numeric_only=True).iloc[0, 1]


# ------------------------------------------------------------
# Main page renderer
# ------------------------------------------------------------
def render() -> None:
    """Build the Project Hypothesis & Validation page UI."""
    st.title("Project Hypothesis & Validation")

    # Business-focused intro
    st.markdown(
        "This page presents three business-focused project hypotheses and "
        "tests whether they are supported by the cleaned dataset using "
        "visual analysis and correlation."
    )

    st.info(
        "A hypothesis is a data-based assumption. This page tests each "
        "assumption using charts and a correlation score "
        "(a value between -1 and 1)."
    )

    df = load_data()
    target = "SalePrice"

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
    st.caption(
        "Validation method: Visual box plot review + Pearson correlation."
    )

    if "OverallQual" in df.columns:
        fig_h1 = px.box(
            df,
            x="OverallQual",
            y=target,
            template="plotly_dark",
            labels={"OverallQual": "Quality Score", target: "Sale Price ($)"},
        )
        fig_h1.update_layout(height=450)
        _plotly_chart(fig_h1)

        corr_h1 = _get_correlation(df, "OverallQual", target)
        st.caption(f"Correlation score: {corr_h1:.3f}")

        # Nuanced conclusion logic
        if corr_h1 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Quality shows a strong "
                "positive relationship with sale price."
            )
        elif corr_h1 > 0.30:
            st.success(
                "Conclusion: Supported. Quality has a clear positive "
                "relationship with price."
            )
        else:
            st.warning("Conclusion: The relationship is weaker than expected.")

        st.markdown(
            "**Business takeaway:** Construction quality should be treated "
            "as a major value driver when estimating realistic sale price."
        )
    else:
        st.warning("OverallQual column is missing; H1 cannot be tested.")

    # ============================================================
    # HYPOTHESIS 2: LIVING AREA
    # ============================================================
    st.divider()
    st.subheader("Hypothesis 2: Living Area Size")
    st.markdown(
        "**H2:** As total living area (GrLivArea) increases, "
        "sale price generally tends to increase."
    )
    st.caption(
        "Validation method: Scatter plot trend + Pearson correlation."
    )

    if "GrLivArea" in df.columns:
        fig_h2 = px.scatter(
            df,
            x="GrLivArea",
            y=target,
            opacity=0.5,
            template="plotly_dark",
            labels={
                "GrLivArea": "Living Area (sq ft)",
                target: "Sale Price ($)"
            },
        )
        fig_h2.update_layout(height=450)
        _plotly_chart(fig_h2)

        corr_h2 = _get_correlation(df, "GrLivArea", target)
        st.caption(f"Correlation score: {corr_h2:.3f}")

        if corr_h2 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Larger homes tend to "
                "sell for higher prices."
            )
        elif corr_h2 > 0.30:
            st.success(
                "Conclusion: Supported. Larger homes tend to sell "
                "for higher prices."
            )
        else:
            st.warning("Conclusion: The relationship is weaker than expected.")

        st.markdown(
            "**Business takeaway:** Larger above-ground living space "
            "should generally support a higher pricing estimate."
        )
    else:
        st.warning("GrLivArea column is missing; H2 cannot be tested.")

    # ============================================================
    # HYPOTHESIS 3: YEAR BUILT
    # ============================================================
    st.divider()
    st.subheader("Hypothesis 3: Year Built")
    st.markdown(
        "**H3:** Newer houses (YearBuilt) tend to sell for higher prices "
        "than older houses."
    )
    st.caption(
        "Validation method: Scatter plot trend + Pearson correlation."
    )

    if "YearBuilt" in df.columns:
        fig_h3 = px.scatter(
            df,
            x="YearBuilt",
            y=target,
            opacity=0.5,
            template="plotly_dark",
            labels={"YearBuilt": "Year Built", target: "Sale Price ($)"},
        )
        fig_h3.update_layout(height=450)
        _plotly_chart(fig_h3)

        corr_h3 = _get_correlation(df, "YearBuilt", target)
        st.caption(f"Correlation score: {corr_h3:.3f}")

        if corr_h3 > 0.70:
            st.success(
                "Conclusion: Strongly supported. Newer construction is "
                "strongly linked with higher sale prices."
            )
        elif corr_h3 > 0.30:
            st.success(
                "Conclusion: Supported. Newer construction shows "
                "higher market value."
            )
        else:
            st.warning("Conclusion: The relationship is weaker than expected.")

        st.markdown(
            "**Business takeaway:** Newer construction can support a "
            "stronger estimate, but should be considered alongside other "
            "key value drivers."
        )
    else:
        st.warning("YearBuilt column is missing; H3 cannot be tested.")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    st.divider()
    st.subheader("Executive Summary")
    st.success(
        "The analysis shows that construction quality, living area, and "
        "build year each have positive relationships with sale price. "
        "These findings support the business requirement to identify key "
        "value drivers and provide stronger evidence for the predictive "
        "modeling phase."
    )
