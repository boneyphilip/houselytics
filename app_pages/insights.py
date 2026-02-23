"""
Data Insights page (Streamlit).

Goal:
- Explain which features influence SalePrice in the dataset.
- Provide interactive charts + a beginner-friendly data table.

This file is imported by app.py and called using render().
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# 1) DATA LOADING (Cached)
# ------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed training dataset and cache it.

    Why caching?
    - Streamlit reruns the script many times when you interact.
    - Caching avoids re-reading the CSV again and again.
    """
    return pd.read_csv("data/processed/clean_train.csv")


# ------------------------------------------------------------
# 2) HELPER: Convert raw column name to a nicer label
# ------------------------------------------------------------
def _pretty_label(feature: str) -> str:
    """
    Convert dataset column names into human-friendly labels.

    Example:
    - '1stFlrSF' becomes '1st floor area (sq ft)'
    """
    mapping = {
        "1stFlrSF": "1st floor area (sq ft)",
        "2ndFlrSF": "2nd floor area (sq ft)",
        "GrLivArea": "Total living area (sq ft)",
        "GarageArea": "Garage area (sq ft)",
        "LotArea": "Lot size (sq ft)",
        "TotalBsmtSF": "Basement area (sq ft)",
        "YearBuilt": "Year built",
        "YearRemodAdd": "Remodel year",
        "OverallQual": "Construction quality (1-10)",
        "OverallCond": "Overall condition (1-9)",
        "BedroomAbvGr": "Bedrooms above ground",
        "FullBath": "Full bathrooms",
        "HalfBath": "Half bathrooms",
        "TotRmsAbvGrd": "Total rooms above ground",
    }
    return mapping.get(feature, feature)


# ------------------------------------------------------------
# 3) HELPER: Find top correlations with SalePrice
# ------------------------------------------------------------
def _top_correlations(
    df: pd.DataFrame,
    target: str,
    n: int = 12,
) -> pd.Series:
    """
    Return the top 'n' correlations with the target.

    Important:
    - We sort by absolute correlation strength (abs value),
      but we keep the sign (+/-) for the chart.

    Output:
    - Pandas Series where:
      index = feature name
      value = signed correlation value
    """
    corr_series = df.corr(numeric_only=True)[target].drop(
        labels=[target],
        errors="ignore",
    )

    # Sort by absolute correlation strength, pick top N.
    top_features = corr_series.abs().sort_values(
        ascending=False,
    ).head(n).index

    # Reindex back using original signed values.
    return corr_series.reindex(top_features)


# ------------------------------------------------------------
# 4) MAIN PAGE FUNCTION
# ------------------------------------------------------------
def render() -> None:
    """
    Render the Data Insights page.

    This function is called from app.py when user selects
    "Data Insights" from sidebar navigation.
    """
    # Load dataset
    df = load_data()

    # Target column in this dataset (what we predict)
    target = "SalePrice"

    # -----------------------
    # Page Title + Intro
    # -----------------------
    st.title("Data Insights")
    st.markdown(
        "Explore relationships between property attributes and market value. "
        "This analysis highlights key drivers that influence house prices "
        "in Ames, Iowa."
    )

    # -----------------------
    # Quick Stats Row
    # -----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        # Total number of rows (houses)
        st.metric("Total Records", f"{df.shape[0]:,}")

    with col2:
        # Total number of columns (features + target)
        st.metric("Features Available", f"{df.shape[1]:,}")

    with col3:
        # Name of the target column
        st.metric("Primary Target", target)

    st.divider()

    # -----------------------
    # Section: Market Drivers
    # -----------------------
    st.subheader("Market Drivers")
    st.markdown(
        "Which features have the strongest relationship with price? "
        "Increase the slider to view more variables."
    )

    # Slider lets user choose how many top features to show.
    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=20,
        value=10,
        key="ins_top_n",
    )

    # Compute top correlations
    top_corr = _top_correlations(df, target, n=top_n)

    # Plotly horizontal bar chart
    fig_corr = px.bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation="h",
        labels={"x": "Correlation (signed)", "y": "Feature"},
        color=top_corr.values,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )

    # Make chart look neat (height + margins)
    fig_corr.update_layout(
        showlegend=False,
        height=420,
        margin=dict(t=10, b=10),
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Explain correlation meaning in simple words
    st.caption(
        "Correlation measures relationship strength. "
        "Positive (right) means price rises with the feature; "
        "negative (left) means price falls as the feature rises."
    )

    st.info(
        "**Pro-Tip:** Strong correlations (above 0.6 magnitude) are often "
        "good signals, but the model uses many features together."
    )

    st.divider()

    # -----------------------
    # Section: Feature Analysis Tool
    # -----------------------
    st.subheader("Feature Analysis Tool")
    st.markdown(
        "Pick one feature and see how it behaves with price. "
        "Use Visualization for charts, and Data Sample for real examples."
    )

    # Create a list of numeric features (exclude target)
    numeric_features = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col != target
    ]

    # Safety check: If no numeric features found, stop page gracefully.
    if not numeric_features:
        st.warning("No numeric features found to analyze.")
        return

    # Prefer GrLivArea as default if it exists (nice for demos).
    if "GrLivArea" in numeric_features:
        default_idx = numeric_features.index("GrLivArea")
    else:
        default_idx = 0

    # Two columns for controls: feature dropdown + analysis mode radio
    col_feat, col_mode = st.columns([2, 1])

    with col_feat:
        feature = st.selectbox(
            "Select feature to explore",
            options=numeric_features,
            index=default_idx,
            key="ins_feature",
            # format_func shows nicer names in dropdown
            format_func=_pretty_label,
        )

    with col_mode:
        plot_mode = st.radio(
            "Analysis Mode",
            ["Relationship", "Distribution"],
            key="ins_mode",
        )

    # Tabs: one for charts, one for example rows
    tab_plot, tab_data = st.tabs(["Visualization", "Data Sample"])

    # -----------------------
    # Tab 1: Visualization
    # -----------------------
    with tab_plot:
        feature_label = _pretty_label(feature)

        if plot_mode == "Relationship":
            # Scatter plot: feature vs sale price
            fig = px.scatter(
                df,
                x=feature,
                y=target,
                opacity=0.5,
                labels={feature: feature_label, target: "Sale Price ($)"},
                template="plotly_dark",
            )
            fig.update_layout(height=480, margin=dict(t=20, b=20))

            st.markdown(f"**How '{feature_label}' relates to Sale Price**")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histogram: distribution of selected feature
            fig = px.histogram(
                df,
                x=feature,
                nbins=30,
                labels={feature: feature_label},
                template="plotly_dark",
            )
            fig.update_layout(height=480, margin=dict(t=20, b=20))

            st.markdown(
                f"**Distribution of '{feature_label}' in the dataset**"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Tab 2: Data Sample (Beginner-friendly table)
    # -----------------------
    with tab_data:
        feature_label = _pretty_label(feature)

        st.markdown("### What this table shows (simple explanation)")
        st.write(
            "This table shows **real historical data** from the dataset "
            "(not a prediction). Each row represents one house record. "
            "The left column is the selected feature value and the right "
            "column is the **actual sale price** for that house."
        )

        # Give user control over which examples to see.
        sample_mode = st.radio(
            "Choose examples to show",
            [
                "Random 10 examples",
                "Top 10 highest prices",
                "Bottom 10 lowest prices",
            ],
            key="ins_sample_mode",
            horizontal=True,
        )

        # Prepare sample dataframe depending on selection.
        if sample_mode == "Random 10 examples":
            sample_df = df[[feature, target]].sample(10, random_state=42)
        elif sample_mode == "Top 10 highest prices":
            sample_df = (
                df[[feature, target]]
                .sort_values(by=target, ascending=False)
                .head(10)
            )
        else:
            sample_df = (
                df[[feature, target]]
                .sort_values(by=target, ascending=True)
                .head(10)
            )

        # Rename columns for readability.
        display_df = sample_df.rename(
            columns={
                feature: feature_label,
                target: "Sale price ($)",
            }
        ).reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True)

        st.caption(
            "Tip: Use the Visualization tab to see the trend across "
            "ALL houses."
        )

    st.divider()
