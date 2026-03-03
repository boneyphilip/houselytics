"""
Data Insights page (Streamlit).

Goal:
- Explain which features influence SalePrice in the dataset.
- Provide interactive charts + a data table.
- Satisfy Business Requirement 1.
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
    """Load the processed training dataset and cache it."""
    return pd.read_csv("data/processed/clean_train.csv")


# ------------------------------------------------------------
# 2) HELPERS: Labels & Charts
# ------------------------------------------------------------
def _pretty_label(feature: str) -> str:
    """Convert dataset column names into human-friendly labels."""
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


def _plotly_chart(fig) -> None:
    """Render Plotly charts in a responsive way."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _top_correlations(
    df: pd.DataFrame,
    target: str,
    n: int = 12,
) -> pd.Series:
    """Return the top 'n' signed correlations with the target."""
    corr_series = df.corr(numeric_only=True)[target].drop(
        labels=[target],
        errors="ignore",
    )
    top_features = corr_series.abs().sort_values(
        ascending=False,
    ).head(n).index
    return corr_series.reindex(top_features)


# ------------------------------------------------------------
# 3) MAIN PAGE FUNCTION
# ------------------------------------------------------------
def render() -> None:
    """Render the Data Insights page."""
    df = load_data()
    target = "SalePrice"

    # Robustness Check: Ensure target exists
    if target not in df.columns:
        st.error(f"Target column '{target}' is missing from the dataset.")
        return

    st.title("Data Insights")
    st.markdown(
        "Explore relationships between property attributes and market value. "
        "This page supports **Business Requirement 1** by highlighting the "
        "key drivers that influence house prices in Ames, Iowa."
    )

    # Quick Stats Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns in Dataset", f"{df.shape[1]:,}")
    with col3:
        st.metric("Primary Target", target)

    st.divider()

    # Section: Market Drivers
    st.subheader("Market Drivers")
    st.markdown(
        "Which features have the strongest relationship with price? "
        "Increase the slider to view more variables."
    )

    top_n = st.slider(
        "Number of features",
        min_value=5, max_value=20, value=10, key="ins_top_n",
    )

    top_corr = _top_correlations(df, target, n=top_n)

    # Polish: Use pretty labels on Y-axis
    fig_corr = px.bar(
        x=top_corr.values,
        y=[_pretty_label(col) for col in top_corr.index],
        orientation="h",
        labels={"x": "Correlation (signed)", "y": "Feature"},
        color=top_corr.values,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig_corr.update_layout(
        showlegend=False, height=420, margin=dict(t=10, b=10),
    )
    _plotly_chart(fig_corr)

    st.success(
        "This section identifies the strongest relationships with "
        "`SalePrice`, helping explain the main market value drivers."
    )
    st.caption(
        "Positive correlation means price generally rises with the feature. "
        "Negative correlation means price tends to fall as the feature rises."
    )

    st.divider()

    # Section: Feature Analysis Tool
    st.subheader("Feature Analysis Tool")
    numeric_features = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col != target
    ]

    if not numeric_features:
        st.warning("No numeric features found to analyze.")
        return

    default_idx = (
        numeric_features.index("GrLivArea")
        if "GrLivArea" in numeric_features else 0
    )

    col_feat, col_mode = st.columns([2, 1])
    with col_feat:
        feature = st.selectbox(
            "Select feature to explore",
            options=numeric_features,
            index=default_idx,
            format_func=_pretty_label,
            key="ins_feature",
        )
    with col_mode:
        plot_mode = st.radio(
            "Analysis Mode", ["Relationship", "Distribution"], key="ins_mode",
        )

    tab_plot, tab_data = st.tabs(["Visualization", "Data Sample"])

    with tab_plot:
        feature_label = _pretty_label(feature)
        f_corr = df[[feature, target]].corr(numeric_only=True).iloc[0, 1]
        st.caption(f"Correlation with SalePrice: {f_corr:.3f}")

        if plot_mode == "Relationship":
            fig = px.scatter(
                df, x=feature, y=target, opacity=0.5,
                labels={feature: feature_label, target: "Sale Price ($)"},
                template="plotly_dark",
            )
            fig.update_layout(height=480, margin=dict(t=20, b=20))
            st.markdown(f"**How '{feature_label}' relates to Sale Price**")
            _plotly_chart(fig)

            # Automated interpretation
            if abs(f_corr) > 0.60:
                st.success(
                    f"**Insight:** '{feature_label}' is a strong driver."
                )
            elif abs(f_corr) > 0.30:
                st.info(
                    f"**Insight:** '{feature_label}' has moderate impact."
                )
            else:
                st.warning(
                    f"**Insight:** '{feature_label}' has a weak link."
                )
        else:
            fig = px.histogram(
                df, x=feature, nbins=30,
                labels={feature: feature_label},
                template="plotly_dark",
            )
            fig.update_layout(height=480, margin=dict(t=20, b=20))
            st.markdown(f"**Distribution of '{feature_label}'**")
            _plotly_chart(fig)

    with tab_data:
        feature_label = _pretty_label(feature)
        st.markdown("### Historical Data Sample")
        sample_mode = st.radio(
            "Filter examples",
            ["Random 10", "Top 10 Prices", "Bottom 10 Prices"],
            key="ins_sample_mode", horizontal=True,
        )

        if sample_mode == "Random 10":
            s_df = df[[feature, target]].sample(10, random_state=42)
        elif sample_mode == "Top 10 Prices":
            s_df = df[[feature, target]].sort_values(
                target, ascending=False
            ).head(10)
        else:
            s_df = df[[feature, target]].sort_values(
                target, ascending=True
            ).head(10)

        display_df = s_df.rename(
            columns={feature: feature_label, target: "Sale price ($)"}
        ).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)

    st.divider()
    st.subheader("Insight Summary")
    st.success(
        "These insights confirm that Ames house prices are driven by a "
        "mix of size, quality, and age. This evidence supports better "
        "pricing decisions and provides a strong analytical foundation."
    )
