import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render() -> None:
    st.header("Data Insights")

    train_df = pd.read_csv("data/processed/clean_train.csv")
    target = "SalePrice"

    feature_options = [c for c in train_df.columns if c != target]
    feature = st.selectbox("Select a feature to compare with SalePrice", feature_options)

    st.caption("This interactive plot helps answer Business Requirement 1.")

    fig = plt.figure()
    plt.scatter(train_df[feature], train_df[target], alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{feature} vs {target}")
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write(
        "Look for patterns: if points go upward as the feature increases, "
        "that feature may be positively correlated with SalePrice."
    )
