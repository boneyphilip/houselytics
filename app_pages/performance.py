import streamlit as st
import json


def render() -> None:
    st.header("Model Performance")

    with open("src/model_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    st.subheader("Scores (from modeling notebook)")
    col1, col2 = st.columns(2)
    col1.metric("Train R²", f"{report['r2_train']:.3f}")
    col2.metric("Test R²", f"{report['r2_test']:.3f}")

    st.write(
        "A higher R² means the model explains more variance in house prices. "
        "Test score matters most because it shows performance on unseen data."
    )

    st.subheader("Conclusion")
    st.write(
        "This model demonstrates strong predictive performance and provides "
        "tangible value by estimating sale prices for Lydia’s inherited houses and new user inputs."
    )
