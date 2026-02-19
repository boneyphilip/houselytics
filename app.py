import streamlit as st

from app_pages.summary import render as render_summary
from app_pages.insights import render as render_insights
from app_pages.predictor import render as render_predictor
from app_pages.performance import render as render_performance


def main() -> None:
    st.set_page_config(page_title="Houselytics", layout="wide")

    # Minimal clean styling (no boxes)
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .app-title { text-align:center; font-size:44px; font-weight:800;
                     margin-bottom:0.2rem; }
        .app-subtitle { text-align:center; color:#6b7280; margin-top:0;
                        margin-bottom:1.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='app-title'>Houselytics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-subtitle'>AI Powered Property Valuation "
        "Dashboard</div>",
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    pages = {
        "Project Summary": render_summary,
        "Data Insights": render_insights,
        "Price Predictor": render_predictor,
        "Model Performance": render_performance,
    }

    with st.sidebar:
        st.header("Navigation")
        choice = st.radio("Go to", list(pages.keys()))
        st.caption("Project 5 — Predictive Analytics")

    pages[choice]()


if __name__ == "__main__":
    main()
