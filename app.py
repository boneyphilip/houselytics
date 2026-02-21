import streamlit as st

from app_pages.summary import render as render_summary
from app_pages.insights import render as render_insights
from app_pages.predictor import render as render_predictor
from app_pages.performance import render as render_performance


# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------------------------------
st.set_page_config(page_title="Houselytics", layout="wide")


# -------------------------------------------------
# GLOBAL UI THEME (Premium Dark)
# -------------------------------------------------
st.markdown("""
<style>
/* Page background */
.stApp {
    background: radial-gradient(
    1200px circle at 20% 0%,
    #0f1b33 0%,
    #070b14 55%,
    #05070d 100%
);

}

/* Main container spacing */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e2a44;
}
section[data-testid="stSidebar"] * {
    font-size: 0.95rem;
}

/* Titles */
.h-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.h-subtitle {
    text-align: center;
    color: #9aa6bf;
    margin-top: 0px;
    margin-bottom: 26px;
}

/* Divider line */
.hr {
    height: 1px;
    background: linear-gradient(90deg, transparent, #22314f, transparent);
    margin: 18px 0 26px 0;
}

/* Cards */
.card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(45, 60, 92, 0.55);
    border-radius: 14px;
    padding: 18px 18px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: #041018;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 700;
}
.stButton>button:hover {
    filter: brightness(1.05);
}

/* Inputs subtle border */
div[data-baseweb="input"], div[data-baseweb="select"],
div[data-baseweb="textarea"] {
    border-radius: 10px !important;
}

/* Metric box (clean, no ugly big box) */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(45, 60, 92, 0.5);
    border-radius: 14px;
    padding: 12px 14px;
}
</style>
""", unsafe_allow_html=True)


def main() -> None:
    # Header
    st.markdown(
        "<div class='h-title'>Houselytics</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='h-subtitle'>AI Powered Property Valuation "
        "Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Pages
    pages = {
        "Project Summary": render_summary,
        "Data Insights": render_insights,
        "Price Predictor": render_predictor,
        "Model Performance": render_performance,
    }

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        choice = st.radio("Go to", list(pages.keys()))
        st.caption("Project 5 — Predictive Analytics")

    pages[choice]()


if __name__ == "__main__":
    main()
