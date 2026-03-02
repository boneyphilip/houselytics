from __future__ import annotations

# -----------------------------
# Imports (libraries we need)
# -----------------------------
import joblib  # to load the saved ML model file (.pkl)
import numpy as np  # for math operations like square root
import pandas as pd  # for loading and working with tables (DataFrames)
import plotly.express as px  # for interactive charts
import streamlit as st  # for building the web dashboard
from sklearn.metrics import (
    mean_absolute_error,  # average error in money terms
    mean_squared_error,  # used to compute RMSE
    r2_score,  # "how well model explains price"
)
from sklearn.model_selection import (
    train_test_split,  # split data into train/test
)


# -----------------------------
# Load model once (cached)
# -----------------------------
@st.cache_resource
def load_model():
    """
    Load the saved ML model once.

    Streamlit reuses the cached object after the first load.
    """
    return joblib.load("src/house_price_model.pkl")


# -----------------------------
# Load dataset once (cached)
# -----------------------------
@st.cache_data
def load_train_data() -> pd.DataFrame:
    """
    Load the processed training dataset once.

    Caching avoids re-reading the CSV on every interaction.
    """
    return pd.read_csv("data/processed/clean_train.csv")


# -----------------------------
# Helper for responsive Plotly charts
# -----------------------------
def _plotly_chart(fig) -> None:
    """
    Render Plotly charts in a responsive way.

    Streamlit historically used: use_container_width=True
    Newer versions are moving toward: width="stretch"
    This helper works for both.
    """
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


def render() -> None:
    """
    Model Performance page.

    This page proves the model works by showing:
    - Metrics (R², MAE, RMSE)
    - Charts (Actual vs Predicted, Residuals)
    - Feature Importance (top drivers)
    """

    # -----------------------------
    # Page header and explanation
    # -----------------------------
    st.header("Model Performance")
    st.markdown(
        "This page evaluates the accuracy of the **final tuned Gradient "
        "Boosting model**. The dataset is split into **Train** (80%) and "
        "**Test** (20%) sets to measure how well the model performs on "
        "houses it has not seen before."
    )

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = load_train_data()
    target = "SalePrice"

    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataset.")
        return

    # -----------------------------
    # Separate features (X) and target (y)
    # -----------------------------
    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # Split into train and test sets
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------
    # Load trained model and predict
    # -----------------------------
    model = load_model()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # -----------------------------
    # Display metrics as cards
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Train R²", f"{r2_train:.3f}")
    with col2:
        st.metric("Test R²", f"{r2_test:.3f}")
    with col3:
        st.metric("Average Error (MAE)", f"${mae_test:,.0f}")
    with col4:
        st.metric("RMSE", f"${rmse_test:,.0f}")

    st.info(
        "Simple guide: R² closer to 1.0 is better. "
        "MAE is average money error. RMSE penalizes bigger mistakes more."
    )

    if r2_test >= 0.85:
        st.success(
            "The final tuned model successfully meets the predictive "
            "business requirement. "
            f"It achieved a **Test R² of {r2_test:.3f}**, which indicates "
            "strong predictive performance on unseen houses."
        )
    else:
        st.warning(
            "The model is usable, but its current "
            f"**Test R² of {r2_test:.3f}** suggests that further "
            "improvement may be needed."
        )

    st.divider()

    # -----------------------------
    # Chart 1: Actual vs Predicted
    # -----------------------------
    st.subheader("Visual Accuracy Check (Actual vs Predicted)")
    st.markdown(
        "Each dot is a house. If dots follow the red dashed line, "
        "the model is accurate."
    )

    results_df = pd.DataFrame(
        {"Actual": y_test, "Predicted": y_pred_test},
    )

    fig_scatter = px.scatter(
        results_df,
        x="Actual",
        y="Predicted",
        opacity=0.6,
        template="plotly_dark",
        labels={"Actual": "Real Price ($)", "Predicted": "Model Guess ($)"},
    )

    lim_min = min(y_test.min(), y_pred_test.min())
    lim_max = max(y_test.max(), y_pred_test.max())
    fig_scatter.add_shape(
        type="line",
        x0=lim_min,
        y0=lim_min,
        x1=lim_max,
        y1=lim_max,
        line=dict(color="red", dash="dash"),
    )
    fig_scatter.update_layout(height=520)

    _plotly_chart(fig_scatter)

    st.divider()

    # -----------------------------
    # Chart 2: Residuals
    # -----------------------------
    st.subheader("The 'Miss' Map (Residuals)")
    st.markdown(
        "A **Residual** is simply the difference between the Real Price "
        "and the Model's Guess. This chart helps identify if the model "
        "is biased toward over-pricing or under-pricing certain houses."
    )

    st.latex(r"\text{Residual} = y_{\text{actual}} - y_{\text{predicted}}")

    st.write(
        "* **Dots on the 0 line:** The guess was exactly right.\n"
        "* **Dots ABOVE the line:** The house sold for MORE than predicted.\n"
        "* **Dots BELOW the line:** The house sold for LESS than predicted."
    )

    res_df = pd.DataFrame(
        {
            "Predicted": y_pred_test,
            "Residual": (y_test - y_pred_test),
        }
    )

    fig_res = px.scatter(
        res_df,
        x="Predicted",
        y="Residual",
        opacity=0.6,
        template="plotly_dark",
        labels={"Predicted": "Model Guess ($)", "Residual": "Error ($)"},
    )

    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    fig_res.update_layout(height=480)

    _plotly_chart(fig_res)

    st.divider()

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("Price Drivers (Feature Importance)")
    st.markdown(
        "This shows which features the model relied on most. "
        "Higher importance means stronger influence in predictions."
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = X.columns

        imp_df = pd.DataFrame(
            {"Feature": feat_names, "Importance": importances},
        ).sort_values(by="Importance", ascending=False)

        top_k = st.slider(
            "Show top features",
            5,
            20,
            10,
            key="perf_slider",
        )

        imp_top = imp_df.head(top_k)

        fig_imp = px.bar(
            imp_top,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark",
            color="Importance",
            color_continuous_scale="Viridis",
        )

        fig_imp.update_layout(
            showlegend=False,
            yaxis_autorange="reversed",
            height=450,
        )

        _plotly_chart(fig_imp)

        top_features = imp_df.head(3)["Feature"].tolist()
        st.markdown(
            "**Top price drivers identified by the model:** "
            f"{', '.join(top_features)}."
        )
    else:
        st.warning("This model does not provide feature importance values.")

    st.divider()

    # -----------------------------
    # Final Conclusion
    # -----------------------------
    st.subheader("Final Assessment")
    st.success(
        f"The final tuned model achieved **Train R² = {r2_train:.3f}** "
        f"and **Test R² = {r2_test:.3f}**, with an average test error of "
        f"**${mae_test:,.0f}**. This indicates strong predictive "
        "performance and makes the model suitable for guiding house "
        "price estimation. Predictions should still be used as informed "
        "guidance rather than a guaranteed sale price."
    )
