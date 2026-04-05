from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_clean_data
from src.models.inference import MODEL_BUNDLE_PATH, score_customer


SUMMARY_PATH = Path("models/artifacts/training_summary.json")
FIGURE_DIR = Path("reports/figures")


def build_default_customer(data: pd.DataFrame) -> dict:
    defaults: dict[str, object] = {}
    for column in data.columns:
        if data[column].dtype.kind in {"i", "f"}:
            defaults[column] = float(data[column].median())
        else:
            defaults[column] = data[column].mode().iat[0]
    return defaults


def load_training_summary() -> dict:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_customer_form(data: pd.DataFrame) -> pd.DataFrame:
    defaults = build_default_customer(data)
    input_frame = {}

    col1, col2 = st.columns(2)

    with col1:
        input_frame["gender"] = st.selectbox("Gender", sorted(data["gender"].unique()), index=0)
        input_frame["SeniorCitizen"] = st.selectbox("Senior citizen", [0, 1], index=0)
        input_frame["Partner"] = st.selectbox("Partner", sorted(data["Partner"].unique()))
        input_frame["Dependents"] = st.selectbox("Dependents", sorted(data["Dependents"].unique()))
        input_frame["tenure"] = st.slider("Tenure (months)", 0, 72, int(defaults["tenure"]))
        input_frame["PhoneService"] = st.selectbox(
            "Phone service",
            sorted(data["PhoneService"].unique()),
        )
        input_frame["MultipleLines"] = st.selectbox(
            "Multiple lines",
            sorted(data["MultipleLines"].unique()),
        )
        input_frame["InternetService"] = st.selectbox(
            "Internet service",
            sorted(data["InternetService"].unique()),
        )

    with col2:
        for column in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]:
            input_frame[column] = st.selectbox(column, sorted(data[column].unique()))

        input_frame["MonthlyCharges"] = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=float(defaults["MonthlyCharges"]),
            step=1.0,
        )
        input_frame["TotalCharges"] = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(defaults["TotalCharges"]),
            step=10.0,
        )

    return pd.DataFrame([input_frame])


def main() -> None:
    st.set_page_config(page_title="Customer Retention ML System", layout="wide")
    st.title("Customer Retention ML System")
    st.caption("Predict churn risk and turn it into a retention action recommendation.")

    if not MODEL_BUNDLE_PATH.exists():
        st.error("Model artifacts are missing. Run `python -m src.models.train` first.")
        return

    data = load_clean_data().drop(columns=["customerID", "Churn"])
    summary = load_training_summary()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Selected model", summary["selected_model"].replace("_", " ").title())
    metric_col2.metric("ROC-AUC", f"{summary['test_roc_auc']:.3f}")
    metric_col3.metric("PR-AUC", f"{summary['test_pr_auc']:.3f}")
    metric_col4.metric("Best threshold value", f"${summary['best_threshold_net_value_usd']:.0f}")

    st.subheader("Score a customer profile")
    customer_frame = build_customer_form(data)

    if st.button("Run prediction", type="primary"):
        result = score_customer(customer_frame)

        pred_col1, pred_col2, pred_col3 = st.columns(3)
        pred_col1.metric("Churn probability", f"{result['churn_probability']:.1%}")
        pred_col2.metric("Risk band", result["risk_band"].title())
        pred_col3.metric("Expected value", f"${result['expected_value_usd']:.2f}")

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric("Estimated customer value", f"${result['estimated_customer_value']:.2f}")
        detail_col2.metric("Decision threshold used in training", f"{result['threshold']:.2f}")
        detail_col3.metric("Best business threshold", f"{summary['best_business_threshold']:.2f}")

        st.markdown(f"**Recommended action:** {result['recommended_action']}")
        st.markdown(f"**Why this action:** {result['action_reason']}")

        st.subheader("Top prediction drivers")
        st.dataframe(result["top_drivers"], hide_index=True, use_container_width=True)

    st.subheader("Project visuals")
    image_names = [
        "model_comparison_pr_auc.png",
        "threshold_tradeoff.png",
        "threshold_economics.png",
        "top_feature_importance.png",
    ]
    first_row = st.columns(2)
    second_row = st.columns(2)

    for column, image_name in zip(first_row + second_row, image_names):
        image_path = FIGURE_DIR / image_name
        if image_path.exists():
            column.image(str(image_path), use_column_width=True)


if __name__ == "__main__":
    main()
