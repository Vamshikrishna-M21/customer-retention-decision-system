from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.business.rules import (
    RetentionAssumptions,
    assign_risk_band,
    estimate_customer_value,
    estimate_expected_value,
    explain_action_rule,
    recommend_action,
)


MODEL_BUNDLE_PATH = Path("models/artifacts/model_bundle.joblib")
FEATURE_IMPORTANCE_PATH = Path("models/artifacts/feature_importance.csv")


def load_model_bundle(path: Path | str = MODEL_BUNDLE_PATH) -> dict:
    return joblib.load(path)


def load_global_importance(path: Path | str = FEATURE_IMPORTANCE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def format_feature_name(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "")

    if feature_name.startswith("cat__"):
        cleaned = feature_name.replace("cat__", "")
        parts = cleaned.split("_", 1)
        if len(parts) == 2:
            feature, value = parts
            return f"{feature} = {value}"
        return cleaned

    return feature_name


def score_customer(input_frame: pd.DataFrame, bundle: dict | None = None) -> dict:
    bundle = bundle or load_model_bundle()
    pipeline = bundle["model"]
    threshold = bundle["threshold"]
    model_name = bundle["model_name"]
    assumptions = RetentionAssumptions()

    churn_probability = float(pipeline.predict_proba(input_frame)[0, 1])
    risk_band = assign_risk_band(churn_probability)
    customer_value = estimate_customer_value(input_frame.iloc[0], assumptions)
    expected_value = estimate_expected_value(
        churn_probability,
        risk_band,
        customer_value,
        assumptions,
    )

    response = {
        "model_name": model_name,
        "threshold": threshold,
        "churn_probability": churn_probability,
        "risk_band": risk_band,
        "estimated_customer_value": customer_value,
        "recommended_action": recommend_action(risk_band),
        "action_reason": explain_action_rule(risk_band),
        "expected_value_usd": expected_value,
    }

    if model_name == "logistic_regression":
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(input_frame)[0]
        feature_names = preprocessor.get_feature_names_out()
        contributions = pd.DataFrame(
            {
                "feature": feature_names,
                "contribution": transformed * model.coef_[0],
            }
        )
        contributions["abs_contribution"] = contributions["contribution"].abs()
        top_drivers = contributions.sort_values("abs_contribution", ascending=False).head(6).copy()
        top_drivers["feature"] = top_drivers["feature"].apply(format_feature_name)
        top_drivers["effect"] = top_drivers["contribution"].apply(
            lambda value: "Raises churn risk" if value > 0 else "Lowers churn risk"
        )
        response["top_drivers"] = top_drivers[["feature", "effect", "contribution"]]
    else:
        top_drivers = load_global_importance().head(6).copy()
        top_drivers["feature"] = top_drivers["feature"].apply(format_feature_name)
        top_drivers["effect"] = top_drivers["importance"].apply(
            lambda value: "Raises churn risk" if value > 0 else "Lowers churn risk"
        )
        response["top_drivers"] = top_drivers[["feature", "effect", "importance"]]

    return response
