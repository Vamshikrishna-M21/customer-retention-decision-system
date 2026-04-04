from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.business.rules import assign_risk_band, estimate_expected_value, recommend_action


MODEL_BUNDLE_PATH = Path("models/artifacts/model_bundle.joblib")
FEATURE_IMPORTANCE_PATH = Path("models/artifacts/feature_importance.csv")


def load_model_bundle(path: Path | str = MODEL_BUNDLE_PATH) -> dict:
    return joblib.load(path)


def load_global_importance(path: Path | str = FEATURE_IMPORTANCE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def score_customer(input_frame: pd.DataFrame, bundle: dict | None = None) -> dict:
    bundle = bundle or load_model_bundle()
    pipeline = bundle["model"]
    threshold = bundle["threshold"]
    model_name = bundle["model_name"]

    churn_probability = float(pipeline.predict_proba(input_frame)[0, 1])
    risk_band = assign_risk_band(churn_probability)
    expected_value = estimate_expected_value(churn_probability, risk_band)

    response = {
        "model_name": model_name,
        "threshold": threshold,
        "churn_probability": churn_probability,
        "risk_band": risk_band,
        "recommended_action": recommend_action(risk_band),
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
        response["top_drivers"] = contributions.sort_values(
            "abs_contribution", ascending=False
        ).head(6)[["feature", "contribution"]]
    else:
        response["top_drivers"] = load_global_importance().head(6)[["feature", "importance"]]

    return response

