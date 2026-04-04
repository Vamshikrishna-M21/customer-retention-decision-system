from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RetentionAssumptions:
    customer_lifetime_value: float = 1200.0
    high_risk_offer_cost: float = 120.0
    medium_risk_offer_cost: float = 40.0
    high_risk_success_rate: float = 0.35
    medium_risk_success_rate: float = 0.18


def assign_risk_band(churn_probability: float) -> str:
    if churn_probability >= 0.65:
        return "high"
    if churn_probability >= 0.4:
        return "medium"
    return "low"


def recommend_action(risk_band: str) -> str:
    actions = {
        "high": "Offer a targeted retention discount and proactive support outreach.",
        "medium": "Send a personalized retention email with plan-fit messaging.",
        "low": "Monitor behavior and avoid unnecessary incentive spend.",
    }
    return actions[risk_band]


def estimate_expected_value(
    churn_probability: float,
    risk_band: str,
    assumptions: RetentionAssumptions | None = None,
) -> float:
    assumptions = assumptions or RetentionAssumptions()

    if risk_band == "high":
        saved_value = (
            churn_probability
            * assumptions.customer_lifetime_value
            * assumptions.high_risk_success_rate
        )
        return saved_value - assumptions.high_risk_offer_cost

    if risk_band == "medium":
        saved_value = (
            churn_probability
            * assumptions.customer_lifetime_value
            * assumptions.medium_risk_success_rate
        )
        return saved_value - assumptions.medium_risk_offer_cost

    return 0.0


def build_retention_frame(
    customer_frame: pd.DataFrame,
    churn_probabilities: pd.Series,
) -> pd.DataFrame:
    output = customer_frame.copy()
    output["churn_probability"] = churn_probabilities
    output["risk_band"] = output["churn_probability"].apply(assign_risk_band)
    output["recommended_action"] = output["risk_band"].apply(recommend_action)
    output["expected_value_usd"] = output.apply(
        lambda row: estimate_expected_value(row["churn_probability"], row["risk_band"]),
        axis=1,
    )
    return output

