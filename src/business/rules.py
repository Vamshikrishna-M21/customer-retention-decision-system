from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RetentionAssumptions:
    remaining_months_by_contract: dict[str, int] = field(
        default_factory=lambda: {
            "Month-to-month": 8,
            "One year": 14,
            "Two year": 24,
        }
    )
    gross_margin_rate: float = 0.8
    high_risk_offer_cost: float = 90.0
    medium_risk_offer_cost: float = 20.0
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


def get_action_cost(risk_band: str, assumptions: RetentionAssumptions | None = None) -> float:
    assumptions = assumptions or RetentionAssumptions()
    if risk_band == "high":
        return assumptions.high_risk_offer_cost
    if risk_band == "medium":
        return assumptions.medium_risk_offer_cost
    return 0.0


def get_success_rate(risk_band: str, assumptions: RetentionAssumptions | None = None) -> float:
    assumptions = assumptions or RetentionAssumptions()
    if risk_band == "high":
        return assumptions.high_risk_success_rate
    if risk_band == "medium":
        return assumptions.medium_risk_success_rate
    return 0.0


def estimate_customer_value(
    customer_row: pd.Series,
    assumptions: RetentionAssumptions | None = None,
) -> float:
    assumptions = assumptions or RetentionAssumptions()
    remaining_months = assumptions.remaining_months_by_contract.get(
        customer_row["Contract"],
        assumptions.remaining_months_by_contract["Month-to-month"],
    )
    return float(customer_row["MonthlyCharges"] * remaining_months * assumptions.gross_margin_rate)


def estimate_expected_value(
    churn_probability: float,
    risk_band: str,
    customer_value: float,
    assumptions: RetentionAssumptions | None = None,
) -> float:
    assumptions = assumptions or RetentionAssumptions()
    success_rate = get_success_rate(risk_band, assumptions)
    action_cost = get_action_cost(risk_band, assumptions)
    expected_saved_value = churn_probability * customer_value * success_rate
    return expected_saved_value - action_cost


def explain_action_rule(risk_band: str) -> str:
    explanations = {
        "high": "High-risk customers receive the strongest intervention because the model estimates a substantial churn probability.",
        "medium": "Medium-risk customers receive a lower-cost outreach to balance retention upside and intervention spend.",
        "low": "Low-risk customers are monitored instead of targeted so the team does not overspend on likely retained accounts.",
    }
    return explanations[risk_band]


def build_retention_frame(
    customer_frame: pd.DataFrame,
    churn_probabilities: pd.Series,
    assumptions: RetentionAssumptions | None = None,
) -> pd.DataFrame:
    assumptions = assumptions or RetentionAssumptions()

    output = customer_frame.copy()
    output["churn_probability"] = churn_probabilities
    output["risk_band"] = output["churn_probability"].apply(assign_risk_band)
    output["estimated_customer_value"] = output.apply(
        lambda row: estimate_customer_value(row, assumptions),
        axis=1,
    )
    output["action_cost_usd"] = output["risk_band"].apply(
        lambda band: get_action_cost(band, assumptions)
    )
    output["success_rate"] = output["risk_band"].apply(
        lambda band: get_success_rate(band, assumptions)
    )
    output["recommended_action"] = output["risk_band"].apply(recommend_action)
    output["action_reason"] = output["risk_band"].apply(explain_action_rule)
    output["expected_saved_value_usd"] = (
        output["churn_probability"] * output["estimated_customer_value"] * output["success_rate"]
    )
    output["expected_value_usd"] = output["expected_saved_value_usd"] - output["action_cost_usd"]
    return output
