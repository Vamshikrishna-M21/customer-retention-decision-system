from src.business.rules import (
    RetentionAssumptions,
    assign_risk_band,
    estimate_customer_value,
    estimate_expected_value,
    recommend_action,
)


def test_risk_band_thresholds():
    assert assign_risk_band(0.2) == "low"
    assert assign_risk_band(0.45) == "medium"
    assert assign_risk_band(0.8) == "high"


def test_recommended_actions_are_defined():
    for band in ["low", "medium", "high"]:
        assert recommend_action(band)


def test_expected_value_increases_for_high_risk_case():
    assumptions = RetentionAssumptions(
        remaining_months_by_contract={"Month-to-month": 8, "One year": 12, "Two year": 24},
        gross_margin_rate=1.0,
        high_risk_offer_cost=100,
        medium_risk_offer_cost=20,
        high_risk_success_rate=0.4,
        medium_risk_success_rate=0.2,
    )

    high_value = estimate_expected_value(0.8, "high", 1200, assumptions)
    medium_value = estimate_expected_value(0.5, "medium", 800, assumptions)

    assert high_value > medium_value


def test_customer_value_uses_contract_horizon():
    assumptions = RetentionAssumptions(
        remaining_months_by_contract={"Month-to-month": 6, "One year": 12, "Two year": 24},
        gross_margin_rate=0.5,
    )
    customer = {"MonthlyCharges": 100.0, "Contract": "Two year"}

    assert estimate_customer_value(customer, assumptions) == 1200.0
