from src.data.loader import create_data_splits, load_clean_data
from src.models.train import fit_and_score_models


def test_clean_data_has_expected_target_values():
    data = load_clean_data()
    assert set(data["Churn"].unique()) == {0, 1}
    assert data["TotalCharges"].isna().sum() == 0


def test_split_sizes_cover_full_dataset():
    data = load_clean_data()
    splits = create_data_splits(data)
    total_rows = len(splits.X_train) + len(splits.X_valid) + len(splits.X_test)
    assert total_rows == len(data)


def test_model_training_returns_all_candidates():
    metrics, trained_models, _ = fit_and_score_models()
    assert set(metrics["model_name"]) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert set(trained_models.keys()) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }

