from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

MPL_CONFIG_DIR = Path(".mplconfig")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message=".*oneOf.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*")

import joblib
import matplotlib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.business.rules import RetentionAssumptions, build_retention_frame
from src.data.loader import create_data_splits, load_clean_data
from src.features.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_linear_preprocessor,
    build_tree_preprocessor,
)
from src.models.evaluate import build_threshold_table, evaluate_predictions


RANDOM_STATE = 42
ARTIFACT_DIR = Path("models/artifacts")
PROCESSED_DIR = Path("data/processed")
FIGURE_DIR = Path("reports/figures")

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_model_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_linear_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_tree_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        max_depth=10,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_tree_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_depth=5,
                        max_iter=300,
                        min_samples_leaf=25,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


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


def ensure_directories() -> None:
    for path in (ARTIFACT_DIR, PROCESSED_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fit_and_score_models() -> tuple[pd.DataFrame, dict[str, Pipeline], object]:
    data = load_clean_data()
    splits = create_data_splits(data)

    candidates = get_model_candidates()
    metrics_rows: list[dict[str, float | int | str]] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in candidates.items():
        fitted_pipeline = clone(pipeline)
        fitted_pipeline.fit(splits.X_train, splits.y_train)

        valid_scores = fitted_pipeline.predict_proba(splits.X_valid)[:, 1]
        result = evaluate_predictions(model_name, splits.y_valid, valid_scores, threshold=0.5)
        metrics_rows.append(result.as_dict())
        trained_models[model_name] = fitted_pipeline

    metrics = pd.DataFrame(metrics_rows).sort_values(
        by=["pr_auc", "recall", "roc_auc"], ascending=False
    )
    return metrics, trained_models, splits


def select_threshold(model_name: str, trained_model: Pipeline, splits: object) -> tuple[float, pd.DataFrame]:
    valid_scores = trained_model.predict_proba(splits.X_valid)[:, 1]
    threshold_table = build_threshold_table(
        model_name,
        splits.y_valid,
        valid_scores,
        thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    )

    eligible = threshold_table[threshold_table["recall"] >= 0.70]
    if not eligible.empty:
        chosen_row = eligible.sort_values(by=["precision", "f1"], ascending=False).iloc[0]
    else:
        chosen_row = threshold_table.sort_values(by=["f1", "recall"], ascending=False).iloc[0]

    return float(chosen_row["threshold"]), threshold_table


def build_threshold_economics(
    features: pd.DataFrame,
    y_true: pd.Series,
    y_score: pd.Series,
    threshold_table: pd.DataFrame,
    assumptions: RetentionAssumptions | None = None,
) -> pd.DataFrame:
    assumptions = assumptions or RetentionAssumptions()
    economics_rows: list[dict[str, float | int]] = []

    for threshold in threshold_table["threshold"]:
        predicted_positive = y_score >= threshold
        retention_frame = build_retention_frame(features, y_score, assumptions)
        targeted = retention_frame.loc[predicted_positive].copy()

        economics_rows.append(
            {
                "threshold": float(threshold),
                "targeted_customers": int(predicted_positive.sum()),
                "target_rate": float(predicted_positive.mean()),
                "actual_churners_targeted": int(y_true.loc[predicted_positive].sum()),
                "expected_saved_value_usd": float(targeted["expected_saved_value_usd"].sum()),
                "total_action_cost_usd": float(targeted["action_cost_usd"].sum()),
                "net_expected_value_usd": float(targeted["expected_value_usd"].sum()),
            }
        )

    return pd.DataFrame(economics_rows)


def save_feature_importance(model_name: str, trained_model: Pipeline, splits: object) -> pd.DataFrame:
    if model_name == "logistic_regression":
        feature_names = trained_model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = trained_model.named_steps["model"].coef_[0]
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coefficients,
                "abs_importance": abs(coefficients),
            }
        ).sort_values("abs_importance", ascending=False)
    else:
        result = permutation_importance(
            trained_model,
            splits.X_valid,
            splits.y_valid,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="average_precision",
        )
        importance = pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "importance": result.importances_mean,
                "abs_importance": result.importances_mean.abs(),
            }
        ).sort_values("abs_importance", ascending=False)

    importance["display_feature"] = importance["feature"].apply(format_feature_name)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    return importance


def create_figures(
    metrics: pd.DataFrame,
    threshold_table: pd.DataFrame,
    threshold_economics: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        metrics["model_name"],
        metrics["pr_auc"],
        color=["#4c78a8", "#72b7b2", "#54a24b"],
    )
    ax.set_title("Validation PR-AUC by model")
    ax.set_xlabel("")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison_pr_auc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(threshold_table["threshold"], threshold_table["recall"], marker="o", label="Recall")
    ax.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        marker="o",
        label="Precision",
    )
    ax.set_title("Threshold trade-off on validation split")
    ax.set_ylabel("Score")
    ax.set_xlabel("Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_tradeoff.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        threshold_economics["threshold"],
        threshold_economics["net_expected_value_usd"],
        marker="o",
        color="#e15759",
    )
    ax.set_title("Net expected value by threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net expected value (USD)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_economics.png", dpi=200)
    plt.close(fig)

    top_features = importance.head(12).sort_values("abs_importance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_features["display_feature"], top_features["abs_importance"], color="#59a14f")
    ax.set_title("Top feature signals")
    ax.set_xlabel("Absolute importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_feature_importance.png", dpi=200)
    plt.close(fig)


def train_and_save_artifacts() -> dict[str, float | str]:
    ensure_directories()

    metrics, trained_models, splits = fit_and_score_models()
    selected_model_name = metrics.iloc[0]["model_name"]
    selected_model = trained_models[selected_model_name]
    assumptions = RetentionAssumptions()

    selected_threshold, threshold_table = select_threshold(
        selected_model_name,
        selected_model,
        splits,
    )
    valid_scores = pd.Series(selected_model.predict_proba(splits.X_valid)[:, 1])
    threshold_economics = build_threshold_economics(
        splits.X_valid,
        splits.y_valid,
        valid_scores,
        threshold_table,
        assumptions,
    )
    best_business_row = threshold_economics.sort_values(
        by=["net_expected_value_usd", "actual_churners_targeted"],
        ascending=False,
    ).iloc[0]

    combined_features = pd.concat([splits.X_train, splits.X_valid], axis=0).reset_index(drop=True)
    combined_target = pd.concat([splits.y_train, splits.y_valid], axis=0).reset_index(drop=True)
    final_model = clone(get_model_candidates()[selected_model_name])
    final_model.fit(combined_features, combined_target)

    test_scores = final_model.predict_proba(splits.X_test)[:, 1]
    test_result = evaluate_predictions(
        selected_model_name,
        splits.y_test,
        test_scores,
        threshold=selected_threshold,
    )
    test_metrics = pd.DataFrame([test_result.as_dict()])

    retention_output = build_retention_frame(splits.X_test, pd.Series(test_scores), assumptions)
    retention_output.insert(0, "actual_churn", splits.y_test)

    importance = save_feature_importance(selected_model_name, final_model, splits)
    create_figures(metrics, threshold_table, threshold_economics, importance)

    metrics.to_csv(ARTIFACT_DIR / "validation_model_metrics.csv", index=False)
    threshold_table.to_csv(ARTIFACT_DIR / "threshold_analysis.csv", index=False)
    threshold_economics.to_csv(ARTIFACT_DIR / "threshold_economics.csv", index=False)
    test_metrics.to_csv(ARTIFACT_DIR / "test_metrics.csv", index=False)
    retention_output.to_csv(PROCESSED_DIR / "retention_predictions.csv", index=False)

    joblib.dump(
        {"model": final_model, "threshold": selected_threshold, "model_name": selected_model_name},
        ARTIFACT_DIR / "model_bundle.joblib",
    )

    summary = {
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "best_business_threshold": round(float(best_business_row["threshold"]), 2),
        "test_roc_auc": round(float(test_result.roc_auc), 4),
        "test_pr_auc": round(float(test_result.pr_auc), 4),
        "test_precision": round(float(test_result.precision), 4),
        "test_recall": round(float(test_result.recall), 4),
        "test_f1": round(float(test_result.f1), 4),
        "best_threshold_net_value_usd": round(float(best_business_row["net_expected_value_usd"]), 2),
    }

    with open(ARTIFACT_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


if __name__ == "__main__":
    output = train_and_save_artifacts()
    print(json.dumps(output, indent=2))
