from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    return data


def clean_telco_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
    return cleaned


def load_clean_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return clean_telco_data(load_raw_data(path))


def get_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=[TARGET_COLUMN, ID_COLUMN])


def create_data_splits(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    features = get_feature_frame(data)
    target = data[TARGET_COLUMN]

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    adjusted_valid_size = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=adjusted_valid_size,
        stratify=y_train_valid,
        random_state=random_state,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )

