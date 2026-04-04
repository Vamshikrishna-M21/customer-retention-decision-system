from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    model_name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "brier_score": self.brier_score,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> EvaluationResult:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return EvaluationResult(
        model_name=model_name,
        threshold=threshold,
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        brier_score=brier_score_loss(y_true, y_score),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def build_threshold_table(
    model_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        result = evaluate_predictions(model_name, y_true, y_score, threshold=threshold)
        rows.append(result.as_dict())
    return pd.DataFrame(rows)

