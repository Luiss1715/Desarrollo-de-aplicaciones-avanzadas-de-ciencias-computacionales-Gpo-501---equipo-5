from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PREDICTION_COLUMNS = [
    "user_id",
    "text_id",
    "title",
    "text",
    "true_label",
    "predicted_label",
    "score",
    "method",
]


def calculate_protocol_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> dict[str, int | float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 1 and predicted == 1)
    tn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 0 and predicted == 0)
    fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 1 and predicted == 0)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    metrics: dict[str, int | float] = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "AUC": (1 + tpr - fpr) / 2,
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
    return metrics


def find_best_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    num_thresholds: int = 101,
) -> float:
    best_threshold = 0.5
    best_auc = -1.0
    for threshold in np.linspace(0.0, 1.0, num_thresholds):
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        metrics = calculate_protocol_metrics(y_true, y_pred, y_score)
        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            best_threshold = threshold
    return best_threshold


def build_prediction_frame(
    source: pd.DataFrame,
    indices: Sequence[int],
    predicted: Sequence[int],
    scores: Sequence[float],
    method: str,
) -> pd.DataFrame:
    selected = source.iloc[list(indices)].reset_index(drop=True)
    frame = pd.DataFrame()
    for column in ["user_id", "text_id", "title", "text"]:
        frame[column] = selected[column] if column in selected.columns else ""
    frame["true_label"] = selected["is_suicide"].map(normalize_label)
    frame["predicted_label"] = ["yes" if value == 1 else "no" for value in predicted]
    frame["score"] = [float(value) for value in scores]
    frame["method"] = method
    return frame[PREDICTION_COLUMNS]


def normalize_label(value: object) -> str:
    return "yes" if str(value).strip().lower() == "yes" else "no"


def labels_from_frame(frame: pd.DataFrame) -> list[int]:
    return [1 if normalize_label(value) == "yes" else 0 for value in frame["is_suicide"]]


def save_metrics(metrics: dict[str, int | float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }
    path.write_text(json.dumps(rounded, indent=2), encoding="utf-8")
