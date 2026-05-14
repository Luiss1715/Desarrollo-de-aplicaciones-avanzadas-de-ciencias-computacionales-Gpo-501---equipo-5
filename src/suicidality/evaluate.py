from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class EvalResult:
    tp: int
    tn: int
    fp: int
    fn: int
    tpr: float
    fpr: float
    auc: float


def evaluate_auc(y_true: List[int], y_score: List[float], threshold: float = 0.5) -> EvalResult:
    tp = sum(1 for yt, yp in zip(y_true, y_score) if yt == 1 and yp >= threshold)
    tn = sum(1 for yt, yp in zip(y_true, y_score) if yt == 0 and yp < threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_score) if yt == 0 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_score) if yt == 1 and yp < threshold)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Protocol formula: AUC = (1 + TPR - FPR) / 2
    auc = (1 + tpr - fpr) / 2

    return EvalResult(tp=tp, tn=tn, fp=fp, fn=fn, tpr=tpr, fpr=fpr, auc=auc)


def save_metrics(result: EvalResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tp": result.tp,
        "tn": result.tn,
        "fp": result.fp,
        "fn": result.fn,
        "tpr": round(result.tpr, 6),
        "fpr": round(result.fpr, 6),
        "auc": round(result.auc, 6),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_roc_curve(y_true, y_score, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
