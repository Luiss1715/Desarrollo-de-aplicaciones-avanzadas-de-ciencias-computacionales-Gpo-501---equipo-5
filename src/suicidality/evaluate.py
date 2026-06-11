from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class EvalResult:
    tp: int
    tn: int
    fp: int
    fn: int
    tpr: float
    fpr: float
    auc: float
    roc_auc: float | None


def evaluate_auc(y_true: List[int], y_score: List[float], threshold: float = 0.5) -> EvalResult:
    tp = sum(1 for yt, yp in zip(y_true, y_score) if yt == 1 and yp >= threshold)
    tn = sum(1 for yt, yp in zip(y_true, y_score) if yt == 0 and yp < threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_score) if yt == 0 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_score) if yt == 1 and yp < threshold)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Protocol formula: AUC = (1 + TPR - FPR) / 2
    auc = (1 + tpr - fpr) / 2

    roc_auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None
    return EvalResult(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        auc=auc,
        roc_auc=roc_auc,
    )


def save_metrics(result: EvalResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "TP": result.tp,
        "TN": result.tn,
        "FP": result.fp,
        "FN": result.fn,
        "TPR": round(result.tpr, 6),
        "FPR": round(result.fpr, 6),
        "AUC": round(result.auc, 6),
    }
    if result.roc_auc is not None:
        payload["ROC-AUC"] = round(result.roc_auc, 6)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_roc_curve(y_true, y_score, output_path: Path) -> None:
    import matplotlib.pyplot as plt

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
