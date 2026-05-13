from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class EvalResult:
    auc: float


def evaluate_auc(y_true, y_score) -> EvalResult:
    auc = float(roc_auc_score(y_true, y_score))
    return EvalResult(auc=auc)


def save_metrics(result: EvalResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"roc_auc": result.auc}
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
