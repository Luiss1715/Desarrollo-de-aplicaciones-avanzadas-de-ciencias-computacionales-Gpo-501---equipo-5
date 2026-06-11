import json

import pytest

from suicidality.protocol_metrics import calculate_protocol_metrics, save_metrics


def test_protocol_metrics_counts_and_auc():
    metrics = calculate_protocol_metrics(
        y_true=[1, 1, 0, 0],
        y_pred=[1, 0, 1, 0],
        y_score=[0.9, 0.4, 0.8, 0.1],
    )

    assert metrics["TP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TPR"] == 0.5
    assert metrics["FPR"] == 0.5
    assert metrics["AUC"] == 0.5
    assert metrics["ROC-AUC"] == pytest.approx(0.75)


def test_save_metrics_rounds_values(tmp_path):
    output = tmp_path / "metrics.json"
    save_metrics({"TP": 1, "TPR": 1 / 3, "ROC-AUC": 0.75}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "TP": 1,
        "TPR": 0.333333,
        "ROC-AUC": 0.75,
    }
