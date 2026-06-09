import pytest

from suicidality.protocol_metrics import calculate_protocol_metrics


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
    assert metrics["roc_auc"] == pytest.approx(0.75)
