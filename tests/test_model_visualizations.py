import pandas as pd

from suicidality.fase3.model_visualizations import (
    generate_model_comparison_visualizations,
    load_prediction_frames,
)


def _prediction_frame(method, predicted, scores):
    return pd.DataFrame(
        {
            "true_label": ["yes", "yes", "no", "no"],
            "predicted_label": ["yes" if value else "no" for value in predicted],
            "score": scores,
            "method": method,
        }
    )


def test_generate_model_comparison_visualizations(tmp_path):
    predictions = {
        "phase2": _prediction_frame("phase2", [1, 0, 1, 0], [0.9, 0.4, 0.8, 0.1]),
        "phase3_ensemble": _prediction_frame(
            "phase3_ensemble", [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]
        ),
    }
    comparison = pd.DataFrame(
        [
            {
                "method": "phase2",
                "TP": 1,
                "TN": 1,
                "FP": 1,
                "FN": 1,
                "TPR": 0.5,
                "FPR": 0.5,
                "AUC": 0.5,
                "ROC-AUC": 0.75,
            },
            {
                "method": "phase3_ensemble",
                "TP": 2,
                "TN": 2,
                "FP": 0,
                "FN": 0,
                "TPR": 1.0,
                "FPR": 0.0,
                "AUC": 1.0,
                "ROC-AUC": 1.0,
            },
        ]
    )

    generated = generate_model_comparison_visualizations(
        predictions,
        comparison,
        output_dir=tmp_path,
        thresholds={"phase2": 0.5, "phase3_ensemble": 0.5},
    )

    assert len(generated) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in generated)


def test_load_prediction_frames_uses_comparison_methods(tmp_path):
    comparison = pd.DataFrame({"method": ["phase2", "phase3_ensemble"]})
    comparison.to_csv(tmp_path / "comparison_table.csv", index=False)
    _prediction_frame("phase2", [1, 0, 1, 0], [0.9, 0.4, 0.8, 0.1]).to_csv(
        tmp_path / "phase2_predictions.csv", index=False
    )

    loaded_comparison, predictions = load_prediction_frames(tmp_path)

    assert loaded_comparison["method"].tolist() == ["phase2", "phase3_ensemble"]
    assert list(predictions) == ["phase2"]
