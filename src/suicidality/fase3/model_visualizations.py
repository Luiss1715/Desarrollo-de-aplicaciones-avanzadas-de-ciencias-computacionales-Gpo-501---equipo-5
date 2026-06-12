from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from ..protocol_metrics import calculate_protocol_metrics


METHOD_DISPLAY_NAMES = {
    "phase2": "Fase 2",
    "phase3_latent_mlp": "MLP latente",
    "phase3_latent_logreg": "Logistica latente",
    "phase3_latent_svm": "SVM latente",
    "phase3_latent_random_forest": "Random Forest latente",
    "phase3_nli_zero_shot": "NLI zero-shot",
    "phase3_nli_supervised": "NLI supervisado",
    "phase3_ensemble": "Ensamble",
}


def load_prediction_frames(
    reports_dir: str | Path,
    comparison_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    reports_path = Path(reports_dir)
    comparison = pd.read_csv(comparison_path or reports_path / "comparison_table.csv")
    predictions = {}
    for method in comparison["method"]:
        path = reports_path / f"{method}_predictions.csv"
        if path.exists():
            predictions[str(method)] = pd.read_csv(path)
    if not predictions:
        raise ValueError("No prediction CSV files were found for the comparison methods")
    return comparison, predictions


def generate_model_comparison_visualizations(
    predictions: Mapping[str, pd.DataFrame],
    comparison: pd.DataFrame,
    output_dir: str | Path = "reports",
    thresholds: Mapping[str, float] | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ordered_predictions = {
        str(method): predictions[str(method)]
        for method in comparison["method"]
        if str(method) in predictions
    }
    _validate_prediction_frames(ordered_predictions)

    generated = [
        _plot_metric_comparison(comparison, output_path / "model_metrics_comparison.png"),
        _plot_confusion_matrices(comparison, output_path / "confusion_matrices.png"),
        _plot_roc_curves(ordered_predictions, output_path / "roc_curves_comparison.png"),
        _plot_precision_recall_curves(
            ordered_predictions, output_path / "precision_recall_curves.png"
        ),
        _plot_error_tradeoff(comparison, output_path / "fp_fn_tradeoff.png"),
        _plot_threshold_sensitivity(
            ordered_predictions,
            thresholds or {},
            output_path / "threshold_sensitivity.png",
        ),
    ]
    return generated


def generate_visualizations_from_reports(
    reports_dir: str | Path = "reports",
    thresholds_path: str | Path | None = "models/phase3_thresholds.json",
) -> list[Path]:
    comparison, predictions = load_prediction_frames(reports_dir)
    thresholds = {}
    if thresholds_path and Path(thresholds_path).exists():
        thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    return generate_model_comparison_visualizations(
        predictions,
        comparison,
        output_dir=reports_dir,
        thresholds=thresholds,
    )


def _validate_prediction_frames(predictions: Mapping[str, pd.DataFrame]) -> None:
    required = {"true_label", "predicted_label", "score"}
    expected_labels = None
    for method, frame in predictions.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{method} predictions are missing columns: {', '.join(sorted(missing))}")
        labels = _binary_labels(frame["true_label"])
        if expected_labels is None:
            expected_labels = labels
        elif not np.array_equal(labels, expected_labels):
            raise ValueError("All prediction files must contain the same true labels in the same order")


def _binary_labels(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.lower().eq("yes").astype(int).to_numpy()


def _display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("phase3_", "").replace("_", " "))


def _save_figure(figure, path: Path, use_tight_layout: bool = True) -> Path:
    if use_tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def _plot_metric_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    metrics = ["TPR", "FPR", "AUC", "ROC-AUC"]
    names = [_display_name(str(method)) for method in comparison["method"]]
    positions = np.arange(len(names))
    width = 0.19
    figure, axis = plt.subplots(figsize=(14, 7))
    for index, metric in enumerate(metrics):
        axis.bar(
            positions + (index - 1.5) * width,
            comparison[metric].astype(float),
            width,
            label=metric,
        )
    axis.set_title("Comparacion de metricas por clasificador")
    axis.set_ylabel("Proporcion")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=28, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center")
    return _save_figure(figure, path)


def _plot_confusion_matrices(comparison: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    count = len(comparison)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.7 * rows), squeeze=False)
    for axis, (_, row) in zip(axes.flat, comparison.iterrows()):
        matrix = np.asarray([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        axis.imshow(matrix, cmap="Blues")
        for y in range(2):
            for x in range(2):
                axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=13)
        axis.set_title(_display_name(str(row["method"])))
        axis.set_xticks([0, 1], labels=["Pred. no", "Pred. yes"])
        axis.set_yticks([0, 1], labels=["Real no", "Real yes"])
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.suptitle("Matrices de confusion", fontsize=16)
    figure.subplots_adjust(top=0.9, hspace=0.35, wspace=0.35)
    return _save_figure(figure, path, use_tight_layout=False)


def _plot_roc_curves(predictions: Mapping[str, pd.DataFrame], path: Path) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8))
    for method, frame in predictions.items():
        labels = _binary_labels(frame["true_label"])
        scores = frame["score"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = float(roc_auc_score(labels, scores))
        axis.plot(fpr, tpr, linewidth=2, label=f"{_display_name(method)} ({roc_auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    axis.set_title("Curvas ROC de todos los clasificadores")
    axis.set_xlabel("Tasa de falsos positivos")
    axis.set_ylabel("Tasa de verdaderos positivos")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    return _save_figure(figure, path)


def _plot_precision_recall_curves(
    predictions: Mapping[str, pd.DataFrame], path: Path
) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8))
    baseline = None
    for method, frame in predictions.items():
        labels = _binary_labels(frame["true_label"])
        scores = frame["score"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        baseline = float(labels.mean())
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{_display_name(method)} (AP={average_precision:.3f})",
        )
    if baseline is not None:
        axis.axhline(baseline, linestyle="--", color="gray", label=f"Base ({baseline:.3f})")
    axis.set_title("Curvas Precision-Recall")
    axis.set_xlabel("Recall / sensibilidad")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", fontsize=8)
    return _save_figure(figure, path)


def _plot_error_tradeoff(comparison: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(comparison["FP"], comparison["FN"], s=100, c=comparison["AUC"], cmap="viridis")
    for _, row in comparison.iterrows():
        axis.annotate(
            _display_name(str(row["method"])),
            (float(row["FP"]), float(row["FN"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Balance entre falsos positivos y falsos negativos")
    axis.set_xlabel("Falsos positivos (menor es mejor)")
    axis.set_ylabel("Falsos negativos (menor es mejor)")
    axis.grid(alpha=0.25)
    return _save_figure(figure, path)


def _plot_threshold_sensitivity(
    predictions: Mapping[str, pd.DataFrame],
    thresholds: Mapping[str, float],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    threshold_values = np.linspace(0.0, 1.0, 101)
    figure, axis = plt.subplots(figsize=(11, 8))
    for method, frame in predictions.items():
        labels = _binary_labels(frame["true_label"]).tolist()
        scores = frame["score"].astype(float).to_numpy()
        protocol_auc = [
            calculate_protocol_metrics(
                labels,
                (scores >= threshold).astype(int).tolist(),
            )["AUC"]
            for threshold in threshold_values
        ]
        line = axis.plot(
            threshold_values,
            protocol_auc,
            linewidth=2,
            label=_display_name(method),
        )[0]
        selected = thresholds.get(method)
        if selected is not None:
            selected_auc = calculate_protocol_metrics(
                labels,
                (scores >= float(selected)).astype(int).tolist(),
            )["AUC"]
            axis.scatter([selected], [selected_auc], color=line.get_color(), s=35)
    axis.set_title("Sensibilidad de la AUC del protocolo al umbral")
    axis.set_xlabel("Umbral de clasificacion")
    axis.set_ylabel("AUC del protocolo")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return _save_figure(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model comparison visualizations")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--thresholds", default="models/phase3_thresholds.json")
    args = parser.parse_args()
    generated = generate_visualizations_from_reports(args.reports_dir, args.thresholds)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
