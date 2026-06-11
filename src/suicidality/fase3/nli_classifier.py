from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .embeddings import prepare_texts
from .latent_pipeline import load_phase3_dataset
from ..protocol_metrics import (
    build_prediction_frame,
    calculate_protocol_metrics,
    labels_from_frame,
    save_metrics,
)


DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
SUICIDAL_LABEL = "suicidal ideation"
NON_SUICIDAL_LABEL = "no suicidal ideation"
DEFAULT_HYPOTHESIS_TEMPLATES = (
    "This text expresses {}.",
    "This text is {}.",
)


class ZeroShotNLIClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        threshold: float = 0.5,
        device: int | None = None,
        hypothesis_templates: Sequence[str] | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.hypothesis_templates = list(hypothesis_templates or DEFAULT_HYPOTHESIS_TEMPLATES)
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError("transformers is required for NLI classification") from exc
            kwargs = {"task": "zero-shot-classification", "model": self.model_name}
            if self.device is not None:
                kwargs["device"] = self.device
            self._classifier = pipeline(**kwargs)
        return self._classifier

    def predict(self, texts: Sequence[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        classifier = self._load_classifier()
        scores = np.zeros(len(texts), dtype=float)
        for template in self.hypothesis_templates:
            results = classifier(
                list(texts),
                candidate_labels=[SUICIDAL_LABEL, NON_SUICIDAL_LABEL],
                hypothesis_template=template,
                multi_label=False,
                batch_size=batch_size,
                truncation=True,
            )
            if isinstance(results, dict):
                results = [results]
            scores += np.asarray(
                [
                    float(result["scores"][result["labels"].index(SUICIDAL_LABEL)])
                    for result in results
                ]
            )
        scores /= max(1, len(self.hypothesis_templates))
        return (scores >= self.threshold).astype(int), scores


def evaluate_nli(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_NLI_MODEL,
    threshold: float = 0.5,
    hypothesis_templates: Sequence[str] | None = None,
    predictions_path: str | Path = "reports/phase3_nli_predictions.csv",
    metrics_path: str | Path = "reports/phase3_nli_metrics.json",
    batch_size: int = 8,
    device: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    classifier = ZeroShotNLIClassifier(
        model_name, threshold, device=device, hypothesis_templates=hypothesis_templates
    )
    predicted, scores = classifier.predict(prepare_texts(frame), batch_size=batch_size)
    labels = labels_from_frame(frame)
    predictions = build_prediction_frame(frame, range(len(frame)), predicted, scores, "phase3_nli")
    metrics = calculate_protocol_metrics(labels, predicted.tolist(), scores.tolist())
    output = Path(predictions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    save_metrics(metrics, metrics_path)
    return predictions, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot NLI suicidality classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("eval", help="Evaluate NLI on a CSV dataset")
    eval_parser.add_argument("--csv", required=True)
    eval_parser.add_argument("--model-name", default=DEFAULT_NLI_MODEL)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument(
        "--hypothesis-template",
        nargs="+",
        default=list(DEFAULT_HYPOTHESIS_TEMPLATES),
    )
    eval_parser.add_argument("--predictions-out", default="reports/phase3_nli_predictions.csv")
    eval_parser.add_argument("--metrics-out", default="reports/phase3_nli_metrics.json")
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--device", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_phase3_dataset(args.csv)
    evaluate_nli(
        frame,
        model_name=args.model_name,
        threshold=args.threshold,
        hypothesis_templates=args.hypothesis_template,
        predictions_path=args.predictions_out,
        metrics_path=args.metrics_out,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
