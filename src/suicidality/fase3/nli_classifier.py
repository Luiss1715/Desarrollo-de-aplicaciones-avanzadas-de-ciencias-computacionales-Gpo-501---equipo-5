from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .embeddings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_WORDS,
    prepare_texts,
    split_text_chunks,
)
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
DEFAULT_SUPERVISED_NLI_LABELS = (
    "suicidal ideation",
    "desire to die",
    "intent to self-harm",
    "hopelessness",
    "no suicidal ideation",
    "an ordinary daily experience",
)


class ZeroShotNLIClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        threshold: float = 0.5,
        device: int | None = None,
        hypothesis_templates: Sequence[str] | None = None,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.hypothesis_templates = list(hypothesis_templates or DEFAULT_HYPOTHESIS_TEMPLATES)
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
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
        features = self._predict_candidate_scores(
            texts,
            [SUICIDAL_LABEL, NON_SUICIDAL_LABEL],
            batch_size=batch_size,
            multi_label=False,
        )
        scores = features[:, 0]
        return (scores >= self.threshold).astype(int), scores

    def predict_features(
        self,
        texts: Sequence[str],
        candidate_labels: Sequence[str] = DEFAULT_SUPERVISED_NLI_LABELS,
        batch_size: int = 8,
    ) -> np.ndarray:
        return self._predict_candidate_scores(
            texts,
            candidate_labels,
            batch_size=batch_size,
            multi_label=True,
        )

    def _predict_candidate_scores(
        self,
        texts: Sequence[str],
        candidate_labels: Sequence[str],
        batch_size: int,
        multi_label: bool,
    ) -> np.ndarray:
        labels = list(candidate_labels)
        if not labels:
            raise ValueError("candidate_labels must contain at least one label")
        if not texts:
            return np.empty((0, len(labels)), dtype=float)

        classifier = self._load_classifier()
        chunks = []
        owners = []
        for owner, text in enumerate(texts):
            text_chunks = split_text_chunks(text, self.chunk_words, self.chunk_overlap)
            chunks.extend(text_chunks)
            owners.extend([owner] * len(text_chunks))
        chunk_scores = np.zeros((len(chunks), len(labels)), dtype=float)
        for template in self.hypothesis_templates:
            results = classifier(
                chunks,
                candidate_labels=labels,
                hypothesis_template=template,
                multi_label=multi_label,
                batch_size=batch_size,
                truncation=True,
            )
            if isinstance(results, dict):
                results = [results]
            for row, result in enumerate(results):
                for column, label in enumerate(labels):
                    chunk_scores[row, column] += float(
                        result["scores"][result["labels"].index(label)]
                    )
        chunk_scores /= max(1, len(self.hypothesis_templates))
        scores = np.zeros((len(texts), len(labels)), dtype=float)
        for owner, chunk_score in zip(owners, chunk_scores):
            scores[owner] = np.maximum(scores[owner], chunk_score)
        return scores


class SupervisedNLIClassifier:
    def __init__(
        self,
        feature_extractor: ZeroShotNLIClassifier | None = None,
        candidate_labels: Sequence[str] = DEFAULT_SUPERVISED_NLI_LABELS,
        seed: int = 42,
    ):
        self.feature_extractor = feature_extractor or ZeroShotNLIClassifier()
        self.candidate_labels = list(candidate_labels)
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )

    def extract_features(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        return self.feature_extractor.predict_features(
            texts,
            candidate_labels=self.candidate_labels,
            batch_size=batch_size,
        )

    def fit_features(
        self, features: np.ndarray, labels: Sequence[int]
    ) -> "SupervisedNLIClassifier":
        self.classifier.fit(features, labels)
        return self

    def fit(
        self, texts: Sequence[str], labels: Sequence[int], batch_size: int = 8
    ) -> "SupervisedNLIClassifier":
        return self.fit_features(self.extract_features(texts, batch_size), labels)

    def predict_from_features(
        self, features: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        classes = list(self.classifier.classes_)
        scores = self.classifier.predict_proba(features)[:, classes.index(1)]
        return (scores >= threshold).astype(int), scores

    def predict(
        self, texts: Sequence[str], threshold: float = 0.5, batch_size: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_from_features(self.extract_features(texts, batch_size), threshold)


def evaluate_nli(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_NLI_MODEL,
    threshold: float = 0.5,
    hypothesis_templates: Sequence[str] | None = None,
    predictions_path: str | Path = "reports/phase3_nli_zero_shot_predictions.csv",
    metrics_path: str | Path = "reports/phase3_nli_zero_shot_metrics.json",
    batch_size: int = 8,
    device: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    classifier = ZeroShotNLIClassifier(
        model_name, threshold, device=device, hypothesis_templates=hypothesis_templates
    )
    predicted, scores = classifier.predict(prepare_texts(frame), batch_size=batch_size)
    labels = labels_from_frame(frame)
    predictions = build_prediction_frame(
        frame, range(len(frame)), predicted, scores, "phase3_nli_zero_shot"
    )
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
    eval_parser.add_argument(
        "--predictions-out", default="reports/phase3_nli_zero_shot_predictions.csv"
    )
    eval_parser.add_argument(
        "--metrics-out", default="reports/phase3_nli_zero_shot_metrics.json"
    )
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
