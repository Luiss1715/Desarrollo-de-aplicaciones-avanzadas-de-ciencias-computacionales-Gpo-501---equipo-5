from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LATENT_CLASSIFIER_NAMES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
)


def build_latent_classifier(name: str, seed: int = 42) -> ClassifierMixin:
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(
        f"Unknown latent classifier '{name}'. Choose from: {', '.join(LATENT_CLASSIFIER_NAMES)}"
    )


def fit_latent_classifier(
    name: str,
    embeddings: np.ndarray,
    labels: Sequence[int],
    seed: int = 42,
) -> ClassifierMixin:
    classifier = build_latent_classifier(name, seed)
    classifier.fit(embeddings, labels)
    return classifier


def predict_positive_scores(classifier: ClassifierMixin, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(embeddings), dtype=float)
        classes = list(classifier.classes_)
        return probabilities[:, classes.index(1)]
    if hasattr(classifier, "decision_function"):
        return expit(np.asarray(classifier.decision_function(embeddings), dtype=float))
    raise TypeError("Classifier must implement predict_proba or decision_function")
