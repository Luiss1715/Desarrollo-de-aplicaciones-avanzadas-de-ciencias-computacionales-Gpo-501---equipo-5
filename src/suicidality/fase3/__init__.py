"""Fase 3: embeddings de Transformer, clasificadores latentes y NLI."""

from .embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .latent_classifiers import (
    LATENT_CLASSIFIER_NAMES,
    build_latent_classifier,
    fit_latent_classifier,
    predict_positive_scores,
)
from .latent_nn import LatentMLP, LatentNNConfig, predict_labels, predict_scores, train_mlp
from .nli_classifier import DEFAULT_NLI_MODEL, SupervisedNLIClassifier, ZeroShotNLIClassifier

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "TransformerEmbedder",
    "prepare_texts",
    "LATENT_CLASSIFIER_NAMES",
    "build_latent_classifier",
    "fit_latent_classifier",
    "predict_positive_scores",
    "LatentMLP",
    "LatentNNConfig",
    "train_mlp",
    "predict_scores",
    "predict_labels",
    "DEFAULT_NLI_MODEL",
    "ZeroShotNLIClassifier",
    "SupervisedNLIClassifier",
]
