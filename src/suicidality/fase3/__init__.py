"""Fase 3: Embeddings de Transformer, MLP y clasificación NLI."""

from .embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .latent_nn import LatentMLP, LatentNNConfig, train_mlp, predict_scores, predict_labels
from .nli_classifier import DEFAULT_NLI_MODEL, ZeroShotNLIClassifier

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "TransformerEmbedder",
    "prepare_texts",
    "LatentMLP",
    "LatentNNConfig",
    "train_mlp",
    "predict_scores",
    "predict_labels",
    "DEFAULT_NLI_MODEL",
    "ZeroShotNLIClassifier",
]
