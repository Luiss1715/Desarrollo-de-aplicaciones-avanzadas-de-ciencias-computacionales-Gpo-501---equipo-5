"""Fase 2: TF-IDF, banderas léxicas y regresión logística."""

from .ingest import load_dataset
from .lexing import RiskLexer
from .preprocess import preprocess_corpus, preprocess_text
from .features import FeatureExtractor
from .model import SuicideRiskModel
from .pipeline import SuicidalityPipeline

__all__ = [
    "load_dataset",
    "RiskLexer",
    "preprocess_corpus",
    "preprocess_text",
    "FeatureExtractor",
    "SuicideRiskModel",
    "SuicidalityPipeline",
]
