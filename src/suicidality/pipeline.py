from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import TrainConfig
from .features import FeatureExtractor
from .lexing import LexResult, RiskLexer
from .model import SuicideRiskModel
from .preprocess import preprocess_corpus, preprocess_text


@dataclass
class SuicidalityPipeline:
    config: TrainConfig
    lexer: RiskLexer
    features: FeatureExtractor
    model: SuicideRiskModel

    def _lex_batch(self, texts: List[str]) -> List[LexResult]:
        return [self.lexer.scan(t) for t in texts]

    def fit(self, texts: List[str], labels: List[int]) -> None:
        lex_results = self._lex_batch(texts)
        preprocessed = preprocess_corpus(texts, use_spacy=self.config.use_spacy)
        flags = [r.flags for r in lex_results]
        x = self.features.fit_transform(preprocessed, flags)
        self.model.fit(x, labels)

    def predict_proba(self, texts: List[str]) -> List[float]:
        lex_results = self._lex_batch(texts)
        preprocessed = preprocess_corpus(texts, use_spacy=self.config.use_spacy)
        flags = [r.flags for r in lex_results]
        x = self.features.transform(preprocessed, flags)
        return self.model.predict_proba(x).tolist()

    def predict(self, texts: List[str]) -> List[int]:
        proba = self.predict_proba(texts)
        return [1 if p >= 0.5 else 0 for p in proba]

    def predict_text(self, text: str) -> tuple[int, float, List[str]]:
        lex = self.lexer.scan(text)
        cleaned = preprocess_text(text, use_spacy=self.config.use_spacy)
        x = self.features.transform([cleaned], [lex.flags])
        score = float(self.model.predict_proba(x)[0])
        label = 1 if score >= 0.5 else 0
        return label, score, lex.critical_matches
