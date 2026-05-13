from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class FeatureExtractor:
    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(min_df=self.min_df, ngram_range=self.ngram_range)

    def fit(self, texts: List[str], lex_flags: List[dict[str, int]]) -> None:
        self.vectorizer.fit(texts)
        self._fit_flag_keys(lex_flags)

    def _fit_flag_keys(self, lex_flags: List[dict[str, int]]) -> None:
        keys = set()
        for flags in lex_flags:
            keys.update(flags.keys())
        self.flag_keys = sorted(keys)

    def _flags_to_matrix(self, lex_flags: List[dict[str, int]]) -> sparse.csr_matrix:
        rows = []
        for flags in lex_flags:
            rows.append([flags.get(k, 0) for k in self.flag_keys])
        arr = np.asarray(rows, dtype=float)
        return sparse.csr_matrix(arr)

    def transform(self, texts: List[str], lex_flags: List[dict[str, int]]) -> sparse.csr_matrix:
        text_matrix = self.vectorizer.transform(texts)
        flag_matrix = self._flags_to_matrix(lex_flags)
        return sparse.hstack([text_matrix, flag_matrix])

    def fit_transform(self, texts: List[str], lex_flags: List[dict[str, int]]) -> sparse.csr_matrix:
        self.fit(texts, lex_flags)
        return self.transform(texts, lex_flags)
