from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
LEXICON_PATH = PACKAGE_DIR / "resources" / "risk_lexicon.json"


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_seed: int = 42
    use_spacy: bool = False
    max_text_length: int = 5000
    tfidf_min_df: int = 2
    tfidf_ngram_range: tuple[int, int] = (1, 2)
