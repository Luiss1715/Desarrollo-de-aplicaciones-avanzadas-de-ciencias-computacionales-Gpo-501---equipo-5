from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def combine_title_text(title: object, text: object) -> str:
    title_value = "" if pd.isna(title) else str(title).strip()
    text_value = "" if pd.isna(text) else str(text).strip()
    return " ".join(value for value in [title_value, text_value] if value)


def prepare_texts(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in ["title", "text"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return [
        combine_title_text(title, text)
        for title, text in zip(frame["title"], frame["text"])
    ]


class TransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required to generate embeddings"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
