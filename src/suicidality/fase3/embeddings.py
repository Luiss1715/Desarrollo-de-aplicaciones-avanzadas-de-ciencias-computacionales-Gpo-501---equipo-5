from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CHUNK_WORDS = 220
DEFAULT_CHUNK_OVERLAP = 40


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


def split_text_chunks(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")
    words = str(text).split()
    if not words:
        return [""]
    step = chunk_words - overlap_words
    return [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
        if words[start : start + chunk_words]
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

    def encode_chunked(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        overlap_words: int = DEFAULT_CHUNK_OVERLAP,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        chunks = []
        owners = []
        for owner, text in enumerate(texts):
            text_chunks = split_text_chunks(text, chunk_words, overlap_words)
            chunks.extend(text_chunks)
            owners.extend([owner] * len(text_chunks))
        chunk_embeddings = self.encode(chunks, batch_size, show_progress_bar)
        pooled = np.zeros((len(texts), chunk_embeddings.shape[1]), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for owner, embedding in zip(owners, chunk_embeddings):
            pooled[owner] += embedding
            counts[owner] += 1
        return pooled / np.maximum(counts[:, None], 1.0)


def save_embeddings(embeddings: np.ndarray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
