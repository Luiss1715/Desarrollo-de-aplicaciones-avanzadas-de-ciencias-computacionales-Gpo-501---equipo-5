from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[int]


def _merge_text(title: str, text: str) -> str:
    title = title or ""
    text = text or ""
    merged = f"{title} {text}".strip()
    return merged


def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    for col in ["title", "text", "is_suicide"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.fillna("")
    texts = [_merge_text(r["title"], r["text"]) for _, r in df.iterrows()]
    labels = [1 if str(v).strip().lower() == "yes" else 0 for v in df["is_suicide"]]
    return Dataset(texts=texts, labels=labels)
