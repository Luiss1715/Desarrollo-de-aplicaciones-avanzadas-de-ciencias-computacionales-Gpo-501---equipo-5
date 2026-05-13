from __future__ import annotations

import re
from typing import Iterable, List, Optional

try:
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover - optional dependency
    spacy = None
    Language = None

try:
    from langdetect import detect
except Exception:  # pragma: no cover - optional dependency
    detect = None


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\b")
_WS_RE = re.compile(r"\s+")


def _detect_lang(text: str) -> str:
    if detect is None:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"


def _load_spacy(lang: str) -> Optional["Language"]:
    if spacy is None:
        return None
    model = "en_core_web_sm" if lang.startswith("en") else "es_core_news_sm"
    try:
        return spacy.load(model, disable=["parser", "ner"])
    except Exception:
        return None


def _basic_clean(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_text(text: str, use_spacy: bool = False) -> str:
    cleaned = _basic_clean(text)
    if not use_spacy:
        return cleaned

    lang = _detect_lang(cleaned)
    nlp = _load_spacy(lang)
    if nlp is None:
        return cleaned

    doc = nlp(cleaned)
    lemmas = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(lemmas).strip()


def preprocess_corpus(texts: Iterable[str], use_spacy: bool = False) -> List[str]:
    return [preprocess_text(t, use_spacy=use_spacy) for t in texts]
