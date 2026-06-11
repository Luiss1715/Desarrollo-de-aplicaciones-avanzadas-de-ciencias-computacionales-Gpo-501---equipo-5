from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

import numpy as np


DEFAULT_PROMPT_MODEL = "google/flan-t5-small"


def build_prompt(text: str) -> str:
    return (
        "Classify whether the following text expresses suicidal ideation. "
        'Return only JSON with this format: {"label": "yes" or "no", "score": 0.0 to 1.0}.\n'
        f"Text: {text}"
    )


def parse_prompt_response(response: str) -> tuple[int, float]:
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response must contain a JSON object")
    payload = json.loads(match.group(0))
    label = str(payload.get("label", "")).strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError("LLM response label must be 'yes' or 'no'")
    score = float(payload.get("score"))
    if not 0.0 <= score <= 1.0:
        raise ValueError("LLM response score must be between 0 and 1")
    return (1 if label == "yes" else 0), score


class PromptLLMClassifier:
    def __init__(self, generator: Callable[[str], str]):
        self.generator = generator

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        parsed = [parse_prompt_response(self.generator(build_prompt(text))) for text in texts]
        labels, scores = zip(*parsed) if parsed else ([], [])
        return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


class HuggingFacePromptLLMClassifier(PromptLLMClassifier):
    def __init__(self, model_name: str = DEFAULT_PROMPT_MODEL, device: int | None = None):
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        super().__init__(self._generate)

    def _generate(self, prompt: str) -> str:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for the optional prompt classifier"
                ) from exc
            kwargs = {"task": "text2text-generation", "model": self.model_name}
            if self.device is not None:
                kwargs["device"] = self.device
            self._pipeline = pipeline(**kwargs)
        result = self._pipeline(prompt, max_new_tokens=40, do_sample=False)
        return str(result[0]["generated_text"])
