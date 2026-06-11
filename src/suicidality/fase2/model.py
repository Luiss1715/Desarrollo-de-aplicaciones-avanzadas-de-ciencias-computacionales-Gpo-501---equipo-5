from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression


@dataclass
class SuicideRiskModel:
    max_iter: int = 1000

    def __post_init__(self) -> None:
        self.model = LogisticRegression(max_iter=self.max_iter, class_weight="balanced")

    def fit(self, x, y) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def predict(self, x):
        return self.model.predict(x)
