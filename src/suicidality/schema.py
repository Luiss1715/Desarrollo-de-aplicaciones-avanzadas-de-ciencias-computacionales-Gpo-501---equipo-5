from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class InputRecord:
    text: str
    source_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class OutputRecord:
    risk_label: str
    risk_score: float
    alert_tokens: List[str]
    model_version: str
