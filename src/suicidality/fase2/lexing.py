from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class LexResult:
    tokens: List[str]
    flags: Dict[str, int]
    critical_matches: List[str]


class RiskLexer:
    def __init__(self, lexicon_path: Path):
        self.lexicon_path = lexicon_path
        self._patterns: dict[str, list[re.Pattern]] = {}
        self._load()

    def _load(self) -> None:
        data = json.loads(self.lexicon_path.read_text(encoding="utf-8"))
        for category, phrases in data.items():
            self._patterns[category] = [self._compile(p) for p in phrases]

    @staticmethod
    def _compile(phrase: str) -> re.Pattern:
        escaped = re.escape(phrase)
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z']+", text.lower())

    def scan(self, text: str) -> LexResult:
        tokens = self._tokenize(text)
        flags: Dict[str, int] = {k: 0 for k in self._patterns.keys()}
        critical_matches: List[str] = []
        for category, patterns in self._patterns.items():
            for pat in patterns:
                if pat.search(text):
                    flags[category] = 1
                    if category == "critical":
                        critical_matches.append(pat.pattern.strip("\\b"))
        return LexResult(tokens=tokens, flags=flags, critical_matches=critical_matches)
