"""Computes lexicon-based lexical word rates.

This extractor loads project lexicons once in __init__ and computes
death/harm, absolutist, negation, modal, and hedge token rates. Citations:
Al-Mosaiwi & Johnstone (2018), Pennebaker (2011), Hossain et al. (2025).
FTD construct: Emptiness, via harm content, absolutist language, negation,
modality, and uncertainty markers.
"""

from pathlib import Path

import numpy as np

from scripts.config import (
    ABSOLUTIST_LEXICON_PATH,
    DEATH_HARM_LEXICON_PATH,
    HEDGE_LEXICON_PATH,
    MODAL_LEXICON_PATH,
    NEGATION_LEXICON_PATH,
)
from scripts.features.base import FeatureExtractorBase, WORD_RE


class WordRatesExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [
        "death_harm_rate",
        "absolutist_rate",
        "negation_rate",
        "modal_rate",
        "hedge_rate",
    ]
    DIM = 5

    def __init__(self):
        self.death_harm = self._load_lexicon(DEATH_HARM_LEXICON_PATH)
        self.absolutist = self._load_lexicon(ABSOLUTIST_LEXICON_PATH)
        self.negation = self._load_lexicon(NEGATION_LEXICON_PATH)
        self.modal = self._load_lexicon(MODAL_LEXICON_PATH)
        self.hedge = self._load_lexicon(HEDGE_LEXICON_PATH)

    def _load_lexicon(self, path: Path) -> frozenset[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing lexical resource: {path}")
        words = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not words:
            raise ValueError(f"Lexicon file is empty: {path}")
        return frozenset(words)

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        tokens = [m.group(0).lower() for m in WORD_RE.finditer(text)]
        if not tokens:
            return np.zeros(self.DIM, dtype=np.float32)
        total = len(tokens)
        return self._validate(
            [
                sum(t in self.death_harm for t in tokens) / total,
                sum(t in self.absolutist for t in tokens) / total,
                sum(t in self.negation for t in tokens) / total,
                sum(t in self.modal for t in tokens) / total,
                sum(t in self.hedge for t in tokens) / total,
            ]
        )
