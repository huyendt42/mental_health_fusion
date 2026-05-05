"""Computes closed-list pronoun rates.

This extractor uses fixed English pronoun sets to compute first-person
singular, first-person plural, and second-person token rates. Citations:
Pennebaker (2011), Edwards et al. meta-analytic work on self-focus markers.
FTD construct: Emptiness, via self/other reference patterns.
"""

import numpy as np

from scripts.features.base import FeatureExtractorBase, WORD_RE


class PronounExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [
        "first_person_singular_rate",
        "first_person_plural_rate",
        "second_person_rate",
    ]
    DIM = 3

    def __init__(self):
        self.first_singular = frozenset({"i", "me", "my", "mine", "myself"})
        self.first_plural = frozenset({"we", "us", "our", "ours", "ourselves"})
        self.second_person = frozenset({"you", "your", "yours", "yourself"})

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        tokens = [m.group(0).lower() for m in WORD_RE.finditer(text)]
        if not tokens:
            return np.zeros(self.DIM, dtype=np.float32)
        total = len(tokens)
        return self._validate(
            [
                sum(t in self.first_singular for t in tokens) / total,
                sum(t in self.first_plural for t in tokens) / total,
                sum(t in self.second_person for t in tokens) / total,
            ]
        )
