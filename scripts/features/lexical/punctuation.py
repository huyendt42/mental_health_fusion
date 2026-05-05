"""Computes sentence-normalized punctuation markers.

This extractor uses regex sentence splitting and raw character counts to
compute question mark and ellipsis rates. Citations: Lagutina et al. (2019)
stylometric punctuation features; Hossain et al. (2025) for uncertainty
markers. FTD construct: Emptiness, via incomplete or uncertain expression.
"""

import re

import numpy as np

from scripts.features.base import FeatureExtractorBase


class PunctuationExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["question_mark_rate", "ellipsis_rate"]
    DIM = 2

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        sentence_count = max(1, len(sentences))
        ellipsis_count = text.count("...") + text.count("…")
        return self._validate(
            [text.count("?") / sentence_count, ellipsis_count / sentence_count]
        )
