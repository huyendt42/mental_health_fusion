"""Computes lexical diversity using MTLD.

This extractor uses the lexicalrichness library to calculate Measure of
Textual Lexical Diversity (MTLD), a length-robust vocabulary diversity
measure. Citation: McCarthy & Jarvis (2010); Tanaka et al. (2022) for mental
health NLP usage. FTD construct: Emptiness, via reduced lexical variety.
"""

import logging

import numpy as np
from lexicalrichness import LexicalRichness

from scripts.features.base import FeatureExtractorBase

logger = logging.getLogger(__name__)


class MTLDExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["mtld"]
    DIM = 1

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        try:
            lex = LexicalRichness(text)
            if lex.words < 2:
                return np.zeros(self.DIM, dtype=np.float32)
            return self._validate([float(lex.mtld(threshold=0.72))])
        except Exception as exc:
            logger.debug("MTLD failed for text prefix %r: %s", str(text)[:50], exc)
            return np.zeros(self.DIM, dtype=np.float32)
