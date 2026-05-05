"""Computes spaCy POS-tag ratio features.

This extractor uses spaCy POS tags to compute adjective, adverb, and overall
pronoun ratios. Citations: Coh-Metrix tradition (McNamara et al., 2014),
Pennebaker (2011), Tanaka et al. (2022). FTD construct: Disorganization,
via grammatical category usage patterns.
"""

import logging

import numpy as np
import spacy

from scripts.features.base import FeatureExtractorBase

logger = logging.getLogger(__name__)


class POSRatioExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["adjective_ratio", "adverb_ratio", "pronoun_ratio"]
    DIM = 3

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy English model not found. Run: python -m spacy download en_core_web_sm"
            ) from exc

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.debug("spaCy parse failed for text prefix %r: %s", str(text)[:50], exc)
            return np.zeros(self.DIM, dtype=np.float32)

        tokens = [token for token in doc if not token.is_space and not token.is_punct]
        if not tokens:
            return np.zeros(self.DIM, dtype=np.float32)
        total = len(tokens)
        return self._validate(
            [
                sum(t.pos_ == "ADJ" or t.dep_ == "amod" for t in tokens) / total,
                sum(t.pos_ == "ADV" for t in tokens) / total,
                sum(t.pos_ == "PRON" for t in tokens) / total,
            ]
        )
