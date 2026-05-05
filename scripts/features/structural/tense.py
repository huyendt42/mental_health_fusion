"""Computes spaCy morphology tense distribution features.

This extractor uses spaCy POS/morphology and simple future auxiliary rules
to compute past, present, and future verb ratios. Citation: Coh-Metrix
temporal focus tradition; Plank & Zlomuzica (2024) for coherence-oriented
mental-health context. FTD construct: Incoherence, via temporal organization
of discourse.
"""

import logging

import numpy as np
import spacy

from scripts.features.base import FeatureExtractorBase

logger = logging.getLogger(__name__)


class TenseExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["past_ratio", "present_ratio", "future_ratio"]
    DIM = 3

    def __init__(self):
        self.future_modals = frozenset({"will", "shall", "'ll"})
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

        past = present = future = 0
        future_indices = set()
        for token in doc:
            if token.lemma_.lower() in self.future_modals:
                future += 1
                future_indices.add(token.i)
                if token.head.pos_ == "VERB":
                    future_indices.add(token.head.i)

        for token in doc:
            if token.pos_ not in ("VERB", "AUX") or token.i in future_indices:
                continue
            tense = token.morph.get("Tense")
            if not tense:
                continue
            if tense[0] == "Past":
                past += 1
            elif tense[0] == "Pres":
                present += 1

        total = past + present + future
        if total == 0:
            return np.zeros(self.DIM, dtype=np.float32)
        return self._validate([past / total, present / total, future / total])
