"""Computes sentence embedding discourse coherence features.

This extractor uses sentence-transformers all-MiniLM-L6-v2 to embed
sentences and calculate consecutive coherence mean/std, first-last topic
drift, and coherence break rate. Citations: Plank & Zlomuzica (2024);
Bedi et al. (2015). FTD construct: Incoherence, via disrupted discourse
continuity.
"""

import logging

import numpy as np

from scripts.features.base import FeatureExtractorBase, split_sentences

logger = logging.getLogger(__name__)


class CoherenceExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [
        "mean_consecutive_coherence",
        "std_consecutive_coherence",
        "topic_drift",
        "coherence_break_rate",
    ]
    DIM = 4

    def __init__(self, sentence_model=None):
        self.sentence_model = sentence_model

    def _model(self):
        if self.sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.sentence_model

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            logger.debug("Coherence skipped: fewer than 2 sentences.")
            return np.zeros(self.DIM, dtype=np.float32)
        sentences = split_sentences(text)
        if len(sentences) < 2:
            logger.debug("Coherence skipped: fewer than 2 sentences.")
            return np.zeros(self.DIM, dtype=np.float32)

        embeddings = np.asarray(
            self._model().encode(sentences, show_progress_bar=False, convert_to_numpy=True),
            dtype=np.float32,
        )
        similarities = np.array(
            [self._cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)],
            dtype=np.float32,
        )
        return self._validate(
            [
                float(similarities.mean()),
                float(similarities.std(ddof=0)),
                self._cosine(embeddings[0], embeddings[-1]),
                float(np.mean(similarities < 0.3)),
            ]
        )
