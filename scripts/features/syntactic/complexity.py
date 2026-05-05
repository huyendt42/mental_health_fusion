"""Computes parser-based syntactic complexity features.

This extractor uses spaCy dependency parses to compute mean dependency
distance, mean parse tree depth, and mean sentence length. Citations:
McNamara et al. (2014), Tanaka et al. (2022), Spencer et al. (2025).
FTD construct: Disorganization, via reduced or irregular grammatical
integration.
"""

import logging

import numpy as np
import spacy

from scripts.features.base import FeatureExtractorBase

logger = logging.getLogger(__name__)


class ComplexityExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [
        "mean_dependency_distance",
        "mean_parse_tree_depth",
        "mean_sentence_length",
    ]
    DIM = 3

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy English model not found. Run: python -m spacy download en_core_web_sm"
            ) from exc

    def _tree_depth(self, root) -> int:
        max_depth = depth = 0
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            stack.extend((child, depth + 1) for child in node.children)
        return max_depth

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.debug("spaCy parse failed for text prefix %r: %s", str(text)[:50], exc)
            return np.zeros(self.DIM, dtype=np.float32)

        tokens = [token for token in doc if not token.is_space]
        word_tokens = [token for token in tokens if not token.is_punct]
        if not word_tokens:
            return np.zeros(self.DIM, dtype=np.float32)

        dep_distances = [abs(token.i - token.head.i) for token in tokens]
        sentence_lengths = []
        sentence_depths = []
        for sent in doc.sents:
            sent_words = [t for t in sent if not t.is_punct and not t.is_space]
            if sent_words:
                sentence_lengths.append(len(sent_words))
            roots = [t for t in sent if t.head == t]
            if roots:
                sentence_depths.append(max(self._tree_depth(root) for root in roots))

        return self._validate(
            [
                float(np.mean(dep_distances)) if dep_distances else 0.0,
                float(np.mean(sentence_depths)) if sentence_depths else 0.0,
                float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            ]
        )
