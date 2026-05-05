"""Computes VADER sentence-level sentiment arc statistics.

This extractor uses vaderSentiment to compute compound sentiment per sentence
and returns mean, standard deviation, and range. Citation: Hutto & Gilbert
(2014). FTD construct: none directly; sentiment dynamics capture affective
variation complementary to FTD features.
"""

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scripts.features.base import FeatureExtractorBase, split_sentences


class VADERExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["sentiment_mean", "sentiment_std", "sentiment_range"]
    DIM = 3

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        sentences = split_sentences(text)
        if not sentences:
            return np.zeros(self.DIM, dtype=np.float32)
        scores = np.asarray(
            [self.analyzer.polarity_scores(sentence)["compound"] for sentence in sentences],
            dtype=np.float32,
        )
        return self._validate(
            [float(scores.mean()), float(scores.std(ddof=0)), float(scores.max() - scores.min())]
        )
