"""Computes readability indices with textstat.

This extractor uses textstat to compute Flesch-Kincaid Grade Level and
Gunning Fog Index. Citations: Flesch-Kincaid and Gunning Fog readability
formulas; Tanaka et al. (2022) for mental-health feature use. FTD construct:
Disorganization, via surface complexity and readability.
"""

import logging
import math

import numpy as np
import textstat

from scripts.features.base import FeatureExtractorBase

logger = logging.getLogger(__name__)


class ReadabilityExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["flesch_kincaid_grade", "gunning_fog_index"]
    DIM = 2

    def _clean(self, value: float) -> float:
        if not isinstance(value, (int, float)) or math.isnan(value):
            return 0.0
        return max(0.0, float(value))

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        try:
            return self._validate(
                [
                    self._clean(textstat.flesch_kincaid_grade(text)),
                    self._clean(textstat.gunning_fog(text)),
                ]
            )
        except Exception as exc:
            logger.debug("textstat failed for text prefix %r: %s", str(text)[:50], exc)
            return np.zeros(self.DIM, dtype=np.float32)
