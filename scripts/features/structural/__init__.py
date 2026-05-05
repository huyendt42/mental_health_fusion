from __future__ import annotations

from typing import Dict, List

import numpy as np

from scripts.features.base import FeatureExtractorBase
from scripts.features.structural.coherence import CoherenceExtractor
from scripts.features.structural.tense import TenseExtractor


class StructuralExtractor(FeatureExtractorBase):
    DIM = 7

    def __init__(self, sentence_model=None):
        self.coherence = CoherenceExtractor(sentence_model=sentence_model)
        self.tense = TenseExtractor()

    def extract(self, text: str) -> np.ndarray:
        return self._validate(
            np.concatenate([self.coherence.extract(text), self.tense.extract(text)])
        )

    @property
    def feature_names(self) -> List[str]:
        return self.coherence.feature_names + self.tense.feature_names

    @property
    def sub_extractors(self) -> Dict[str, FeatureExtractorBase]:
        return {"coherence": self.coherence, "tense": self.tense}


STRUCTURAL_FEATURE_NAMES: List[str] = (
    CoherenceExtractor.FEATURE_NAMES + TenseExtractor.FEATURE_NAMES
)


def compute_structural_features(text: str, sentence_model=None) -> np.ndarray:
    return StructuralExtractor(sentence_model=sentence_model).extract(text)


__all__ = ["STRUCTURAL_FEATURE_NAMES", "StructuralExtractor", "compute_structural_features"]
