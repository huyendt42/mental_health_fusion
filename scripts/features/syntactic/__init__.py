from __future__ import annotations

from typing import Dict, List

import numpy as np

from scripts.features.base import FeatureExtractorBase
from scripts.features.syntactic.complexity import ComplexityExtractor
from scripts.features.syntactic.pos_ratios import POSRatioExtractor
from scripts.features.syntactic.readability import ReadabilityExtractor


class SyntacticExtractor(FeatureExtractorBase):
    DIM = 8

    def __init__(self):
        self.complexity = ComplexityExtractor()
        self.pos_ratios = POSRatioExtractor()
        self.readability = ReadabilityExtractor()

    def extract(self, text: str) -> np.ndarray:
        return self._validate(
            np.concatenate(
                [
                    self.complexity.extract(text),
                    self.pos_ratios.extract(text),
                    self.readability.extract(text),
                ]
            )
        )

    @property
    def feature_names(self) -> List[str]:
        return (
            self.complexity.feature_names
            + self.pos_ratios.feature_names
            + self.readability.feature_names
        )

    @property
    def sub_extractors(self) -> Dict[str, FeatureExtractorBase]:
        return {
            "complexity": self.complexity,
            "pos_ratios": self.pos_ratios,
            "readability": self.readability,
        }


SYNTACTIC_FEATURE_NAMES: List[str] = (
    ComplexityExtractor.FEATURE_NAMES
    + POSRatioExtractor.FEATURE_NAMES
    + ReadabilityExtractor.FEATURE_NAMES
)


def compute_syntactic_features(text: str) -> np.ndarray:
    return SyntacticExtractor().extract(text)


__all__ = ["SYNTACTIC_FEATURE_NAMES", "SyntacticExtractor", "compute_syntactic_features"]
