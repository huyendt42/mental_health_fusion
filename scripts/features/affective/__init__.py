from __future__ import annotations

from typing import Dict, List

import numpy as np

from scripts.features.affective.goemotions import GoEmotionsExtractor
from scripts.features.affective.vad import VADExtractor
from scripts.features.affective.vader import VADERExtractor
from scripts.features.base import FeatureExtractorBase


class AffectiveExtractor(FeatureExtractorBase):
    DIM = 34

    def __init__(self, emotion_pipeline=None):
        self.goemotions = GoEmotionsExtractor(emotion_pipeline=emotion_pipeline)
        self.vad = VADExtractor()
        self.vader = VADERExtractor()

    def extract(self, text: str) -> np.ndarray:
        return self._validate(
            np.concatenate(
                [
                    self.goemotions.extract(text),
                    self.vad.extract(text),
                    self.vader.extract(text),
                ]
            )
        )

    @property
    def feature_names(self) -> List[str]:
        return self.goemotions.feature_names + self.vad.feature_names + self.vader.feature_names

    @property
    def sub_extractors(self) -> Dict[str, FeatureExtractorBase]:
        return {"goemotions": self.goemotions, "vad": self.vad, "vader": self.vader}


AFFECTIVE_FEATURE_NAMES: List[str] = (
    GoEmotionsExtractor.FEATURE_NAMES
    + VADExtractor.FEATURE_NAMES
    + VADERExtractor.FEATURE_NAMES
)
GOEMOTIONS_LABELS: List[str] = [
    name.replace("goemotions_mean_", "")
    for name in GoEmotionsExtractor.FEATURE_NAMES
]


def compute_affective_features(text: str, emotion_pipeline=None) -> np.ndarray:
    return AffectiveExtractor(emotion_pipeline=emotion_pipeline).extract(text)


def compute_vad_features(text: str) -> np.ndarray:
    return VADExtractor().extract(text)


__all__ = [
    "AFFECTIVE_FEATURE_NAMES",
    "AffectiveExtractor",
    "GOEMOTIONS_LABELS",
    "compute_affective_features",
    "compute_vad_features",
]
