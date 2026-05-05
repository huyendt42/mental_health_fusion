from __future__ import annotations

from typing import Dict, List

import numpy as np

from scripts.features.base import FeatureExtractorBase
from scripts.features.lexical.diversity import MTLDExtractor
from scripts.features.lexical.pronouns import PronounExtractor
from scripts.features.lexical.punctuation import PunctuationExtractor
from scripts.features.lexical.word_rates import WordRatesExtractor


class LexicalExtractor(FeatureExtractorBase):
    DIM = 11

    def __init__(self):
        self.diversity = MTLDExtractor()
        self.word_rates = WordRatesExtractor()
        self.pronouns = PronounExtractor()
        self.punctuation = PunctuationExtractor()

    def extract(self, text: str) -> np.ndarray:
        return self._validate(
            np.concatenate(
                [
                    self.diversity.extract(text),
                    self.word_rates.extract(text),
                    self.pronouns.extract(text),
                    self.punctuation.extract(text),
                ]
            )
        )

    @property
    def feature_names(self) -> List[str]:
        return (
            self.diversity.feature_names
            + self.word_rates.feature_names
            + self.pronouns.feature_names
            + self.punctuation.feature_names
        )

    @property
    def sub_extractors(self) -> Dict[str, FeatureExtractorBase]:
        return {
            "diversity": self.diversity,
            "word_rates": self.word_rates,
            "pronouns": self.pronouns,
            "punctuation": self.punctuation,
        }


LEXICAL_FEATURE_NAMES: List[str] = (
    MTLDExtractor.FEATURE_NAMES
    + WordRatesExtractor.FEATURE_NAMES
    + PronounExtractor.FEATURE_NAMES
    + PunctuationExtractor.FEATURE_NAMES
)


def compute_lexical_features(text: str) -> np.ndarray:
    return LexicalExtractor().extract(text)


__all__ = ["LEXICAL_FEATURE_NAMES", "LexicalExtractor", "compute_lexical_features"]
