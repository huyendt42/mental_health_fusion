from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

# Shared across all lexical/token-rate extractors (word_rates, pronouns, vad).
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Shared sentence splitter used by coherence, goemotions, and vader.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


class FeatureExtractorBase(ABC):
    FEATURE_NAMES: List[str] = []
    DIM: int = 0

    @abstractmethod
    def extract(self, text: str) -> np.ndarray:
        pass

    @property
    def feature_names(self) -> List[str]:
        return list(self.FEATURE_NAMES)

    @property
    def sub_extractors(self) -> Dict[str, "FeatureExtractorBase"]:
        return {}

    def _validate(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape != (self.DIM,):
            raise ValueError(
                f"{self.__class__.__name__} returned shape {arr.shape}; expected {(self.DIM,)}"
            )
        return arr
