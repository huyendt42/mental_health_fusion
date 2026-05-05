"""Computes NRC-VAD mean affective ratings.

This extractor uses the NRC Valence-Arousal-Dominance lexicon to average
matched token valence, arousal, and dominance scores. Citation: Mohammad
(2018). FTD construct: none directly; VAD captures affective tone outside
the FTD dimensions.
"""

import numpy as np

from scripts.config import NRC_VAD_LEXICON_PATH
from scripts.features.base import FeatureExtractorBase, WORD_RE


class VADExtractor(FeatureExtractorBase):
    FEATURE_NAMES = ["mean_valence", "mean_arousal", "mean_dominance"]
    DIM = 3

    def __init__(self):
        self.lexicon = self._load_lexicon()
        self.stopwords = frozenset({
            "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
            "ourselves", "you", "your", "yours", "yourself", "a", "an",
            "the", "am", "is", "are", "was", "were", "be", "been", "being",
        })

    def _load_lexicon(self) -> dict[str, tuple[float, float, float]]:
        if not NRC_VAD_LEXICON_PATH.exists():
            raise FileNotFoundError(
                "Missing NRC-VAD lexicon. Download it from "
                "http://saifmohammad.com/WebPages/nrc-vad.html and place it at "
                f"{NRC_VAD_LEXICON_PATH}"
            )
        lexicon = {}
        with NRC_VAD_LEXICON_PATH.open("r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    word, valence, arousal, dominance = parts
                    lexicon[word.lower()] = (float(valence), float(arousal), float(dominance))
        if not lexicon:
            raise ValueError(f"NRC-VAD lexicon is empty or malformed: {NRC_VAD_LEXICON_PATH}")
        return lexicon

    def extract(self, text: str) -> np.ndarray:
        neutral = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        if not isinstance(text, str) or not text.strip():
            return neutral
        tokens = [m.group(0).lower() for m in WORD_RE.finditer(text)]
        matches = [
            self.lexicon[token]
            for token in tokens
            if token not in self.stopwords and token in self.lexicon
        ]
        if not matches:
            return neutral
        return self._validate(np.asarray(matches, dtype=np.float32).mean(axis=0))
