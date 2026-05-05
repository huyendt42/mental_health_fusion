"""Computes GoEmotions per-sentence mean emotion probabilities.

This extractor uses HuggingFace transformers with
SamLowe/roberta-base-go_emotions to score each sentence and averages the
28 emotion probabilities across sentences. Citation: Demszky et al. (2020).
FTD construct: none directly; this captures condition-specific affective
tone alongside FTD-anchored features.
"""

import numpy as np

from scripts.config import EMOTION_BATCH_SIZE, EMOTION_MODEL_NAME
from scripts.features.base import FeatureExtractorBase, split_sentences


class GoEmotionsExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [
        "goemotions_mean_admiration",
        "goemotions_mean_amusement",
        "goemotions_mean_anger",
        "goemotions_mean_annoyance",
        "goemotions_mean_approval",
        "goemotions_mean_caring",
        "goemotions_mean_confusion",
        "goemotions_mean_curiosity",
        "goemotions_mean_desire",
        "goemotions_mean_disappointment",
        "goemotions_mean_disapproval",
        "goemotions_mean_disgust",
        "goemotions_mean_embarrassment",
        "goemotions_mean_excitement",
        "goemotions_mean_fear",
        "goemotions_mean_gratitude",
        "goemotions_mean_grief",
        "goemotions_mean_joy",
        "goemotions_mean_love",
        "goemotions_mean_nervousness",
        "goemotions_mean_optimism",
        "goemotions_mean_pride",
        "goemotions_mean_realization",
        "goemotions_mean_relief",
        "goemotions_mean_remorse",
        "goemotions_mean_sadness",
        "goemotions_mean_surprise",
        "goemotions_mean_neutral",
    ]
    DIM = 28

    def __init__(self, emotion_pipeline=None):
        self.emotion_pipeline = emotion_pipeline

    def _pipeline(self):
        if self.emotion_pipeline is None:
            import torch
            from transformers import pipeline

            self.emotion_pipeline = pipeline(
                task="text-classification",
                model=EMOTION_MODEL_NAME,
                top_k=None,
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
            )
        return self.emotion_pipeline

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        sentences = split_sentences(text)
        if not sentences:
            return np.zeros(self.DIM, dtype=np.float32)
        results = self._pipeline()(sentences, batch_size=EMOTION_BATCH_SIZE)
        labels = [name.replace("goemotions_mean_", "") for name in self.FEATURE_NAMES]
        rows = []
        for result in results:
            scores = {item["label"]: float(item["score"]) for item in result}
            rows.append([scores.get(label, 0.0) for label in labels])
        return self._validate(np.asarray(rows, dtype=np.float32).mean(axis=0))
