"""Computes MentalRoBERTa CLS semantic embeddings.

This extractor uses HuggingFace transformers with the MentalRoBERTa model
configured in scripts.config to produce the final-layer CLS embedding.
Citation: Ji et al. (2022), MentalBERT/MentalRoBERTa public mental-health
language models. FTD construct: none directly; semantic embeddings provide
contextual meaning complementary to FTD-anchored handcrafted features.
"""

import numpy as np
import torch

from scripts.config import MAX_LENGTH, MENTAL_ROBERTA_NAME
from scripts.features.base import FeatureExtractorBase


class MentalRobertaExtractor(FeatureExtractorBase):
    FEATURE_NAMES = [f"mental_roberta_cls_{i}" for i in range(768)]
    DIM = 768

    def __init__(self, model=None, tokenizer=None, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MENTAL_ROBERTA_NAME)
        self.model = AutoModel.from_pretrained(MENTAL_ROBERTA_NAME).to(self.device)
        self.model.eval()

    def extract(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.DIM, dtype=np.float32)
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._load()
        normalized = [
            text if isinstance(text, str) and text.strip() else ""
            for text in texts
        ]
        outputs = []
        for start in range(0, len(normalized), batch_size):
            batch_texts = normalized[start:start + batch_size]
            non_empty = [bool(text.strip()) for text in batch_texts]
            encoded_texts = [text if text.strip() else " " for text in batch_texts]
            encoded = self.tokenizer(
                encoded_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                model_outputs = self.model(**encoded)
                cls = (
                    model_outputs.last_hidden_state[:, 0, :]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            for row_index, keep in enumerate(non_empty):
                if not keep:
                    cls[row_index] = 0.0
            outputs.append(cls)
        if not outputs:
            return np.zeros((0, self.DIM), dtype=np.float32)
        matrix = np.vstack(outputs).astype(np.float32)
        if matrix.shape[1] != self.DIM:
            raise ValueError(f"MentalRoBERTa returned dim {matrix.shape[1]}; expected {self.DIM}")
        return matrix
