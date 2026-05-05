import unittest

import numpy as np
import torch

from scripts.features.semantic.mental_roberta import MentalRobertaExtractor


class FakeTokenizer:
    def __call__(self, text, return_tensors, truncation, padding, max_length):
        batch_size = len(text) if isinstance(text, list) else 1
        return {"input_ids": torch.ones((batch_size, 3), dtype=torch.long)}


class FakeOutput:
    def __init__(self, batch_size):
        self.last_hidden_state = torch.ones((batch_size, 3, 768), dtype=torch.float32)


class FakeModel:
    def __call__(self, **kwargs):
        return FakeOutput(kwargs["input_ids"].shape[0])


class MentalRobertaExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = MentalRobertaExtractor(
            model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu"
        ).extract("sample text")
        self.assertEqual(features.shape, (768,))
        self.assertTrue(np.isfinite(features).all())

    def test_batch_shape_and_empty_rows(self):
        features = MentalRobertaExtractor(
            model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu"
        ).extract_batch(["sample text", ""], batch_size=2)
        self.assertEqual(features.shape, (2, 768))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[1] == 0.0))


if __name__ == "__main__":
    unittest.main()
