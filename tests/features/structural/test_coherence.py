import unittest

import numpy as np

from scripts.features.structural.coherence import CoherenceExtractor


class FakeSentenceModel:
    def encode(self, sentences, show_progress_bar=False, convert_to_numpy=True):
        return np.ones((len(sentences), 3), dtype=np.float32)


class CoherenceExtractorTests(unittest.TestCase):
    def test_shape_and_identical_coherence(self):
        features = CoherenceExtractor(sentence_model=FakeSentenceModel()).extract(
            "Same sentence. Same sentence. Same sentence."
        )
        self.assertEqual(features.shape, (4,))
        self.assertAlmostEqual(float(features[0]), 1.0, places=6)
        self.assertAlmostEqual(float(features[1]), 0.0, places=6)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
