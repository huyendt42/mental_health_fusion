import unittest

import numpy as np

from scripts.features.lexical.punctuation import PunctuationExtractor


class PunctuationExtractorTests(unittest.TestCase):
    def test_shape_and_rates(self):
        features = PunctuationExtractor().extract("Why? I don't know...")
        self.assertEqual(features.shape, (2,))
        self.assertAlmostEqual(float(features[0]), 0.5, places=6)
        self.assertAlmostEqual(float(features[1]), 0.5, places=6)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
