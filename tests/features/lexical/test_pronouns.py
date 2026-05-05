import unittest

import numpy as np

from scripts.features.lexical.pronouns import PronounExtractor


class PronounExtractorTests(unittest.TestCase):
    def test_shape_and_known_rate(self):
        features = PronounExtractor().extract("I I I am very sad. We never go.")
        self.assertEqual(features.shape, (3,))
        self.assertAlmostEqual(float(features[0]), 3 / 9, places=6)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
