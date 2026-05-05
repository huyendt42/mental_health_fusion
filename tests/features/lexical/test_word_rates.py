import unittest

import numpy as np

from scripts.features.lexical.word_rates import WordRatesExtractor


class WordRatesExtractorTests(unittest.TestCase):
    def test_shape_and_known_rates(self):
        features = WordRatesExtractor().extract("death always not might maybe")
        self.assertEqual(features.shape, (5,))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features > 0))


if __name__ == "__main__":
    unittest.main()
