import unittest

import numpy as np

from scripts.features.syntactic.readability import ReadabilityExtractor


class ReadabilityExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = ReadabilityExtractor().extract("This is a simple sentence. This is another simple sentence.")
        self.assertEqual(features.shape, (2,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
