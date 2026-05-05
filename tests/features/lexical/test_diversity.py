import unittest

import numpy as np

from scripts.features.lexical.diversity import MTLDExtractor


class MTLDExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = MTLDExtractor().extract("I feel sad but I can explain why I feel sad.")
        self.assertEqual(features.shape, (1,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
