import unittest

import numpy as np

from scripts.features.structural.tense import TenseExtractor


class TenseExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = TenseExtractor().extract("I went home. I am here. I will go.")
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
