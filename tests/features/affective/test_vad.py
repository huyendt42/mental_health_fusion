import unittest

import numpy as np

from scripts.features.affective.vad import VADExtractor


class VADExtractorTests(unittest.TestCase):
    def test_terrified_vad(self):
        features = VADExtractor().extract("I am terrified")
        self.assertEqual(features.shape, (3,))
        self.assertLess(float(features[0]), 0.5)
        self.assertGreater(float(features[1]), 0.5)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
