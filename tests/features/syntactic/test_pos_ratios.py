import unittest

import numpy as np

from scripts.features.syntactic.pos_ratios import POSRatioExtractor


class POSRatioExtractorTests(unittest.TestCase):
    def test_shape_and_adjective_ratio(self):
        features = POSRatioExtractor().extract("The big red house")
        self.assertEqual(features.shape, (3,))
        self.assertAlmostEqual(float(features[0]), 0.5, places=6)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
