import unittest

import numpy as np

from scripts.features.syntactic.complexity import ComplexityExtractor


class ComplexityExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = ComplexityExtractor().extract("The big red house is quiet.")
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
