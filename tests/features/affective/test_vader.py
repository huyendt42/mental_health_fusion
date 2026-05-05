import unittest

import numpy as np

from scripts.features.affective.vader import VADERExtractor


class VADERExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = VADERExtractor().extract("I am sad. I am happy.")
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
