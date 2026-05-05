import unittest

import numpy as np

from scripts.features.affective.goemotions import GoEmotionsExtractor


class FakeEmotionPipeline:
    def __call__(self, sentences, batch_size=16):
        labels = [name.replace("goemotions_mean_", "") for name in GoEmotionsExtractor.FEATURE_NAMES]
        return [
            [{"label": label, "score": 1.0 if label == "neutral" else 0.0} for label in labels]
            for _ in sentences
        ]


class GoEmotionsExtractorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        features = GoEmotionsExtractor(emotion_pipeline=FakeEmotionPipeline()).extract("I am here.")
        self.assertEqual(features.shape, (28,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
