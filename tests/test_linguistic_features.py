import unittest

import numpy as np

from scripts.features.lexical import compute_lexical_features
from scripts.features.syntactic import compute_syntactic_features
from scripts.features.structural import compute_structural_features
from scripts.features.affective import GOEMOTIONS_LABELS, compute_affective_features, compute_vad_features


class FakeSentenceModel:
    def encode(self, sentences, show_progress_bar=False, convert_to_numpy=True):
        rows = []
        for sentence in sentences:
            if sentence.lower().startswith("same"):
                rows.append([1.0, 0.0, 0.0])
            else:
                rows.append([0.0, 1.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


class FakeEmotionPipeline:
    def __call__(self, sentences, batch_size=16):
        rows = []
        for _ in sentences:
            rows.append(
                [
                    {"label": label, "score": 1.0 if label == "neutral" else 0.0}
                    for label in GOEMOTIONS_LABELS
                ]
            )
        return rows


class LinguisticFeatureTests(unittest.TestCase):
    def test_lexical_output_shape(self):
        features = compute_lexical_features("I might be somewhat sad...")
        self.assertEqual(features.shape, (11,))
        self.assertEqual(features.dtype, np.float32)

    def test_syntactic_output_shape(self):
        features = compute_syntactic_features("The big red house is quiet.")
        self.assertEqual(features.shape, (8,))
        self.assertEqual(features.dtype, np.float32)

    def test_first_person_singular_rate(self):
        features = compute_lexical_features("I I I am very sad. We never go.")
        self.assertAlmostEqual(float(features[6]), 3 / 9, places=6)

    def test_adjective_ratio(self):
        features = compute_syntactic_features("The big red house")
        self.assertAlmostEqual(float(features[3]), 0.5, places=6)

    def test_empty_and_single_word_inputs_do_not_crash(self):
        self.assertEqual(compute_lexical_features("").shape, (11,))
        self.assertEqual(compute_lexical_features("sad").shape, (11,))
        self.assertEqual(compute_syntactic_features("").shape, (8,))
        self.assertEqual(compute_syntactic_features("sad").shape, (8,))

    def test_structural_output_shape_and_identical_sentence_coherence(self):
        features = compute_structural_features(
            "Same sentence. Same sentence. Same sentence.",
            sentence_model=FakeSentenceModel(),
        )
        self.assertEqual(features.shape, (7,))
        self.assertAlmostEqual(float(features[0]), 1.0, places=6)
        self.assertAlmostEqual(float(features[1]), 0.0, places=6)

    def test_affective_output_shape(self):
        features = compute_affective_features(
            "I am terrified.",
            emotion_pipeline=FakeEmotionPipeline(),
        )
        self.assertEqual(features.shape, (34,))

    def test_terrified_vad(self):
        vad = compute_vad_features("I am terrified")
        self.assertLess(float(vad[0]), 0.5)
        self.assertGreater(float(vad[1]), 0.5)

    def test_all_values_finite(self):
        structural = compute_structural_features(
            "Same sentence. Same sentence. Same sentence.",
            sentence_model=FakeSentenceModel(),
        )
        affective = compute_affective_features(
            "I am terrified.",
            emotion_pipeline=FakeEmotionPipeline(),
        )
        self.assertTrue(np.isfinite(structural).all())
        self.assertTrue(np.isfinite(affective).all())


if __name__ == "__main__":
    unittest.main()
