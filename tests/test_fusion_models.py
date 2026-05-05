import unittest

import torch

from scripts import config
from scripts.models.fusion import GatedFusion, LateConcatFusion, build_fusion_model
from scripts.models.fusion.feature_loader import GROUP_SUBFEATURES


class FusionModelTests(unittest.TestCase):
    def _inputs(self, batch_size=5):
        return (
            torch.randn(batch_size, config.SEMANTIC_DIM),
            torch.randn(batch_size, config.AFFECTIVE_DIM),
            torch.randn(batch_size, config.HANDCRAFTED_DIM),
        )

    def _assert_forward_backward(self, model):
        semantic, affective, handcrafted = self._inputs()
        logits = model(semantic, affective, handcrafted)
        self.assertEqual(logits.shape, (semantic.shape[0], config.NUM_LABELS))
        self.assertTrue(torch.isfinite(logits).all())
        loss = logits.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(grad is not None for grad in grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads if grad is not None))

    def test_late_concat_forward_backward(self):
        self._assert_forward_backward(LateConcatFusion())

    def test_gated_forward_backward(self):
        self._assert_forward_backward(GatedFusion())

    def test_late_concat_branch_representations(self):
        model = LateConcatFusion()
        semantic, affective, handcrafted = self._inputs(batch_size=3)
        sem_repr, aff_repr, hand_repr = model.get_branch_representations(
            semantic, affective, handcrafted
        )
        self.assertEqual(sem_repr.shape, (3, 256))
        self.assertEqual(aff_repr.shape, (3, 128))
        self.assertEqual(hand_repr.shape, (3, 64))

    def test_factory_reads_config_default(self):
        model = build_fusion_model()
        self.assertIsInstance(model, LateConcatFusion)

    def test_factory_reads_lowercase_config_flag(self):
        original = config.fusion_type
        try:
            config.fusion_type = "gated"
            model = build_fusion_model()
            self.assertIsInstance(model, GatedFusion)
        finally:
            config.fusion_type = original

    def test_factory_gated(self):
        model = build_fusion_model("gated")
        self.assertIsInstance(model, GatedFusion)

    def test_feature_loader_group_registry(self):
        self.assertEqual(GROUP_SUBFEATURES["semantic"], ["mental_roberta"])
        self.assertEqual(
            GROUP_SUBFEATURES["affective"],
            ["goemotions", "vad", "vader"],
        )

    def test_split_aware_feature_loader_path_error(self):
        from scripts.models.fusion.feature_loader import load_subextractor_features

        with self.assertRaises(FileNotFoundError) as ctx:
            load_subextractor_features("semantic", "mental_roberta", split="missing")
        msg = str(ctx.exception)
        self.assertIn("mental_roberta.parquet", msg)
        self.assertIn("missing", msg)


if __name__ == "__main__":
    unittest.main()
