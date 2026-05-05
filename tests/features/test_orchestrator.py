import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features.base import FeatureExtractorBase
from scripts.features.orchestrator import FeatureOrchestrator


class FakeSubExtractor(FeatureExtractorBase):
    def __init__(self, dim):
        self.DIM = dim
        self.FEATURE_NAMES = [f"f{i}" for i in range(dim)]

    def extract(self, text: str) -> np.ndarray:
        return np.ones(self.DIM, dtype=np.float32)


class FakeGroup(FeatureExtractorBase):
    def __init__(self, sub_dims):
        self.parts = {name: FakeSubExtractor(dim) for name, dim in sub_dims.items()}
        self.DIM = sum(sub_dims.values())

    def extract(self, text: str) -> np.ndarray:
        return np.concatenate([part.extract(text) for part in self.parts.values()])

    @property
    def sub_extractors(self):
        return self.parts


class FeatureOrchestratorTests(unittest.TestCase):
    def test_extract_all_shapes(self):
        orchestrator = FeatureOrchestrator(
            groups={
                "semantic": FakeGroup({"mental_roberta": 768}),
                "lexical": FakeGroup({"diversity": 1, "word_rates": 5, "pronouns": 3, "punctuation": 2}),
                "syntactic": FakeGroup({"complexity": 3, "pos_ratios": 3, "readability": 2}),
                "structural": FakeGroup({"coherence": 4, "tense": 3}),
                "affective": FakeGroup({"goemotions": 28, "vad": 3, "vader": 3}),
            }
        )
        outputs = orchestrator.extract_all("sample text")
        self.assertEqual(outputs["semantic"].shape, (768,))
        self.assertEqual(outputs["lexical"].shape, (11,))
        self.assertEqual(outputs["syntactic"].shape, (8,))
        self.assertEqual(outputs["structural"].shape, (7,))
        self.assertEqual(outputs["affective"].shape, (34,))

    def test_split_aware_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = FeatureOrchestrator(
                groups={"lexical": FakeGroup({"pronouns": 3})}
            )
            orchestrator.group_dirs = {"lexical": Path(tmp) / "lexical"}
            df = pd.DataFrame({"post_id": ["a"], "text": ["I am here"]})
            outputs = orchestrator.extract_dataset(
                df,
                text_col="text",
                post_id_col="post_id",
                components="lexical.pronouns",
                split="train",
                force=True,
            )
            output_path = outputs["lexical"]["pronouns"]
            self.assertEqual(output_path.name, "pronouns.parquet")
            self.assertEqual(output_path.parent.name, "train")
            self.assertEqual(output_path.parent.parent.name, "lexical")


if __name__ == "__main__":
    unittest.main()
