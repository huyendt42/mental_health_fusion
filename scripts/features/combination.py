from pathlib import Path
import pandas as pd
from scripts.models.fusion.feature_loader import load_group_features
from scripts import config

GROUPS = ["semantic", "affective", "lexical", "syntactic", "structural"]
SPLITS = ["train", "val", "test"]


def combine_feature_group(group: str, split: str):
    post_ids, matrix = load_group_features(group, split=split)
    print(f"{split} / {group}: {matrix.shape}")

    # Optional: save combined group back to parquet
    out_dir = config.FEATURES_DIR / group / split
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "combined.parquet"
    pd.DataFrame(
        {"post_id": post_ids, "features": matrix.tolist()}
    ).to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    for split in SPLITS:
        for group in GROUPS:
            try:
                combine_feature_group(group, split)
            except Exception as e:
                print(f"Error combining {group}/{split}: {e}")