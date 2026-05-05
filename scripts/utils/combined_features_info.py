"""
Utility to inspect and test combined feature files.
Provides functions to load, validate, and export combined features.
"""
from pathlib import Path
import pandas as pd
import json
from scripts import config

GROUPS = ["semantic", "affective", "lexical", "syntactic", "structural"]
SPLITS = ["train", "val", "test"]


def load_combined_features(group: str, split: str) -> pd.DataFrame:
    """Load combined features for a group and split."""
    path = config.FEATURES_DIR / group / split / "combined.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Combined features not found: {path}")
    return pd.read_parquet(path)


def get_combined_info(group: str, split: str) -> dict:
    """Get information about combined features."""
    try:
        df = load_combined_features(group, split)
        return {
            "group": group,
            "split": split,
            "path": str(config.FEATURES_DIR / group / split / "combined.parquet"),
            "rows": len(df),
            "feature_dim": len(df["features"].iloc[0]) if len(df) > 0 else 0,
            "sample_post_ids": df["post_id"].tolist()[:3],
            "sample_features_shape": len(df["features"].iloc[0]) if len(df) > 0 else 0,
        }
    except Exception as e:
        return {
            "group": group,
            "split": split,
            "error": str(e),
        }


def print_combined_summary():
    """Print summary of all combined features."""
    print("\n" + "="*80)
    print("COMBINED FEATURES SUMMARY")
    print("="*80 + "\n")
    
    for split in SPLITS:
        print(f"\n{'─'*80}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'─'*80}")
        
        for group in GROUPS:
            info = get_combined_info(group, split)
            if "error" in info:
                print(f"  {group:15} ❌ {info['error']}")
            else:
                print(f"  {group:15} ✓ {info['rows']:6} rows × {info['feature_dim']:4} dims | "
                      f"Post_ids: {info['sample_post_ids']}")


def export_combined_summary():
    """Export combined features info as JSON."""
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for group in GROUPS:
            summary[split][group] = get_combined_info(group, split)
    
    output_path = config.FEATURES_DIR / "combined_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary exported to: {output_path}")
    return summary


def load_all_combined_tensors(split: str) -> dict:
    """Load all combined features for a split as numpy arrays."""
    import numpy as np
    tensors = {}
    for group in GROUPS:
        try:
            df = load_combined_features(group, split)
            post_ids = df["post_id"].tolist()
            features = np.asarray(df["features"].tolist(), dtype=np.float32)
            tensors[group] = {
                "post_ids": post_ids,
                "features": features,
                "shape": features.shape,
            }
        except Exception as e:
            print(f"Error loading {group}/{split}: {e}")
    return tensors


def validate_combined_consistency():
    """Validate that post_ids are consistent across groups within each split."""
    print("\n" + "="*80)
    print("VALIDATING POST_ID CONSISTENCY")
    print("="*80 + "\n")
    
    issues = []
    for split in SPLITS:
        print(f"\nSplit: {split}")
        reference_ids = None
        for group in GROUPS:
            try:
                df = load_combined_features(group, split)
                post_ids = df["post_id"].tolist()
                
                if reference_ids is None:
                    reference_ids = post_ids
                    print(f"  {group:15} ✓ Reference: {len(post_ids)} post_ids")
                elif post_ids == reference_ids:
                    print(f"  {group:15} ✓ Consistent")
                else:
                    msg = f"Mismatch in {split}/{group}: {len(post_ids)} vs {len(reference_ids)}"
                    print(f"  {group:15} ❌ {msg}")
                    issues.append(msg)
            except Exception as e:
                print(f"  {group:15} ❌ Error: {e}")
                issues.append(str(e))
    
    if not issues:
        print("\n✅ All combined features have consistent post_ids!")
    else:
        print(f"\n⚠️  Found {len(issues)} validation issues")
        for issue in issues:
            print(f"   - {issue}")
    
    return len(issues) == 0


if __name__ == "__main__":
    print_combined_summary()
    validate_combined_consistency()
    export_combined_summary()
