"""
Test script for combined features.
Demonstrates how to load and use combined feature files for model testing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.utils.combined_features_info import load_combined_features, load_all_combined_tensors
from scripts import config


def test_load_single_group():
    """Test loading combined features for a single group."""
    print("\n" + "="*80)
    print("TEST 1: Load Single Group Combined Features")
    print("="*80)
    
    df = load_combined_features("semantic", "train")
    print(f"Semantic Train Combined:")
    print(f"  Shape: {len(df)} samples × {len(df['features'].iloc[0])} dimensions")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  First 3 post_ids: {df['post_id'].tolist()[:3]}")
    print(f"  First sample features shape: {np.array(df['features'].iloc[0]).shape}")


def test_load_all_groups():
    """Test loading all combined features for a split."""
    print("\n" + "="*80)
    print("TEST 2: Load All Groups for a Split")
    print("="*80)
    
    tensors = load_all_combined_tensors("train")
    print(f"\nLoaded {len(tensors)} groups for train split:")
    for group, data in tensors.items():
        print(f"  {group:12} shape: {data['shape']}")


def test_concatenate_features():
    """Test concatenating all features into a single matrix."""
    print("\n" + "="*80)
    print("TEST 3: Concatenate All Features into Single Matrix")
    print("="*80)
    
    tensors = load_all_combined_tensors("train")
    
    # Concatenate all feature groups
    matrices = [data["features"] for data in tensors.values()]
    combined_matrix = np.concatenate(matrices, axis=1)
    
    print(f"\nConcatenated shape: {combined_matrix.shape}")
    print(f"  Rows: {combined_matrix.shape[0]} (samples)")
    print(f"  Cols: {combined_matrix.shape[1]} (total features)")
    
    # Feature breakdown
    print("\nFeature breakdown:")
    col = 0
    for group, data in tensors.items():
        dims = data["features"].shape[1]
        print(f"  {group:12} cols {col:4}-{col+dims-1:4} ({dims:3} dims)")
        col += dims


def test_post_id_alignment():
    """Test that post_ids are aligned across all groups."""
    print("\n" + "="*80)
    print("TEST 4: Verify Post_ID Alignment")
    print("="*80)
    
    for split in ["train", "val", "test"]:
        print(f"\nSplit: {split}")
        tensors = load_all_combined_tensors(split)
        
        reference_ids = None
        aligned = True
        for group, data in tensors.items():
            post_ids = data["post_ids"]
            if reference_ids is None:
                reference_ids = post_ids
            elif post_ids != reference_ids:
                print(f"  ❌ {group} post_ids misaligned!")
                aligned = False
        
        if aligned:
            print(f"  ✅ All groups aligned ({len(reference_ids)} samples)")


def test_data_types():
    """Test data types and ranges."""
    print("\n" + "="*80)
    print("TEST 5: Data Type and Value Analysis")
    print("="*80)
    
    tensors = load_all_combined_tensors("train")
    for group, data in tensors.items():
        features = data["features"]
        print(f"\n{group}:")
        print(f"  dtype: {features.dtype}")
        print(f"  min: {features.min():.6f}, max: {features.max():.6f}")
        print(f"  mean: {features.mean():.6f}, std: {features.std():.6f}")
        print(f"  NaN count: {np.isnan(features).sum()}")


def test_save_numpy():
    """Test saving combined features as numpy arrays."""
    print("\n" + "="*80)
    print("TEST 6: Save Combined Features as NumPy Arrays")
    print("="*80)
    
    output_dir = config.FEATURES_DIR / "numpy_export"
    output_dir.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        print(f"\nSaving {split}...")
        tensors = load_all_combined_tensors(split)
        
        for group, data in tensors.items():
            features_file = output_dir / f"{split}_{group}_features.npy"
            post_ids_file = output_dir / f"{split}_{group}_post_ids.txt"
            
            np.save(features_file, data["features"])
            with open(post_ids_file, 'w') as f:
                f.write('\n'.join(data["post_ids"]))
            
            print(f"  ✓ {features_file.name} ({data['shape']})")
    
    print(f"\nExported to: {output_dir}")


if __name__ == "__main__":
    test_load_single_group()
    test_load_all_groups()
    test_concatenate_features()
    test_post_id_alignment()
    test_data_types()
    test_save_numpy()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
