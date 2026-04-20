# scripts/features/combine.py

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    LABEL_COL,
    SEMANTIC_FEATURES_DIR,
    AFFECTIVE_FEATURES_DIR,
    STRUCTURAL_FEATURES_DIR,
    STYLISTIC_FEATURES_DIR,
    FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE GROUP REGISTRY
# =============================================================================
# Each group lists its sub-features with their source directory and filename
# prefix. For each sub-feature we also record whether it's stored as CSV
# (for small feature sets, human-readable) or NPZ (for dense embeddings).
#
# This registry is the single source of truth for "what features exist in
# which group". Adding a new feature = adding one entry here.

FEATURE_GROUPS = {
    "semantic": [
        # (sub_name, directory, file_format)
        ("roberta",           SEMANTIC_FEATURES_DIR / "roberta",           "npz"),
        ("psycholinguistic",  SEMANTIC_FEATURES_DIR / "psycholinguistic",  "csv"),
        ("lexical_diversity", SEMANTIC_FEATURES_DIR / "lexical_diversity", "csv"),
    ],
    "affective": [
        ("emotions",      AFFECTIVE_FEATURES_DIR / "emotions",      "csv"),
        ("vad",           AFFECTIVE_FEATURES_DIR / "vad",           "csv"),
        ("sentiment_arc", AFFECTIVE_FEATURES_DIR / "sentiment_arc", "csv"),
    ],
    "structural": [
        ("speech_graph",        STRUCTURAL_FEATURES_DIR / "speech_graph",        "csv"),
        ("discourse_coherence", STRUCTURAL_FEATURES_DIR / "discourse_coherence", "csv"),
        ("tense_distribution",  STRUCTURAL_FEATURES_DIR / "tense_distribution",  "csv"),
    ],
    "stylistic": [
        ("syntactic",   STYLISTIC_FEATURES_DIR / "syntactic",   "csv"),
        ("readability", STYLISTIC_FEATURES_DIR / "readability", "csv"),
        ("hedging",     STYLISTIC_FEATURES_DIR / "hedging",     "csv"),
    ],
}

# Where to save combined per-group matrices
COMBINED_DIR = FEATURES_DIR / "combined"


# =============================================================================
# SINGLE-SUBFEATURE LOADER
# =============================================================================

def load_sub_feature(
    sub_name: str,
    sub_dir: Path,
    file_format: str,
    split: str,
) -> np.ndarray:
    """
    Load one sub-feature file for one split, returning a 2D numpy array.

    Input:
        sub_name    — sub-feature name like "emotions" or "roberta"
        sub_dir     — directory containing the feature files
        file_format — "csv" or "npz"
        split       — "train", "val", or "test"

    Output:
        2D numpy array of shape (num_samples, num_features_for_this_subfeature)
        Row order matches the original processed CSV order.

    Why return ndarray instead of DataFrame?
    The fusion network consumes numpy arrays (converted to torch tensors).
    Returning ndarray here avoids a conversion step later.
    DataFrames retain column names which is useful for analysis but adds
    memory overhead we don't need during training.
    """
    if file_format == "csv":
        file_path = sub_dir / f"{sub_name}_{split}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing feature file: {file_path}\n"
                f"Run the extractor for {sub_name} before combining."
            )

        df = pd.read_csv(file_path)
        # .values gives the underlying numpy array of the DataFrame
        # astype(float32) matches PyTorch's default dtype for efficiency
        return df.values.astype(np.float32)

    elif file_format == "npz":
        file_path = sub_dir / f"{sub_name}_{split}.npz"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing feature file: {file_path}\n"
                f"Run the extractor for {sub_name} before combining."
            )

        # np.load on an .npz file returns a dict-like object;
        # our convention is to save under the key "embeddings"
        data = np.load(file_path)
        return data["embeddings"].astype(np.float32)

    else:
        raise ValueError(f"Unknown file format: {file_format}")


# =============================================================================
# PER-GROUP COMBINER
# =============================================================================

def combine_group(group_name: str, split: str) -> np.ndarray:
    """
    Load and horizontally stack all sub-features for one group.

    Input:
        group_name — "semantic", "affective", "structural", or "stylistic"
        split      — "train", "val", or "test"

    Output:
        2D numpy array of shape (num_samples, total_group_dim)
        For semantic: (N, 775) — 768 roberta + 5 psych + 2 lex
        For affective: (N, 34) — 28 emo + 3 vad + 3 arc
        For structural: (N, 15) — 10 graph + 2 coherence + 3 tense
        For stylistic: (N, 12) — 6 synt + 2 readability + 4 hedging

    Why horizontal stack?
    All sub-features have the same number of ROWS (one per post, in the
    same order). They differ in COLUMNS. np.hstack concatenates along
    axis=1 which is exactly what we need for combining features.

    Alignment check:
    We verify that all loaded sub-features have the same row count before
    stacking. If they don't match, something went wrong in extraction
    (someone dropped rows, shuffled indices, etc.) and we must catch it
    here rather than silently proceed with misaligned data.
    """
    sub_features = FEATURE_GROUPS[group_name]
    arrays       = []

    logger.info(f"  Loading sub-features for {group_name} ({split})...")

    for sub_name, sub_dir, file_format in sub_features:
        arr = load_sub_feature(sub_name, sub_dir, file_format, split)
        logger.info(f"    {sub_name:<20} shape: {arr.shape}")
        arrays.append(arr)

    # Verify row count alignment
    row_counts = [arr.shape[0] for arr in arrays]
    if len(set(row_counts)) != 1:
        raise RuntimeError(
            f"Row count mismatch in {group_name} ({split}): {row_counts}\n"
            f"Sub-features have inconsistent sample counts. "
            f"Re-run feature extraction to fix."
        )

    # Horizontal stack — combines along the column axis
    combined = np.hstack(arrays)
    logger.info(f"  Combined {group_name} ({split}): shape {combined.shape}")

    return combined


# =============================================================================
# LABELS LOADER
# =============================================================================

def load_labels(split: str) -> np.ndarray:
    """
    Load class_id labels for one split.

    Input:
        split — "train", "val", or "test"

    Output:
        1D numpy array of shape (num_samples,), dtype int64

    Why int64?
    PyTorch's cross-entropy loss expects target labels as int64 (LongTensor).
    Loading as int64 directly avoids a type conversion later.
    """
    path_map = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}

    df = pd.read_csv(path_map[split])
    labels = df[LABEL_COL].values.astype(np.int64)
    logger.info(f"  Labels ({split}): shape {labels.shape}, "
                f"unique classes: {sorted(np.unique(labels).tolist())}")
    return labels


# =============================================================================
# SAVE
# =============================================================================

def save_combined(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    split: str,
) -> None:
    """
    Save combined feature matrices and labels for one split.

    Input:
        arrays — dict mapping group_name → combined matrix for that group
        labels — 1D label array
        split  — "train", "val", or "test"

    Output:
        None — saves one .npz file per group containing both features
               and labels, plus one shared labels file.

    File layout after save:
        results/features/combined/
        ├── semantic_train.npz     (contains "features" and "labels")
        ├── affective_train.npz
        ├── structural_train.npz
        ├── stylistic_train.npz
        ├── semantic_val.npz
        ├── ... (same for val and test)

    Why bundle labels with each group file?
    During fusion training, we load all 4 groups plus labels. If labels
    were in a separate file, every script would need to remember to load
    them separately. Bundling them means any single file load gives you
    a self-contained training pair (features + labels) — simpler and
    less error-prone.
    """
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    for group_name, matrix in arrays.items():
        save_path = COMBINED_DIR / f"{group_name}_{split}.npz"

        # np.savez_compressed stores multiple arrays in one file,
        # accessible by the keys we provide ("features", "labels")
        np.savez_compressed(
            save_path,
            features=matrix,
            labels=labels,
        )

        logger.info(
            f"  Saved → {save_path} | "
            f"features: {matrix.shape}, labels: {labels.shape}"
        )


# =============================================================================
# LOADER (for downstream fusion training)
# =============================================================================

def load_combined(group_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a previously combined group file.

    Input:
        group_name — "semantic", "affective", "structural", or "stylistic"
        split      — "train", "val", or "test"

    Output:
        (features, labels) tuple
        features: shape (num_samples, group_dim)
        labels:   shape (num_samples,)

    This function is what scripts/models/fusion/train.py will call.
    Keeping it in the same file as save_combined ensures the save/load
    format stays consistent — if we change save format, the loader
    changes with it automatically.
    """
    file_path = COMBINED_DIR / f"{group_name}_{split}.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Combined features not found: {file_path}\n"
            f"Run `python -m scripts.features.combine` first."
        )

    data = np.load(file_path)
    return data["features"], data["labels"]


# =============================================================================
# PIPELINE
# =============================================================================

def run_combine_pipeline() -> None:
    """
    Combine all sub-features into per-group matrices for all three splits.

    For each split:
        1. Load labels
        2. For each group, load and stack its sub-features
        3. Save (features + labels) as a single .npz per group

    Produces 12 output files total (4 groups × 3 splits).
    """
    splits = ["train", "val", "test"]

    logger.info("=" * 60)
    logger.info("Starting feature combining pipeline")
    logger.info("=" * 60)

    for split in splits:
        logger.info(f"\n--- Processing {split} split ---")

        # Load labels once per split
        labels = load_labels(split)

        # Combine each group for this split
        combined_arrays = {}
        for group_name in FEATURE_GROUPS.keys():
            try:
                combined_arrays[group_name] = combine_group(group_name, split)
            except FileNotFoundError as e:
                logger.error(f"Skipping {group_name}: {e}")
                continue

        # Save all combined arrays for this split
        save_combined(combined_arrays, labels, split)

    logger.info("=" * 60)
    logger.info("Feature combining complete.")
    logger.info(f"Output: {COMBINED_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_combine_pipeline()