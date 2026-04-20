# scripts/analysis/feature_statistics.py

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scripts.config import (
    TRAIN_PATH,
    CLASS_NAME_COL,
    SEMANTIC_FEATURES_DIR,
    AFFECTIVE_FEATURES_DIR,
    STRUCTURAL_FEATURES_DIR,
    STYLISTIC_FEATURES_DIR,
    PLOTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PLOT STYLING
# =============================================================================
# Consistent styling across all heatmaps gives a professional thesis look.
# Set once at module load time so every plot function inherits the same config.

sns.set_theme(style="white")


# =============================================================================
# FEATURE GROUP REGISTRY
# =============================================================================
# Each entry maps group_name → list of (sub_name, sub_dir) tuples.
# This registry is the single source of truth for which features exist and
# where they live on disk. Adding a new feature means adding one entry here;
# the rest of the code adapts automatically.
#
# We exclude roberta_embed (768-dim) because heatmaps of 768 abstract
# dimensions are not interpretable. RoBERTa's contribution is captured
# downstream in the fusion network and gate weight analysis, not here.

FEATURE_GROUPS = {
    "semantic": [
        ("psycholinguistic", SEMANTIC_FEATURES_DIR / "psycholinguistic"),
        ("lexical_diversity", SEMANTIC_FEATURES_DIR / "lexical_diversity"),
    ],
    "affective": [
        ("emotions",      AFFECTIVE_FEATURES_DIR / "emotions"),
        ("vad",           AFFECTIVE_FEATURES_DIR / "vad"),
        ("sentiment_arc", AFFECTIVE_FEATURES_DIR / "sentiment_arc"),
    ],
    "structural": [
        ("speech_graph",        STRUCTURAL_FEATURES_DIR / "speech_graph"),
        ("discourse_coherence", STRUCTURAL_FEATURES_DIR / "discourse_coherence"),
        ("tense_distribution",  STRUCTURAL_FEATURES_DIR / "tense_distribution"),
    ],
    "stylistic": [
        ("syntactic",   STYLISTIC_FEATURES_DIR / "syntactic"),
        ("readability", STYLISTIC_FEATURES_DIR / "readability"),
        ("hedging",     STYLISTIC_FEATURES_DIR / "hedging"),
    ],
}


# =============================================================================
# LOAD FEATURES
# =============================================================================

def load_group_features(group_name: str, split: str = "train") -> pd.DataFrame:
    """
    Load and concatenate all sub-feature files for one group.

    Input:
        group_name — "semantic", "affective", "structural", or "stylistic"
        split      — "train", "val", or "test". Analysis uses train by default.

    Output:
        DataFrame containing all sub-features concatenated column-wise.
        Rows = posts, columns = all feature dimensions in that group.

    Why concatenate them here?
    Each sub-feature was saved as a separate CSV to keep extraction modular.
    For analysis we want all features in one place. We concatenate them
    horizontally (axis=1) because all files have the same number of rows
    in the same order.
    """
    sub_features = FEATURE_GROUPS[group_name]
    dataframes   = []

    for sub_name, sub_dir in sub_features:
        file_path = sub_dir / f"{sub_name}_{split}.csv"
        if not file_path.exists():
            logger.warning(f"Missing feature file: {file_path} — skipping.")
            continue

        df = pd.read_csv(file_path)
        dataframes.append(df)
        logger.info(f"  Loaded {sub_name}: {df.shape}")

    if not dataframes:
        raise RuntimeError(
            f"No feature files found for group '{group_name}'. "
            f"Run feature extraction scripts first."
        )

    # Concatenate horizontally — all DataFrames must have the same row count
    return pd.concat(dataframes, axis=1)


def load_labels(split: str = "train") -> pd.Series:
    """
    Load the class name labels for the given split.

    Input:
        split — "train", "val", or "test"

    Output:
        pandas Series of class name strings ("ADHD", "Anxiety", ...)
    """
    path_map = {"train": TRAIN_PATH}
    df = pd.read_csv(path_map[split])
    return df[CLASS_NAME_COL]


# =============================================================================
# PER-CLASS STATISTICS
# =============================================================================

def compute_class_profile(features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Compute the mean feature value for each class.

    Input:
        features_df — DataFrame of shape (num_samples, num_features)
        labels      — Series of class names of length num_samples

    Output:
        DataFrame of shape (num_classes, num_features) where each row
        is the class-level mean vector for that class.

    Why means specifically?
    The mean is the clinically interpretable summary statistic. "Depression
    posts average a pronoun ratio of 0.18" is a claim a reviewer can
    verify against prior literature. Using median or other statistics
    would complicate interpretation.
    """
    # Combine features and labels so we can groupby
    combined = features_df.copy()
    combined["__class__"] = labels.values

    # Group by class and compute mean across all feature columns
    # .sort_index() ensures classes appear in alphabetical order consistently
    profile = combined.groupby("__class__").mean().sort_index()

    return profile


def min_max_normalize(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each feature column to [0, 1] range.

    Input:
        profile — class-level mean profile DataFrame

    Output:
        Same shape DataFrame with values scaled per column.

    Formula per column:
        normalized = (value - min) / (max - min)

    Why per-column instead of per-row?
    We want to compare how CLASSES differ on each feature — not how
    features differ within a class. Per-column normalization means each
    feature's highest value across classes becomes 1, and lowest becomes 0,
    making the heatmap show "which class leads in this feature?" clearly.

    Edge case:
    If a feature has identical values across all classes (max == min),
    division would be zero. We handle this by returning 0.5 (neutral)
    for that column — it contains no discriminative signal.
    """
    col_min = profile.min()
    col_max = profile.max()
    col_range = col_max - col_min

    # Where range is 0, set normalized value to 0.5 (neutral)
    # The np.where trick vectorizes the conditional cleanly
    normalized = profile.copy()
    for col in profile.columns:
        if col_range[col] == 0:
            normalized[col] = 0.5
        else:
            normalized[col] = (profile[col] - col_min[col]) / col_range[col]

    return normalized


# =============================================================================
# HEATMAP VISUALIZATION
# =============================================================================

def plot_heatmap(
    normalized_profile: pd.DataFrame,
    group_name: str,
    save_dir: Path,
) -> None:
    """
    Generate and save a heatmap of normalized class-level feature means.

    Input:
        normalized_profile — shape (num_classes, num_features), values in [0, 1]
        group_name         — for figure title and filename
        save_dir           — output directory

    Output:
        Saves PNG file to save_dir/{group_name}_heatmap.png

    Heatmap orientation:
        Rows    = features (so long feature names are readable as y-labels)
        Columns = classes  (so the 6 classes fit horizontally on the page)

    This means we transpose the profile DataFrame before plotting.
    """
    # Transpose: features become rows, classes become columns
    plot_data = normalized_profile.T

    # Figure height scales with number of features for readability
    # Each feature gets ~0.35 inches of vertical space
    n_features = plot_data.shape[0]
    fig_height = max(6, n_features * 0.35 + 2)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    sns.heatmap(
        plot_data,
        ax          = ax,
        annot       = True,      # show numeric values in each cell
        fmt         = ".2f",     # 2 decimal places
        cmap        = "YlGnBu",  # yellow-green-blue — colorblind-friendly
        cbar        = True,
        cbar_kws    = {"label": "Normalized class mean", "shrink": 0.7},
        linewidths  = 0.5,
        linecolor   = "white",
        annot_kws   = {"size": 9},
    )

    ax.set_title(
        f"{group_name.title()} Feature Fingerprint by Mental Health Condition",
        fontsize=14, pad=20,
    )
    ax.set_xlabel("Class",    fontsize=12)
    ax.set_ylabel("Feature",  fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()

    save_path = save_dir / f"{group_name}_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved heatmap → {save_path}")


# =============================================================================
# TEXT SUMMARY TABLE
# =============================================================================

def save_profile_table(
    profile: pd.DataFrame,
    normalized_profile: pd.DataFrame,
    group_name: str,
    save_dir: Path,
) -> None:
    """
    Save both raw and normalized class profiles as CSV tables.

    Input:
        profile            — raw per-class means
        normalized_profile — min-max normalized means
        group_name         — used for filenames
        save_dir           — output directory

    Output:
        Saves two CSVs:
        - {group_name}_profile_raw.csv
        - {group_name}_profile_normalized.csv

    Why both?
    The normalized version is for the heatmap — easy visual comparison.
    The raw version is for the thesis table — readers want to see actual
    values like "depression pronoun_ratio = 0.182" not just color intensity.
    Both are cheap to produce so we save both.
    """
    raw_path  = save_dir / f"{group_name}_profile_raw.csv"
    norm_path = save_dir / f"{group_name}_profile_normalized.csv"

    profile.to_csv(raw_path)
    normalized_profile.to_csv(norm_path)

    logger.info(f"  Saved raw table        → {raw_path}")
    logger.info(f"  Saved normalized table → {norm_path}")


# =============================================================================
# TOP-DIFFERENTIATING FEATURES
# =============================================================================

def report_top_features(
    normalized_profile: pd.DataFrame,
    group_name: str,
    top_k: int = 3,
) -> None:
    """
    Log the top features that differentiate each class.

    Input:
        normalized_profile — min-max normalized class profile
        group_name         — for log labeling
        top_k              — number of top features to report per class

    Output:
        None — logs to console

    Why this matters:
    A heatmap is visually informative but reviewers often want a concise
    textual summary: "What are Depression's top 3 features?" This function
    answers that directly. Great for thesis paragraph descriptions like:
    "Depression posts are characterized by the highest values in recurrence
    loops, first-person singular rate, and negative valence."
    """
    logger.info(f"\n--- Top {top_k} discriminative features per class ({group_name}) ---")
    for class_name in normalized_profile.index:
        top_features = (
            normalized_profile.loc[class_name]
            .sort_values(ascending=False)
            .head(top_k)
        )
        feature_strs = [
            f"{feat} ({val:.2f})"
            for feat, val in top_features.items()
        ]
        logger.info(f"  {class_name:<12}: {', '.join(feature_strs)}")


# =============================================================================
# PIPELINE
# =============================================================================

def analyze_group(group_name: str, save_dir: Path) -> None:
    """
    Run the full analysis for one feature group.

    Steps:
        1. Load all sub-features for the group
        2. Load class labels
        3. Compute per-class mean profile
        4. Normalize profile
        5. Generate heatmap
        6. Save raw and normalized CSV tables
        7. Print top discriminative features
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analyzing group: {group_name}")
    logger.info("=" * 60)

    features_df = load_group_features(group_name, split="train")
    labels      = load_labels(split="train")

    # Sanity check — shapes must align
    if len(features_df) != len(labels):
        raise RuntimeError(
            f"Row count mismatch in {group_name}: "
            f"features has {len(features_df)} rows, labels has {len(labels)}"
        )

    profile            = compute_class_profile(features_df, labels)
    normalized_profile = min_max_normalize(profile)

    plot_heatmap(normalized_profile, group_name, save_dir)
    save_profile_table(profile, normalized_profile, group_name, save_dir)
    report_top_features(normalized_profile, group_name)


def run_feature_statistics() -> None:
    """
    Generate feature statistics for all four groups.
    Produces 4 heatmaps + 8 CSV tables in results/plots/feature_statistics/.
    """
    save_dir = PLOTS_DIR / "feature_statistics"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting feature statistics analysis")
    logger.info("=" * 60)

    for group_name in FEATURE_GROUPS.keys():
        try:
            analyze_group(group_name, save_dir)
        except Exception as e:
            logger.error(f"Failed to analyze {group_name}: {e}")
            continue

    logger.info("=" * 60)
    logger.info(f"Feature statistics complete. Results in: {save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_feature_statistics()