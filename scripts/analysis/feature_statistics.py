# scripts/analysis/feature_statistics.py
"""
Per-class feature profile analysis for all interpretable feature groups.

Semantic features (768-dim MentalRoBERTa embeddings) are excluded — individual
dimensions have no linguistic interpretation and cannot be meaningfully plotted.
The four remaining groups (affective, lexical, syntactic, structural) contain
named features that can be read directly off a heatmap.

Normalisation:
    A StandardScaler is fit on the full training split before computing
    per-class mean profiles.  This puts all features on the same scale
    (zero-mean, unit-variance) so that unbounded features such as MTLD or
    readability grades do not visually swamp bounded ratios in the heatmap.
    The scaler is fit once per group and applied to the same training data
    used for profiling.

Output layout (one subdirectory per group):
    results/plots/feature_statistics/{group}/
        {group}_heatmap.png
        {group}_profile_raw.csv          ← actual class-mean values (scaled)
        {group}_profile_normalized.csv   ← min-max [0,1] across classes
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from scripts.config import (
    AFFECTIVE_FEATURES_DIR,
    CLASS_NAME_COL,
    LEXICAL_FEATURES_DIR,
    PLOTS_DIR,
    STRUCTURAL_FEATURES_DIR,
    SYNTACTIC_FEATURES_DIR,
    TRAIN_PATH,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="white")


# =============================================================================
# REGISTRIES
# =============================================================================
# Only interpretable groups — semantic (768-dim embeddings) is excluded.

INTERPRETABLE_GROUPS = ["affective", "lexical", "syntactic", "structural"]

GROUP_BASE_DIRS = {
    "affective":  AFFECTIVE_FEATURES_DIR,
    "lexical":    LEXICAL_FEATURES_DIR,
    "syntactic":  SYNTACTIC_FEATURES_DIR,
    "structural": STRUCTURAL_FEATURES_DIR,
}

# Sub-feature files that exist in each group's split directory
GROUP_SUBFEATURES = {
    "affective":  ["goemotions", "vad", "vader"],
    "lexical":    ["diversity", "word_rates", "pronouns", "punctuation"],
    "syntactic":  ["complexity", "pos_ratios", "readability"],
    "structural": ["coherence", "tense"],
}

# Human-readable column names — must match the order features are concatenated
SUB_FEATURE_NAMES = {
    "affective": {
        "goemotions": [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral",
        ],
        "vad":   ["mean_valence", "mean_arousal", "mean_dominance"],
        "vader": ["sentiment_mean", "sentiment_std", "sentiment_range"],
    },
    "lexical": {
        "diversity":   ["mtld"],
        "word_rates":  ["death_harm_rate", "absolutist_rate", "negation_rate",
                        "modal_rate", "hedge_rate"],
        "pronouns":    ["1p_singular_rate", "1p_plural_rate", "2p_rate"],
        "punctuation": ["question_mark_rate", "ellipsis_rate"],
    },
    "syntactic": {
        "complexity": ["mean_dep_distance", "mean_tree_depth", "mean_sent_length"],
        "pos_ratios": ["adjective_ratio", "adverb_ratio", "pronoun_ratio"],
        "readability": ["flesch_kincaid_grade", "gunning_fog"],
    },
    "structural": {
        "coherence": ["mean_coherence", "std_coherence", "topic_drift", "break_rate"],
        "tense":     ["past_ratio", "present_ratio", "future_ratio"],
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _feature_path(group: str, sub_name: str, split: str) -> Path:
    """Return the parquet path for one sub-feature, split-aware."""
    base = GROUP_BASE_DIRS[group]
    candidate = base / split / f"{sub_name}.parquet"
    if candidate.exists():
        return candidate
    # Fallback: some older extractions saved without split subfolder
    fallback = base / f"{sub_name}.parquet"
    return fallback


def load_group_features(group: str, split: str = "train") -> pd.DataFrame:
    """
    Load all sub-features for *group* / *split*, concatenate horizontally,
    and assign proper column names.

    Returns a DataFrame with shape (n_samples, n_features_in_group).
    """
    frames = []
    for sub_name in GROUP_SUBFEATURES[group]:
        path = _feature_path(group, sub_name, split)
        if not path.exists():
            logger.warning("Missing parquet — skipping: %s", path)
            continue

        df  = pd.read_parquet(path)
        mat = np.asarray(df["features"].tolist(), dtype=np.float32)

        names = SUB_FEATURE_NAMES[group].get(sub_name)
        if names is None or len(names) != mat.shape[1]:
            # Fallback to generic names if something is off
            names = [f"{sub_name}_{i}" for i in range(mat.shape[1])]

        frames.append(pd.DataFrame(mat, columns=names))
        logger.info("  Loaded %s/%s: %s", group, sub_name, mat.shape)

    if not frames:
        raise RuntimeError(
            f"No feature files found for group '{group}' / split '{split}'. "
            "Run feature extraction first."
        )
    return pd.concat(frames, axis=1)


def load_labels(split: str = "train") -> pd.Series:
    from scripts.config import TEST_PATH, VAL_PATH
    path_map = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    return pd.read_csv(path_map[split])[CLASS_NAME_COL]


# =============================================================================
# NORMALISATION + PROFILING
# =============================================================================

def standardize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a StandardScaler on *features_df* and return the scaled DataFrame.

    Each column becomes zero-mean and unit-variance across all training rows.
    This ensures that MTLD (~50-300) and readability grades (~0-20) do not
    visually dominate bounded ratios (~0-1) in the heatmap.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df.values)
    return pd.DataFrame(scaled, columns=features_df.columns)


def compute_class_profile(features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-class mean feature values.  Returns (n_classes, n_features)."""
    combined = features_df.copy()
    combined["__class__"] = labels.values
    return combined.groupby("__class__").mean().sort_index()


def min_max_normalize(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Scale each feature column to [0, 1] across classes so the heatmap shows
    'which class is highest / lowest on this feature' rather than raw magnitude.
    Constant columns (no variation across classes) are set to 0.5.
    """
    col_min   = profile.min()
    col_max   = profile.max()
    col_range = col_max - col_min
    normalized = profile.copy()
    for col in profile.columns:
        if col_range[col] == 0:
            normalized[col] = 0.5
        else:
            normalized[col] = (profile[col] - col_min[col]) / col_range[col]
    return normalized


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_heatmap(normalized_profile: pd.DataFrame, group: str, save_dir) -> None:
    """Heatmap: rows = features, columns = classes, values in [0, 1]."""
    plot_data = normalized_profile.T          # transpose → features as rows
    n_features = plot_data.shape[0]
    fig_height = max(6, n_features * 0.40 + 2)

    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(
        plot_data,
        ax        = ax,
        annot     = True,
        fmt       = ".2f",
        cmap      = "YlGnBu",
        cbar      = True,
        cbar_kws  = {"label": "Normalised class mean", "shrink": 0.7},
        linewidths = 0.5,
        linecolor  = "white",
        annot_kws  = {"size": 9},
    )
    ax.set_title(
        f"{group.title()} Feature Fingerprint by Mental Health Condition",
        fontsize=14, pad=20,
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Feature",   fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()

    path = save_dir / f"{group}_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Heatmap → %s", path)


# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_profile_tables(
    profile: pd.DataFrame,
    normalized_profile: pd.DataFrame,
    group: str,
    save_dir,
) -> None:
    raw_path  = save_dir / f"{group}_profile_raw.csv"
    norm_path = save_dir / f"{group}_profile_normalized.csv"
    profile.to_csv(raw_path)
    normalized_profile.to_csv(norm_path)
    logger.info("  Raw profile       → %s", raw_path)
    logger.info("  Normalised profile→ %s", norm_path)


def report_top_features(normalized_profile: pd.DataFrame, group: str, top_k: int = 3) -> None:
    logger.info("\n  Top %d features per class (%s):", top_k, group)
    for cls in normalized_profile.index:
        top = normalized_profile.loc[cls].sort_values(ascending=False).head(top_k)
        logger.info("    %-12s %s", cls,
                    ", ".join(f"{f} ({v:.2f})" for f, v in top.items()))


# =============================================================================
# PIPELINE
# =============================================================================

def analyze_group(group: str, base_save_dir) -> None:
    """Full analysis pipeline for one feature group."""
    logger.info("\n%s\nAnalysing group: %s\n%s", "=" * 60, group, "=" * 60)

    # Per-group subdirectory
    save_dir = base_save_dir / group
    save_dir.mkdir(parents=True, exist_ok=True)

    features_df = load_group_features(group, split="train")
    labels      = load_labels(split="train")

    if len(features_df) != len(labels):
        raise RuntimeError(
            f"Row count mismatch in {group}: "
            f"features={len(features_df)}, labels={len(labels)}"
        )

    # Normalise raw features before profiling so scales are comparable
    scaled_df = standardize_features(features_df)

    profile            = compute_class_profile(scaled_df, labels)
    normalized_profile = min_max_normalize(profile)

    plot_heatmap(normalized_profile, group, save_dir)
    save_profile_tables(profile, normalized_profile, group, save_dir)
    report_top_features(normalized_profile, group)


def run_feature_statistics() -> None:
    """
    Run analysis for all interpretable groups (affective, lexical, syntactic,
    structural).  Results land in results/plots/feature_statistics/{group}/.
    """
    base_save_dir = PLOTS_DIR / "feature_statistics"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Feature statistics analysis — %d groups", len(INTERPRETABLE_GROUPS))
    logger.info("=" * 60)

    for group in INTERPRETABLE_GROUPS:
        try:
            analyze_group(group, base_save_dir)
        except Exception as exc:
            logger.error("Failed to analyse %s: %s", group, exc)

    logger.info("\n%s", "=" * 60)
    logger.info("Done. Results in: %s", base_save_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_feature_statistics()
