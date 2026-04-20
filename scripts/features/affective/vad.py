# scripts/features/affective/vad.py

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    AFFECTIVE_FEATURES_DIR,
    DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# LEXICON PATH
# =============================================================================
# The NRC-VAD lexicon is a plain text file with columns:
#   word<TAB>valence<TAB>arousal<TAB>dominance
# Download: https://saifmohammad.com/WebPages/nrc-vad.html
# Place the file at: data/lexicons/NRC-VAD-Lexicon.txt
# The lexicon is freely available for research, non-commercial use.

LEXICON_PATH = DATA_DIR / "lexicons" / "NRC-VAD-Lexicon.txt"


# =============================================================================
# LOAD LEXICON
# =============================================================================

def load_vad_lexicon() -> dict[str, tuple[float, float, float]]:
    """
    Load the NRC-VAD lexicon from disk into a fast lookup dict.

    Returns:
        dict mapping word → (valence, arousal, dominance) tuple

    Why dict not DataFrame?
    Looking up a word in a dict is O(1). Looking up a word in a DataFrame
    is much slower. For processing 13k+ posts where each post may contain
    hundreds of words, dict lookup is the only viable option.

    File format (tab-separated):
        aaaaaaah    0.479    0.606    0.291
        aardvark    0.427    0.490    0.437
        abandon     0.052    0.481    0.298
        ...

    Why tuple as value instead of dict?
    Tuples are more memory-efficient than dicts and faster to unpack.
    We will unpack them immediately into separate lists during computation.
    """
    if not LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"NRC-VAD lexicon not found at: {LEXICON_PATH}\n"
            f"Download from: https://saifmohammad.com/WebPages/nrc-vad.html\n"
            f"Place the file at: {LEXICON_PATH}\n"
            f"Register at the NRC site, accept the research license, "
            f"and download 'NRC-VAD-Lexicon.txt'."
        )

    logger.info(f"Loading NRC-VAD lexicon from {LEXICON_PATH}...")

    # Read using pandas for simplicity — file is small (~20k rows)
    # so loading into a DataFrame first and converting to dict is fine
    df = pd.read_csv(
        LEXICON_PATH,
        sep="\t",
        header=0,                              # first row is column names
        names=["word", "valence", "arousal", "dominance"],
        skiprows=1,                            # skip the header we replaced
        dtype={"word": str,
               "valence": float,
               "arousal": float,
               "dominance": float}
    )

    # Convert to dict for fast lookup
    # zip() pairs up corresponding elements from multiple lists
    # .values gets the raw numpy arrays which are faster than pandas Series
    lexicon = dict(zip(
        df["word"].str.lower().values,
        zip(df["valence"].values,
            df["arousal"].values,
            df["dominance"].values)
    ))

    logger.info(f"Loaded {len(lexicon):,} words from NRC-VAD lexicon.")
    return lexicon


# Load lexicon once at module import time.
# Repeated loading inside the per-row function would add minutes of runtime.
VAD_LEXICON = load_vad_lexicon()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_vad_scores(text: str) -> dict:
    """
    Compute mean VAD scores for a single text.

    Input:
        text — a single merged post string

    Output:
        dict with 3 keys:
        - mean_valence   : average valence across matched words, in [0, 1]
        - mean_arousal   : average arousal across matched words, in [0, 1]
        - mean_dominance : average dominance across matched words, in [0, 1]

    Algorithm:
        1. Split text into lowercase word tokens
        2. For each token, look it up in the NRC-VAD lexicon
        3. Collect (valence, arousal, dominance) triples for matched tokens
        4. Return mean across all matched tokens

    Design decisions:
        - Why split() instead of spaCy? VAD matching is just word-level
          lookup — we don't need POS tags or parsing. .split() is 10x faster
          than spaCy tokenization and gives the same match results.

        - Why lowercase? The NRC-VAD lexicon contains only lowercase words.
          Without lowercasing, "Happy" and "HAPPY" would never match.

        - Why return 0.5 for empty matches? VAD is in [0, 1] so 0.5 is the
          neutral midpoint. Returning 0 would bias the feature toward
          "very negative / very calm / very submissive" which is wrong
          for text with no emotional words. 0.5 is the theoretical neutral.
    """
    # Default neutral values for edge cases (empty text, no matched words)
    neutral = {"mean_valence": 0.5, "mean_arousal": 0.5, "mean_dominance": 0.5}

    if not isinstance(text, str) or text.strip() == "":
        return neutral

    # Split on whitespace and lowercase — simple and fast
    tokens = text.lower().split()

    # Collect VAD triples for tokens that appear in the lexicon
    # This is a list comprehension filtering by dict membership
    matched = [VAD_LEXICON[t] for t in tokens if t in VAD_LEXICON]

    if len(matched) == 0:
        # No tokens matched the lexicon — return neutral
        return neutral

    # Convert list of tuples into numpy array for vectorized mean
    # Shape: (num_matches, 3) — rows are matches, columns are V/A/D
    vad_array = np.array(matched)

    # Mean along axis=0 averages down each column
    # Result is a 1D array of 3 values: [mean_v, mean_a, mean_d]
    mean_scores = vad_array.mean(axis=0)

    return {
        "mean_valence"   : round(float(mean_scores[0]), 6),
        "mean_arousal"   : round(float(mean_scores[1]), 6),
        "mean_dominance" : round(float(mean_scores[2]), 6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract VAD features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/affective/vad_{split}.csv
        Shape: (num_samples, 3)

    Runtime:
        VAD extraction is very fast — just dict lookup and mean.
        Expect <30 seconds for 13k samples on any machine, no GPU needed.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    # tqdm.pandas() enables .progress_apply() for a visual progress bar.
    # We use this because VAD is CPU-bound and users appreciate feedback
    # during multi-second loops.
    tqdm.pandas(desc=f"  VAD ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_vad_scores(text))
    )

    save_dir  = AFFECTIVE_FEATURES_DIR / "vad"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"vad_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_vad_pipeline() -> None:
    """
    Extract and save VAD features for all three splits.
    Output: 3-dimensional feature vector per sample [valence, arousal, dominance].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting VAD feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("VAD extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_vad_pipeline()