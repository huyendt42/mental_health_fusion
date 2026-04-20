# scripts/features/affective/sentiment_arc.py

import logging
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    AFFECTIVE_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# NLTK SETUP
# =============================================================================
# NLTK's sentence tokenizer (Punkt) requires a one-time download.
# We check and download silently if missing — users don't need to do it manually.

try:
    # Test whether the tokenizer is available
    sent_tokenize("Test sentence.")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# =============================================================================
# VADER SETUP
# =============================================================================
# SentimentIntensityAnalyzer is lightweight — initialization takes ~0.1 seconds.
# We still load it at module level (not per row) to follow the consistent
# pattern across all feature extractors.

logger.info("Loading VADER sentiment analyzer...")
vader = SentimentIntensityAnalyzer()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_sentiment_arc(text: str) -> dict:
    """
    Compute the sentiment arc statistics for a single post.

    Input:
        text — a single merged post string

    Output:
        dict with 3 keys:
        - mean_sentiment  : average compound sentiment across sentences, in [-1, 1]
        - std_sentiment   : standard deviation of sentence sentiments, >= 0
        - range_sentiment : max - min sentence sentiment, in [0, 2]

    Algorithm:
        1. Split the post into sentences using NLTK's Punkt tokenizer
        2. For each sentence, compute VADER compound sentiment score
        3. Extract three statistics: mean, std, range

    Design decisions:
        - Why compound score only? VADER returns four scores per sentence:
          positive, negative, neutral, and compound. The compound score is
          a single normalized value that combines all four — the most useful
          single number for arc analysis. Using just compound keeps the
          feature vector compact and interpretable.

        - Why three statistics instead of the full arc? A post with 20
          sentences would give 20 sentiment values — too many to use as
          features directly, and variable in length per post. Summarizing
          into mean/std/range gives a fixed 3-dim vector per post regardless
          of post length, while preserving the essential dynamics.

        - Why not max and min separately? Range (= max - min) captures
          emotional spread more compactly than two separate values. If you
          need max and min later, you can still extract them — but for the
          fusion network, 3 dimensions is the right granularity.
    """
    # Default neutral values for edge cases
    neutral = {
        "mean_sentiment" : 0.0,
        "std_sentiment"  : 0.0,
        "range_sentiment": 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return neutral

    # Split into sentences using NLTK's Punkt tokenizer.
    # Punkt handles abbreviations (Dr., Mr., etc.) and multiple punctuation
    # correctly — much better than splitting on ".!?" with regex.
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.debug(f"Sentence tokenization failed: {e}")
        return neutral

    # Need at least 1 sentence with content
    if len(sentences) == 0:
        return neutral

    # Compute compound sentiment for each sentence
    # polarity_scores returns a dict: {neg, neu, pos, compound}
    # We take only compound which is the normalized overall score
    sentiment_scores = [
        vader.polarity_scores(sent)["compound"]
        for sent in sentences
        if sent.strip()   # Skip empty sentences that can result from edge cases
    ]

    # Edge case: all sentences were empty after stripping
    if len(sentiment_scores) == 0:
        return neutral

    # Convert to numpy array for fast statistical operations
    scores = np.array(sentiment_scores)

    # Single-sentence posts have std = 0 and range = 0 — this is correct.
    # We don't return neutral for these because a single strong sentiment
    # is still meaningful (e.g., "I want to die." scores -0.8).
    mean_val  = float(scores.mean())
    std_val   = float(scores.std())
    range_val = float(scores.max() - scores.min())

    return {
        "mean_sentiment" : round(mean_val,  6),
        "std_sentiment"  : round(std_val,   6),
        "range_sentiment": round(range_val, 6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract sentiment arc features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/affective/sentiment_arc_{split}.csv
        Shape: (num_samples, 3)

    Runtime:
        VADER is pure rule-based Python — no GPU or heavy computation.
        Expect ~5-10 minutes for 13k samples depending on avg. post length,
        since longer posts have more sentences to score.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Sentiment arc ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_sentiment_arc(text))
    )

    save_dir  = AFFECTIVE_FEATURES_DIR / "sentiment_arc"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"sentiment_arc_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_sentiment_arc_pipeline() -> None:
    """
    Extract and save sentiment arc features for all three splits.
    Output: 3-dimensional feature vector per sample
            [mean_sentiment, std_sentiment, range_sentiment].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting sentiment arc feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Sentiment arc extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_sentiment_arc_pipeline()