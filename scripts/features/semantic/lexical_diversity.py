# scripts/features/semantic/lexical_diversity.py

import logging
import pandas as pd
from lexicalrichness import LexicalRichness

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    SEMANTIC_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_lexical_diversity(text: str) -> dict:
    """
    Compute TTR and MTLD lexical diversity for a single text.

    Input:
        text — a single merged post string

    Output:
        dict with 2 keys:
        - ttr  : Type-Token Ratio = unique_words / total_words, in [0, 1]
        - mtld : Measure of Textual Lexical Diversity, length-normalized

    Why two measures?
        TTR decreases naturally for longer texts even when vocabulary
        is genuinely rich — a 500-word post will always have lower TTR
        than a 50-word post using the same vocabulary. This makes TTR
        unreliable for comparing posts of different lengths.

        MTLD corrects for this by measuring the average length of
        sequential text segments that maintain TTR above a threshold
        (0.72 by convention). Posts with varied vocabulary maintain
        high TTR for longer before dropping below the threshold.
        This makes MTLD reliable across Reddit posts ranging from
        29 to 6,900+ words.

    Sources:
        - TTR: Standard lexical measure, no single citation needed.
        - MTLD: McCarthy & Jarvis (2010). MTLD, vocd-D, and HD-D:
          A validation study of sophisticated approaches to lexical
          diversity assessment. Behavior Research Methods, 42(2), 381-392.
    """
    if not isinstance(text, str) or text.strip() == "":
        return {"ttr": 0.0, "mtld": 0.0}

    try:
        lex = LexicalRichness(text)

        # Need at least 2 words for any meaningful diversity measure
        if lex.words < 2:
            return {"ttr": 0.0, "mtld": 0.0}

        ttr = lex.ttr

        try:
            # threshold=0.72 is the standard convention from McCarthy &
            # Jarvis (2010) — do not change this value arbitrarily as it
            # affects comparability with other studies using MTLD
            mtld = lex.mtld(threshold=0.72)
        except ZeroDivisionError:
            # Occurs when all tokens are identical (TTR = 0),
            # causing division by zero inside the MTLD algorithm.
            # Return 0.0 rather than crashing the pipeline.
            mtld = 0.0

        return {
            "ttr" : round(ttr,  6),
            "mtld": round(mtld, 6),
        }

    except Exception as e:
        # Catch any other unexpected errors from unusual text inputs.
        # Log the error for debugging but continue processing.
        logger.debug(f"Lexical diversity error: {e} | text[:50]: {str(text)[:50]}")
        return {"ttr": 0.0, "mtld": 0.0}


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract lexical diversity features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/semantic/lexical_diversity_{split}.csv
        Shape: (num_samples, 2)

    Note on runtime:
        LexicalRichness processes text in pure Python with no GPU support.
        Expect ~3-5 minutes per split on CPU for 13k samples.
        This is acceptable since this script only runs once.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    features = df[TEXT_COL].apply(
        lambda text: pd.Series(compute_lexical_diversity(text))
    )

    save_dir  = SEMANTIC_FEATURES_DIR / "lexical_diversity"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"lexical_diversity_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_lexical_diversity_pipeline() -> None:
    """
    Extract and save lexical diversity features for all three splits.
    Output: 2-dimensional feature vector per sample [ttr, mtld].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting lexical diversity feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Lexical diversity extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_lexical_diversity_pipeline()