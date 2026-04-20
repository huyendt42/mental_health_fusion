# scripts/features/stylistic/readability.py

import logging
import pandas as pd
import textstat
from tqdm import tqdm

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    STYLISTIC_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_readability(text: str) -> dict:
    """
    Compute Flesch-Kincaid Grade Level and Gunning Fog Index for a post.

    Input:
        text — a single merged post string

    Output:
        dict with 2 keys:
        - flesch_kincaid_grade : US school grade required (typically 0-20+)
        - gunning_fog_index    : Gunning Fog score (typically 0-20+)

    Scales:
        Both metrics produce scores roughly in the 0-20 range:
        - 5  = very simple (5th grade reading level)
        - 10 = moderate (10th grade level)
        - 15 = complex (college level)
        - 20 = very complex (graduate level)

        Real Reddit posts typically fall in the 5-12 range.
        Scores can occasionally exceed 20 for extremely long-sentence posts.

    Design decisions:
        - Why return 0 for empty or very short text?
          textstat can produce nonsensical values (negative, or NaN) for
          texts with fewer than 2 sentences or no polysyllabic words.
          Returning 0 gives the classifier a clean "no complexity signal"
          indication rather than corrupted data.

        - Why not clamp to a maximum like 25?
          Extreme scores are themselves signal. A post with Grade 50
          (a single 300-word sentence with many complex terms) is
          genuinely unusual and the model should see that. We preserve
          the raw score.

        - Why round to 4 decimals?
          Textstat scores have natural precision around 0.01-0.1.
          More decimals than that are just floating-point noise.
    """
    empty_result = {
        "flesch_kincaid_grade": 0.0,
        "gunning_fog_index"   : 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    # textstat requires at least some minimal content to compute scores.
    # For extremely short texts (< 10 words), formulas produce unreliable
    # or undefined values. We check length and skip if too short.
    word_count = len(text.split())
    if word_count < 10:
        return empty_result

    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
        fog      = textstat.gunning_fog(text)

        # textstat can return NaN or negative values on edge cases.
        # Negative Flesch-Kincaid grade is technically possible mathematically
        # for very simple text but is not meaningful — floor to 0.
        if not isinstance(fk_grade, (int, float)) or fk_grade != fk_grade:
            # NaN check: NaN != NaN is the canonical way to detect NaN
            fk_grade = 0.0
        else:
            fk_grade = max(0.0, float(fk_grade))

        if not isinstance(fog, (int, float)) or fog != fog:
            fog = 0.0
        else:
            fog = max(0.0, float(fog))

        return {
            "flesch_kincaid_grade": round(fk_grade, 4),
            "gunning_fog_index"   : round(fog,      4),
        }

    except Exception as e:
        logger.debug(f"textstat failed on text snippet: {str(text)[:50]} | {e}")
        return empty_result


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract readability features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/stylistic/readability_{split}.csv
        Shape: (num_samples, 2)

    Runtime:
        textstat is pure Python but very fast — it uses regex-based
        syllable counting and simple arithmetic. Expect ~1-2 minutes
        per split for 13k posts on any machine.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Readability ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_readability(text))
    )

    save_dir  = STYLISTIC_FEATURES_DIR / "readability"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"readability_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_readability_pipeline() -> None:
    """
    Extract and save readability features for all three splits.
    Output: 2-dimensional feature vector per sample
            [flesch_kincaid_grade, gunning_fog_index].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting readability feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Readability extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_readability_pipeline()