# scripts/features/structural/tense_distribution.py

import logging
import pandas as pd
import spacy
from tqdm import tqdm

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    STRUCTURAL_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SPACY MODEL
# =============================================================================
# We use en_core_web_sm — same model as psycholinguistic.py.
# We enable the parser here because we need dependency information
# to detect future tense constructions (e.g. "will go", "going to go").
# POS tagger and morphology analyzer are enabled by default.

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy English model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# =============================================================================
# FUTURE TENSE DETECTION
# =============================================================================
# English future tense is not a morphological tense like past/present.
# It is constructed with auxiliaries. We detect two main patterns:
#
#   Pattern 1: MODAL + VERB      → "will go", "shall see", "would do"
#   Pattern 2: BE + going + to   → "I am going to leave"
#
# Modal verbs that indicate future in English:

FUTURE_MODALS = frozenset({
    "will", "shall", "'ll"
})

# Note: "would", "could", "might", "should" can sometimes indicate future
# but more often indicate hypothetical/conditional meaning — including them
# would inflate the future count for hedging language (which we capture
# separately in the stylistic group's hedging feature). We keep only
# clear future markers.


# =============================================================================
# TENSE COUNTING
# =============================================================================

def count_verb_tenses(doc) -> dict:
    """
    Count verbs in each tense category for a spaCy-parsed document.

    Input:
        doc — a spaCy Doc object

    Output:
        dict with 3 integer counts:
        - past
        - present
        - future

    Algorithm:
        For each verb token:
          - Check for future construction (modal + verb, or "going to" + verb)
          - Otherwise check spaCy's morphological tense: Past or Pres
          - Skip verbs where tense cannot be determined

    Why check future before past/present?
    "I will go" — the main verb "go" has no morphological tense (it's base form),
    but the modal "will" indicates future. If we looked only at the main verb,
    we would miss this. By scanning for future constructions first and marking
    them, we correctly classify future sentences.
    """
    past_count    = 0
    present_count = 0
    future_count  = 0

    # Track token indices already counted as future to avoid double-counting
    # (the main verb following a modal should not be re-counted as present).
    future_indices = set()

    # First pass: find future constructions
    for token in doc:
        # Pattern 1: modal verb like "will" or "'ll"
        if token.lemma_.lower() in FUTURE_MODALS:
            future_count += 1
            future_indices.add(token.i)
            # Also mark the verb it modifies as future-context
            for child in token.children:
                if child.pos_ == "VERB":
                    future_indices.add(child.i)
            # And in spaCy's parse, the modal is often a child of the main verb
            if token.head.pos_ == "VERB":
                future_indices.add(token.head.i)
            continue

        # Pattern 2: "going to" construction
        # "I am going to leave" — "going" is a verb with an xcomp child "leave"
        if (token.lemma_.lower() == "go"
                and token.tag_ == "VBG"           # -ing form
                and any(child.dep_ == "aux" and child.lemma_.lower() == "be"
                        for child in token.children)):
            # Check for "to" following
            for child in token.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    future_count += 1
                    future_indices.add(token.i)
                    future_indices.add(child.i)

    # Second pass: count past/present for all verbs not marked as future
    for token in doc:
        # Only look at verbs (main verbs and auxiliaries)
        if token.pos_ not in ("VERB", "AUX"):
            continue

        # Skip if already counted as part of a future construction
        if token.i in future_indices:
            continue

        # Get morphological tense from spaCy
        tense = token.morph.get("Tense")

        if not tense:
            # spaCy couldn't determine tense — common for base forms
            # in infinitive constructions ("I want to go"). Skip these.
            continue

        # tense is a list like ["Past"] or ["Pres"] — take the first value
        tense_value = tense[0]

        if tense_value == "Past":
            past_count += 1
        elif tense_value == "Pres":
            present_count += 1

    return {
        "past"   : past_count,
        "present": present_count,
        "future" : future_count,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_tense_distribution(text: str) -> dict:
    """
    Compute the proportion of verbs in each tense for a single post.

    Input:
        text — a single merged post string

    Output:
        dict with 3 keys, each a float ratio in [0, 1]:
        - past_ratio    : past-tense verbs / total tensed verbs
        - present_ratio : present-tense verbs / total tensed verbs
        - future_ratio  : future-construction verbs / total tensed verbs

    The three ratios sum to 1.0 (or to 0.0 for empty/verbless posts).

    Why ratios instead of raw counts?
    Raw counts are dominated by post length — a 500-word post will have
    many more past-tense verbs than a 50-word post, even if the temporal
    orientation is the same. Ratios normalize for length, making posts
    of different sizes directly comparable.

    Why divide by total TENSED verbs, not total verbs?
    Some verbs have no morphological tense (infinitives like "to go",
    participles in certain contexts). Including them in the denominator
    would artificially reduce all three ratios. Dividing only by the
    number of verbs we successfully classified gives accurate proportions.
    """
    empty_result = {
        "past_ratio"   : 0.0,
        "present_ratio": 0.0,
        "future_ratio" : 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    # Parse the text with spaCy
    # This is the most expensive step — consider caching if running multiple
    # tense-related features. For this project we run it once per post.
    try:
        doc = nlp(text)
    except Exception as e:
        logger.debug(f"spaCy parse failed: {e}")
        return empty_result

    # Count verbs in each tense
    counts = count_verb_tenses(doc)

    total = counts["past"] + counts["present"] + counts["future"]

    if total == 0:
        # No tensed verbs found — return zeros
        return empty_result

    return {
        "past_ratio"   : round(counts["past"]    / total, 6),
        "present_ratio": round(counts["present"] / total, 6),
        "future_ratio" : round(counts["future"]  / total, 6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract tense distribution features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/structural/tense_distribution_{split}.csv
        Shape: (num_samples, 3)

    Runtime:
        spaCy parsing of 13k posts with full morphology and dependencies
        enabled takes ~10-15 minutes on CPU per split.
        spaCy does not meaningfully benefit from GPU for en_core_web_sm.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Tense distribution ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_tense_distribution(text))
    )

    save_dir  = STRUCTURAL_FEATURES_DIR / "tense_distribution"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"tense_distribution_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_tense_distribution_pipeline() -> None:
    """
    Extract and save tense distribution features for all three splits.
    Output: 3-dimensional feature vector per sample
            [past_ratio, present_ratio, future_ratio].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting tense distribution feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Tense distribution extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_tense_distribution_pipeline()