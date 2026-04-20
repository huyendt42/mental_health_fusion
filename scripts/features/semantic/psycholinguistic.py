# scripts/features/semantic/psycholinguistic.py

import logging
import pandas as pd
import spacy
from empath import Empath
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
# LOAD SPACY MODEL
# =============================================================================

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise OSError(
        "spaCy English model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# =============================================================================
# BUILD WORD LISTS FROM FREE VALIDATED SOURCES
# =============================================================================

def build_word_lists() -> dict[str, frozenset]:
    """
    Build all word lists from free, published, citable sources.

    Sources:
        - Grammar              : first-person pronouns
        - Al-Mosaiwi &
          Johnstone (2018)     : absolutist words — validated on Reddit
                                 mental health data specifically
        - VADER / Hutto &
          Gilbert (2014)       : negation words — extracted from VADER's
                                 internal NEGATE constant
        - Empath / Fast
          et al. (2016)        : cognitive process words — from Empath's
                                 cognitive_processes category

    Returns:
        dict mapping feature name → frozenset of words
    """

    # -------------------------------------------------------------------------
    # 1. First-person pronouns — grammatically complete
    #    Fixed by English grammar, no external source needed.
    # -------------------------------------------------------------------------
    first_singular = frozenset({
        "i", "me", "my", "myself", "mine"
    })

    first_plural = frozenset({
        "we", "us", "our", "ourselves", "ours"
    })

    # -------------------------------------------------------------------------
    # 2. Absolutist words — Al-Mosaiwi & Johnstone (2018)
    #    "In an Absolute State: Elevated Use of Absolutist Words Is a Marker
    #    Specific to Anxiety, Depression, and Suicidal Ideation."
    #    Clinical Psychological Science, 6(2), 272-280.
    #    Exact word list from the paper, validated on Reddit mental health
    #    forums — directly applicable to this dataset.
    # -------------------------------------------------------------------------
    absolutist = frozenset({
        "absolutely", "always", "complete", "completely", "constant",
        "constantly", "entire", "entirely", "ever", "every", "everyone",
        "everything", "full", "fully", "impossible", "must", "never",
        "nothing", "totally", "whole"
    })

    # -------------------------------------------------------------------------
    # 3. Negation words — VADER (Hutto & Gilbert, 2014)
    #    VADER maintains an internal NEGATE constant — a list of words that
    #    flip sentiment polarity (e.g. "not good" → negative).
    #    We load it directly from the library rather than maintaining
    #    a manual list. This is citable and handles contractions that
    #    manual lists often miss.
    # -------------------------------------------------------------------------
    try:
        from vaderSentiment.vaderSentiment import NEGATE
        negation = frozenset(NEGATE)
        logger.info(f"Loaded {len(negation)} negation words from VADER.")
    except ImportError:
        logger.warning(
            "Could not import VADER NEGATE list. "
            "Using manual fallback negation list."
        )
        negation = frozenset({
            "no", "not", "never", "neither", "nor", "nobody", "nothing",
            "nowhere", "hardly", "scarcely", "barely", "without",
            "cannot", "cant", "wont", "dont", "didnt", "doesnt",
            "isnt", "arent", "wasnt", "werent", "havent", "hasnt",
            "hadnt", "wouldnt", "shouldnt", "couldnt", "mightnt", "mustnt"
        })

    # -------------------------------------------------------------------------
    # 4. Cognitive process words — Empath (Fast et al., 2016)
    #    Empath is a free neural embedding-based lexicon with 200 semantic
    #    categories. Published at Stanford CHI 2016, freely available via pip.
    #    The "cognitive_processes" category maps directly to LIWC's cognitive
    #    mechanisms category used in mental health NLP research.
    #
    #    Citation: Fast, E., Chen, B., & Bernstein, M. S. (2016).
    #    Empath: Understanding topic signals in large-scale text. CHI 2016.
    # -------------------------------------------------------------------------
    empath       = Empath()
    cognitive_raw = empath.cats.get("cognitive_processes", {})

    if cognitive_raw:
        cognitive = frozenset(cognitive_raw.keys())
        logger.info(f"Loaded {len(cognitive)} cognitive words from Empath.")
    else:
        logger.warning(
            "Empath 'cognitive_processes' category not found. "
            "Using manual fallback cognitive list."
        )
        cognitive = frozenset({
            "think", "know", "consider", "because", "wonder", "realize",
            "understand", "believe", "reason", "cause", "if", "maybe",
            "perhaps", "possibly", "could", "should", "would", "seems",
            "suppose", "assume", "imagine", "expect", "doubt", "certain",
            "uncertain", "guess", "reckon", "reflect", "analyze"
        })

    return {
        "first_singular": first_singular,
        "first_plural"  : first_plural,
        "absolutist"    : absolutist,
        "negation"      : negation,
        "cognitive"     : cognitive,
    }


# Build word lists once at module load time.
# Empath takes ~2 seconds to initialize — doing this inside the per-row
# function would add hours to runtime across 13k+ samples.
logger.info("Building word lists from validated sources...")
WORD_LISTS = build_word_lists()
logger.info(
    f"Word lists ready — "
    f"absolutist: {len(WORD_LISTS['absolutist'])}, "
    f"negation: {len(WORD_LISTS['negation'])}, "
    f"cognitive: {len(WORD_LISTS['cognitive'])}"
)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_psycholinguistic_ratios(text: str) -> dict:
    """
    Compute 5 psycholinguistic word-category ratios for a single text.

    Input:
        text — a single merged post string

    Output:
        dict with 5 keys, each a float ratio in [0, 1]:
        - first_person_singular_rate
        - first_person_plural_rate
        - negation_rate
        - absolutist_rate
        - cognitive_rate

    Ratio formula:
        rate = matching token count / total token count

    Punctuation and whitespace tokens are excluded from both
    the numerator and denominator — they inflate token count
    without contributing to word-category signal.
    """
    empty_result = {
        "first_person_singular_rate": 0.0,
        "first_person_plural_rate"  : 0.0,
        "negation_rate"             : 0.0,
        "absolutist_rate"           : 0.0,
        "cognitive_rate"            : 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    doc = nlp(text)

    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    total = len(tokens)
    if total == 0:
        return empty_result

    return {
        "first_person_singular_rate": sum(1 for t in tokens if t in WORD_LISTS["first_singular"]) / total,
        "first_person_plural_rate"  : sum(1 for t in tokens if t in WORD_LISTS["first_plural"])   / total,
        "negation_rate"             : sum(1 for t in tokens if t in WORD_LISTS["negation"])        / total,
        "absolutist_rate"           : sum(1 for t in tokens if t in WORD_LISTS["absolutist"])      / total,
        "cognitive_rate"            : sum(1 for t in tokens if t in WORD_LISTS["cognitive"])       / total,
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract psycholinguistic features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/semantic/psycholinguistic_{split}.csv
        Shape: (num_samples, 5)
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    features = df[TEXT_COL].apply(
        lambda text: pd.Series(compute_psycholinguistic_ratios(text))
    )

    save_dir  = SEMANTIC_FEATURES_DIR / "psycholinguistic"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"psycholinguistic_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_psycholinguistic_pipeline() -> None:
    """
    Extract and save psycholinguistic features for all three splits.
    Output: 5-dimensional feature vector per sample.
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting psycholinguistic feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Psycholinguistic extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_psycholinguistic_pipeline()