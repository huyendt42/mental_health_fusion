# scripts/features/stylistic/hedging.py

import logging
import pandas as pd
import spacy
import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk

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
# NLTK SETUP
# =============================================================================

try:
    sent_tokenize("Test.")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)


# =============================================================================
# SPACY MODEL
# =============================================================================
# We need tokenization and POS tagging here. Disabling parser and ner
# speeds up spaCy since we don't use syntactic structure in this file.

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise OSError(
        "spaCy English model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# =============================================================================
# WORD LISTS
# =============================================================================
# Modal verbs — closed grammatical class. These are the canonical English
# modals that express possibility, obligation, or uncertainty.
# We deliberately include ALL modals here — not just "future" modals — because
# the goal is to measure hedging/uncertainty in general, not temporal orientation
# (which is captured separately in tense_distribution.py).

MODAL_VERBS = frozenset({
    "might", "could", "would", "may", "should", "can",
    "must", "shall", "ought"
})

# Hedge words and phrases — drawn from Hyland (2005) on academic hedging.
# These express epistemic uncertainty — the writer signaling they are
# not fully committed to the proposition.

HEDGE_WORDS = frozenset({
    "maybe", "perhaps", "possibly", "probably", "presumably",
    "apparently", "seemingly", "allegedly", "supposedly",
    "somewhat", "somehow", "somewhere",
    "sort",       # from "sort of"
    "kind",       # from "kind of"
    "quite",
})

# Multi-word hedges — checked separately using regex since they span
# multiple tokens. Listed as regex patterns with word boundaries.

MULTIWORD_HEDGE_PATTERNS = [
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\ba\s+bit\b",
    r"\bi\s+guess\b",
    r"\bi\s+suppose\b",
    r"\bi\s+think\b",      # hedge when used to qualify ("I think it's fine")
    r"\bi\s+mean\b",       # filler hedge
    r"\byou\s+know\b",     # filler hedge
]

# Compile once for speed — regex compilation is expensive.
# re.IGNORECASE makes matches case-insensitive without needing .lower()
# which would also slow down the main loop.
MULTIWORD_HEDGE_RE = [
    re.compile(pattern, re.IGNORECASE) for pattern in MULTIWORD_HEDGE_PATTERNS
]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_hedging_features(text: str) -> dict:
    """
    Compute 4 hedging/uncertainty features for a single post.

    Input:
        text — a single merged post string

    Output:
        dict with 4 keys:
        - modal_rate    : modal verbs / total word tokens, in [0, 1]
        - hedge_rate    : hedge words+phrases / total word tokens, in [0, 1]
        - question_rate : '?' count / total sentences, >= 0
        - ellipsis_rate : '...' or '…' count / total sentences, >= 0

    Algorithm:
        1. Tokenize with spaCy for modal + hedge word counts
        2. Scan raw text for multi-word hedges using regex
        3. Count '?' and '...' in raw text, divide by sentence count

    Design decisions:
        - Why two different denominators (tokens vs sentences)?
          Modal verbs and hedge words are word-level — their rate per
          total words is the meaningful metric.
          Question marks and ellipses are sentence-level — a post with
          "??? what? really???" has 4 question marks across 3 sentences
          which is more meaningful than 4/100 tokens. Dividing by sentences
          gives the intuitive "questions per sentence" rate.

        - Why include "I think" as a hedge?
          In conversational context, "I think" functions as a hedge
          ("I think it's okay" = "It's okay, but I'm not sure").
          This is consistent with Hyland (2005). It can occasionally
          be used literally but the hedge reading dominates in Reddit text.

        - Why count "..." AND "…"?
          Both are used in informal text. "…" is the Unicode ellipsis
          character, "..." is three periods. Some users type one, some
          type the other. Counting only one would miss a real signal.
    """
    empty_result = {
        "modal_rate"   : 0.0,
        "hedge_rate"   : 0.0,
        "question_rate": 0.0,
        "ellipsis_rate": 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    # -------------------------------------------------------------------------
    # Step 1: Tokenize for word-level counts
    # -------------------------------------------------------------------------
    try:
        doc = nlp(text)
    except Exception as e:
        logger.debug(f"spaCy parse failed: {e}")
        return empty_result

    # Extract lowercase word tokens (skip punctuation and whitespace)
    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    total_tokens = len(tokens)

    if total_tokens == 0:
        return empty_result

    # -------------------------------------------------------------------------
    # Step 2: Count modals and single-word hedges
    # -------------------------------------------------------------------------
    modal_count = sum(1 for t in tokens if t in MODAL_VERBS)
    hedge_count = sum(1 for t in tokens if t in HEDGE_WORDS)

    # -------------------------------------------------------------------------
    # Step 3: Count multi-word hedges via regex on raw text
    # -------------------------------------------------------------------------
    # findall returns a list of all non-overlapping matches, len() gives count
    multiword_count = sum(
        len(pattern.findall(text)) for pattern in MULTIWORD_HEDGE_RE
    )

    # Combine single-word and multi-word hedges
    hedge_count += multiword_count

    # -------------------------------------------------------------------------
    # Step 4: Count questions and ellipses via sentence-level analysis
    # -------------------------------------------------------------------------
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]   # Fallback: treat entire text as one sentence

    num_sentences = max(1, len(sentences))   # Avoid division by zero

    # Count question marks — simple character count across whole text
    question_count = text.count("?")

    # Count ellipses — both Unicode and three-period variants
    ellipsis_count = text.count("...") + text.count("…")

    # -------------------------------------------------------------------------
    # Step 5: Compute rates
    # -------------------------------------------------------------------------
    return {
        "modal_rate"   : round(modal_count    / total_tokens,  6),
        "hedge_rate"   : round(hedge_count    / total_tokens,  6),
        "question_rate": round(question_count / num_sentences, 6),
        "ellipsis_rate": round(ellipsis_count / num_sentences, 6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract hedging features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/stylistic/hedging_{split}.csv
        Shape: (num_samples, 4)

    Runtime:
        Fast — ~5-8 minutes per split for 13k posts.
        spaCy tokenization is the main cost. Regex and string counting
        are essentially free.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Hedging ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_hedging_features(text))
    )

    save_dir  = STYLISTIC_FEATURES_DIR / "hedging"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"hedging_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_hedging_pipeline() -> None:
    """
    Extract and save hedging features for all three splits.
    Output: 4-dimensional feature vector per sample
            [modal_rate, hedge_rate, question_rate, ellipsis_rate].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting hedging feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Hedging extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_hedging_pipeline()