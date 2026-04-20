# scripts/features/stylistic/syntactic.py

import logging
import pandas as pd
import spacy
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
# SPACY MODEL
# =============================================================================
# We enable the dependency parser because we compute dependency distance.
# POS tagging and morphology are already included by default.

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy English model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_syntactic_features(text: str) -> dict:
    """
    Compute 6 syntactic features for a single post.

    Input:
        text — a single merged post string

    Output:
        dict with 6 keys, each a float:
        - noun_ratio        : proportion of NOUN tokens in [0, 1]
        - verb_ratio        : proportion of VERB tokens in [0, 1]
        - adj_ratio         : proportion of ADJ  tokens in [0, 1]
        - adv_ratio         : proportion of ADV  tokens in [0, 1]
        - pronoun_ratio     : proportion of PRON tokens in [0, 1]
        - avg_dep_distance  : mean dependency distance across tokens, >= 0

    Algorithm:
        1. Parse text with spaCy (POS tagging + dependency parsing)
        2. Count tokens by POS category, divide by total non-punctuation tokens
        3. For each token, compute |token.i - token.head.i| (dependency
           distance), then average across all tokens

    Design decisions:
        - Why exclude punctuation from total token count?
          spaCy treats punctuation as tokens. Including them in the
          denominator would deflate all ratios proportional to how much
          punctuation a writer uses. Since POS ratios should measure
          word usage, not punctuation density, we exclude punctuation.

        - Why NOT exclude punctuation from dependency distance?
          Punctuation tokens have dependency relations too (e.g., commas
          link to the sentence head). Removing them would create gaps
          in the token indices, distorting the distance calculation.
          Keeping them gives accurate dependency distances.

        - Why use token.pos_ (universal POS) instead of token.tag_ (fine)?
          spaCy provides two POS systems: universal (NOUN, VERB, ADJ, ADV,
          PRON) and Penn Treebank (NNS, VBZ, JJ, RB, PRP). Universal POS
          is coarser but more interpretable and sufficient for our ratio
          features. The fine-grained Penn tags would produce too many
          sparse features.
    """
    empty_result = {
        "noun_ratio"      : 0.0,
        "verb_ratio"      : 0.0,
        "adj_ratio"       : 0.0,
        "adv_ratio"       : 0.0,
        "pronoun_ratio"   : 0.0,
        "avg_dep_distance": 0.0,
    }

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    try:
        doc = nlp(text)
    except Exception as e:
        logger.debug(f"spaCy parse failed: {e}")
        return empty_result

    # Count tokens by POS tag, excluding punctuation and whitespace.
    # We track counts in a dict so adding/removing POS categories later
    # requires changing only this dict, not the rest of the function.
    pos_counts = {
        "NOUN": 0,
        "VERB": 0,
        "ADJ" : 0,
        "ADV" : 0,
        "PRON": 0,
    }

    total_word_tokens = 0
    dep_distances     = []

    for token in doc:
        # Skip whitespace tokens entirely — they're artifacts of tokenization
        if token.is_space:
            continue

        # Dependency distance: measured in token indices.
        # token.head is the grammatical head of this token.
        # token.i is this token's position in the doc.
        # Absolute difference gives the distance in tokens.
        # For the root token, token.head == token, giving distance 0.
        dep_distances.append(abs(token.i - token.head.i))

        # POS ratios: count only real word tokens (not punctuation)
        if token.is_punct:
            continue

        total_word_tokens += 1

        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

    if total_word_tokens == 0:
        return empty_result

    # Compute ratios
    ratios = {
        f"{pos.lower().replace('pron', 'pronoun')}_ratio":
            round(count / total_word_tokens, 6)
        for pos, count in pos_counts.items()
    }
    # The dict comprehension above produces keys like "noun_ratio",
    # "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio".
    # The replace() call handles the naming mismatch: spaCy uses "PRON"
    # but we want the output key to be "pronoun_ratio" for clarity.

    # Mean dependency distance across all tokens (including punctuation).
    # We use the full set of tokens here because dependency structure
    # includes all tokens — punctuation genuinely participates in parse trees.
    avg_dep_distance = (
        round(sum(dep_distances) / len(dep_distances), 6)
        if dep_distances else 0.0
    )

    return {
        **ratios,
        "avg_dep_distance": avg_dep_distance,
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract syntactic features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/stylistic/syntactic_{split}.csv
        Shape: (num_samples, 6)

    Runtime:
        spaCy full parsing (POS + dependencies + morphology) is the slowest
        non-transformer feature extractor. Expect ~10-15 minutes per split
        on CPU for 13k posts. No meaningful GPU benefit for en_core_web_sm.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Syntactic ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_syntactic_features(text))
    )

    save_dir  = STYLISTIC_FEATURES_DIR / "syntactic"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"syntactic_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_syntactic_pipeline() -> None:
    """
    Extract and save syntactic features for all three splits.
    Output: 6-dimensional feature vector per sample
            [noun_ratio, verb_ratio, adj_ratio, adv_ratio, pronoun_ratio,
             avg_dep_distance].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting syntactic feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Syntactic extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_syntactic_pipeline()