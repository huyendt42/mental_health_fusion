# scripts/data/preprocessing.py

import re
import logging
import pandas as pd
from pathlib import Path

from scripts.config import (
    RAW_TRAIN_PATH, RAW_VAL_PATH, RAW_TEST_PATH,  # Where to read from
    TRAIN_PATH, VAL_PATH, TEST_PATH,               # Where to write to
    RAW_TITLE_COL, RAW_POST_COL,                   # Input column names
    TEXT_COL, LABEL_COL, CLASS_NAME_COL,           # Output column names
    MERGE_SEPARATOR,                               # ": " between title and post
    PROCESSED_DIR,                                 # Folder to create if needed
    SEED                                           # For reproducibility
)

# =============================================================================
# LOGGING SETUP
# =============================================================================
# logging is Python's built-in way to record what your program is doing.
# Unlike print(), logging lets you control the level of detail (DEBUG, INFO,
# WARNING, ERROR) and write to a file as well as the console.
# For a research project, having a log of what preprocessing did is valuable
# when you need to report it in your thesis or debug an issue weeks later.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
# __name__ gives this logger the name of this module (preprocessing),
# so in the log output you can see exactly which file produced each message.


# =============================================================================
# STEP 1 — LOAD
# =============================================================================

def load_raw_splits() -> dict[str, pd.DataFrame]:
    """
    Load the three raw CSV splits into a dictionary of DataFrames.

    Input:  None — paths are read from config.py
    Output: dict with keys "train", "val", "test", values are DataFrames

    Why a dict? So every downstream function can loop over splits cleanly
    instead of handling three separate variables.
    """
    splits = {
        "train" : RAW_TRAIN_PATH,
        "val"   : RAW_VAL_PATH,
        "test"  : RAW_TEST_PATH,
    }

    datasets = {}
    for name, path in splits.items():
        path = Path(path)

        if not path.exists():
            # Raise immediately with a clear message rather than letting
            # pandas raise a confusing FileNotFoundError later.
            raise FileNotFoundError(
                f"Raw data file not found: {path}\n"
                f"Make sure your CSVs are placed in data/original/"
            )

        df = pd.read_csv(path)
        datasets[name] = df
        logger.info(f"Loaded {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    return datasets


# =============================================================================
# STEP 2 — VALIDATE
# =============================================================================

def validate_columns(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Check that every split contains the columns the pipeline depends on.

    Input:  datasets dict from load_raw_splits()
    Output: None — raises ValueError immediately if validation fails

    Why validate? A missing or misspelled column name causes a KeyError
    deep inside your feature extraction code, far from where the actual
    problem is. Catching it here gives you a clear, actionable error message.
    """
    required = {RAW_TITLE_COL, RAW_POST_COL, LABEL_COL, CLASS_NAME_COL}

    for name, df in datasets.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Split '{name}' is missing columns: {missing}\n"
                f"Columns found: {list(df.columns)}"
            )

    logger.info("Column validation passed for all splits.")


# =============================================================================
# STEP 3 — HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Handle null values in title and post columns.

    Input:  datasets dict (raw, after validation)
    Output: datasets dict with nulls filled

    Why fill with empty string rather than dropping?
    Some posts have a title but no body — the title alone still carries
    linguistic signal. Dropping the entire row loses that information.
    Filling with "" means the merge step produces "Title: " which is
    still meaningful. We log exactly how many nulls were filled so this
    is transparent and reportable in your thesis.
    """
    for name, df in datasets.items():
        title_nulls = df[RAW_TITLE_COL].isnull().sum()
        post_nulls  = df[RAW_POST_COL].isnull().sum()

        if title_nulls > 0 or post_nulls > 0:
            logger.info(
                f"{name} — filling nulls: "
                f"{title_nulls} in title, {post_nulls} in post"
            )

        # Fill nulls with empty string before any text operations.
        df[RAW_TITLE_COL] = df[RAW_TITLE_COL].fillna("")
        df[RAW_POST_COL]  = df[RAW_POST_COL].fillna("")

    return datasets


# =============================================================================
# STEP 4 — MERGE TITLE AND POST
# =============================================================================

def merge_title_and_post(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Concatenate title and post into a single text column.

    Input:  datasets dict (after handling missing values)
    Output: datasets dict with a new TEXT_COL column added

    Why merge? RoBERTa and most feature extractors expect a single text input.
    Research (Murarka et al., 2021) shows that merging title + body gives
    richer semantic context and better F1 scores than using either alone.
    The separator ": " is defined in config.py so it can be changed in
    one place if needed.
    """
    for name, df in datasets.items():
        df[TEXT_COL] = df[RAW_TITLE_COL] + MERGE_SEPARATOR + df[RAW_POST_COL]
        logger.info(f"{name} — merged title + post into '{TEXT_COL}' column")

    return datasets


# =============================================================================
# STEP 5 — CLEAN TEXT
# =============================================================================

def clean_text(text: str) -> str:
    """
    Apply targeted cleaning to a single text string.

    Input:  raw text string
    Output: cleaned text string

    Each cleaning decision is deliberate:
    - URLs removed:     carry no linguistic signal relevant to mental health
    - Extra whitespace: normalized so tokenizers don't see double spaces
    - Numbers kept:     "I haven't slept in 3 days" — the number matters
    - Punctuation kept: "...", "?", "!" carry emotional signal in this domain
    - Case kept:        UPPERCASE can signal emotional intensity
    - No stopword removal: we want psycholinguistic ratios to be accurate
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs (http/https and bare www. links)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Normalize whitespace — collapse multiple spaces/tabs/newlines into one space
    # This is important because Reddit posts often have irregular formatting
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def apply_cleaning(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply clean_text() to the merged TEXT_COL across all splits.

    Input:  datasets dict (after merging title and post)
    Output: datasets dict with TEXT_COL cleaned

    Why separate clean_text() from apply_cleaning()?
    clean_text() operates on a single string — it is pure and testable.
    apply_cleaning() handles the DataFrame logic of applying it across rows.
    This separation means you can test clean_text() independently and reuse
    it anywhere a single string needs cleaning.
    """
    for name, df in datasets.items():
        # .apply() runs clean_text on every row of the TEXT_COL column.
        # This is the standard pandas way to apply a function row-by-row.
        df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
        logger.info(f"{name} — text cleaning applied")

    return datasets


# =============================================================================
# STEP 6 — SELECT AND SAVE
# =============================================================================

def select_columns(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Keep only the columns needed by the rest of the pipeline.

    Input:  datasets dict (fully processed)
    Output: datasets dict with only [TEXT_COL, LABEL_COL, CLASS_NAME_COL]

    Why drop the rest? The raw data has columns like 'id', 'title', 'post'
    that are no longer needed. Keeping only what downstream scripts use
    makes the processed files smaller and prevents accidental reliance
    on raw columns that won't be available at inference time.
    """
    keep = [TEXT_COL, LABEL_COL, CLASS_NAME_COL]

    for name, df in datasets.items():
        datasets[name] = df[keep]
        logger.info(f"{name} — kept columns: {keep}")

    return datasets


def save_processed(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Save the processed DataFrames to data/processed/.

    Input:  datasets dict (fully processed, columns selected)
    Output: None — writes three CSV files to disk

    Why index=False? The DataFrame index is just row numbers (0, 1, 2...).
    Saving it creates an unnamed column in the CSV that causes confusion
    when the file is loaded elsewhere.
    """
    # Create the processed directory if it doesn't exist yet.
    # exist_ok=True means no error if the folder already exists.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train" : TRAIN_PATH,
        "val"   : VAL_PATH,
        "test"  : TEST_PATH,
    }

    for name, df in datasets.items():
        path = output_paths[name]
        df.to_csv(path, index=False)
        logger.info(f"Saved {name} → {path} ({df.shape[0]:,} rows)")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def preprocess() -> None:
    """
    Run the full preprocessing pipeline in order.

    This is the function that main.py will call.
    Each step is clearly separated so you can see the exact sequence
    and add or remove steps without touching anything else.
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    datasets = load_raw_splits()
    validate_columns(datasets)
    datasets = handle_missing_values(datasets)
    datasets = merge_title_and_post(datasets)
    datasets = apply_cleaning(datasets)
    datasets = select_columns(datasets)
    save_processed(datasets)

    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    preprocess()