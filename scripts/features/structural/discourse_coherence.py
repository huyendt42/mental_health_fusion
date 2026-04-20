# scripts/features/structural/discourse_coherence.py

import logging
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
# NLTK SETUP
# =============================================================================

try:
    sent_tokenize("Test.")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)


# =============================================================================
# MODEL SETUP
# =============================================================================
# all-MiniLM-L6-v2 is a lightweight sentence embedding model from
# sentence-transformers. It maps sentences to 384-dimensional vectors.
# - ~80MB download (cached locally after first run)
# - No login needed — public on HuggingFace
# - Optimized for fast sentence similarity computation
# - Produces meaningful embeddings even for short sentences

SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_sentence_model() -> SentenceTransformer:
    """
    Load the sentence embedding model.

    Returns:
        Loaded SentenceTransformer model, moved to GPU if available.

    Why GPU matters here:
        We embed potentially 100+ sentences per post across 16,703 posts.
        On CPU this is slow. On GPU this is ~10x faster.
        The model is small enough (~80MB) to fit alongside other models
        without memory pressure.
    """
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = "GPU" if device == "cuda" else "CPU"

    logger.info(f"Loading {SENTENCE_MODEL_NAME} on {device_label}...")

    model = SentenceTransformer(SENTENCE_MODEL_NAME, device=device)
    logger.info("Sentence model ready.")
    return model


# Load model once at module level.
# SentenceTransformer loading takes ~3 seconds — loading it inside the
# per-row function would add ~14 hours across 16,703 samples.
SENTENCE_MODEL = load_sentence_model()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def compute_discourse_coherence(text: str) -> dict:
    """
    Compute discourse coherence statistics for a single post.

    Input:
        text — a single merged post string

    Output:
        dict with 2 keys:
        - mean_coherence : average cosine similarity between adjacent
                           sentence pairs, in [-1, 1]
        - std_coherence  : standard deviation of adjacent similarities, >= 0

    Algorithm:
        1. Split post into sentences using NLTK
        2. Embed each sentence using the sentence transformer (384-dim)
        3. Compute cosine similarity between each consecutive pair
        4. Summarize with mean and std

    Design decisions:
        - Why adjacent pairs only instead of all pairs? Adjacent similarity
          measures discourse FLOW — whether each sentence follows naturally
          from the previous one. All-pairs similarity would measure topic
          DIVERSITY which is different. For mental health research, flow is
          the more clinically meaningful dimension (Elvevåg et al., 2007).

        - Why lowercase before embedding? We don't — the sentence
          transformer is trained on cased text. Lowercasing would degrade
          embedding quality. Unlike word-based features, transformer
          features benefit from preserving case.

        - Why return 0 for posts with 1 or 0 sentences? Coherence requires
          at least 2 sentences to compute any similarity. A single-sentence
          post has no "flow" to measure. Returning 0 (indicating "no
          measurable coherence signal") is correct — not positive coherence,
          not negative, just absent.
    """
    # Default values for edge cases
    empty_result = {"mean_coherence": 0.0, "std_coherence": 0.0}

    if not isinstance(text, str) or text.strip() == "":
        return empty_result

    # Split into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception:
        return empty_result

    # Filter out empty or whitespace-only sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Need at least 2 sentences to compute any pair similarity
    if len(sentences) < 2:
        return empty_result

    # Embed all sentences at once — batching is more GPU-efficient
    # than calling the model per sentence.
    # embeddings shape: (num_sentences, 384)
    embeddings = SENTENCE_MODEL.encode(
        sentences,
        batch_size=32,          # Process up to 32 sentences together
        show_progress_bar=False, # Don't clutter logs — we have tqdm at post level
        convert_to_numpy=True,   # Return numpy arrays, not torch tensors
    )

    # Compute cosine similarity for each adjacent pair.
    # For sentences [s1, s2, s3, s4]:
    #   pairs: (s1,s2), (s2,s3), (s3,s4)
    # This gives num_sentences - 1 similarity values.
    similarities = []
    for i in range(len(embeddings) - 1):
        # cosine_similarity expects 2D arrays, so reshape with [None, :]
        # Result is a 1x1 matrix; extract the scalar with [0][0]
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(float(sim))

    # Compute summary statistics
    sims_array = np.array(similarities)
    mean_sim   = float(sims_array.mean())
    std_sim    = float(sims_array.std(ddof=0))  # Population std, same as speech_graph

    return {
        "mean_coherence": round(mean_sim, 6),
        "std_coherence" : round(std_sim,  6),
    }


# =============================================================================
# PIPELINE
# =============================================================================

def extract_and_save(split_name: str, path: str) -> None:
    """
    Extract discourse coherence features for one split and save to CSV.

    Input:
        split_name — "train", "val", or "test"
        path       — path to the processed CSV for this split

    Output:
        None — saves results/features/structural/discourse_coherence_{split}.csv
        Shape: (num_samples, 2)

    Runtime:
        On GPU: ~10-15 minutes per split for 13k posts
        On CPU: ~45-60 minutes per split

        Sentence embedding is the bottleneck, not the cosine similarity
        computation.
    """
    logger.info(f"Processing {split_name} split...")

    df = pd.read_csv(path)

    tqdm.pandas(desc=f"  Discourse coherence ({split_name})")

    features = df[TEXT_COL].progress_apply(
        lambda text: pd.Series(compute_discourse_coherence(text))
    )

    save_dir  = STRUCTURAL_FEATURES_DIR / "discourse_coherence"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"discourse_coherence_{split_name}.csv"

    features.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path} | shape: {features.shape}")
    logger.info(f"Sample (row 0): {features.iloc[0].to_dict()}")


def run_discourse_coherence_pipeline() -> None:
    """
    Extract and save discourse coherence features for all three splits.
    Output: 2-dimensional feature vector per sample
            [mean_coherence, std_coherence].
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting discourse coherence feature extraction")
    logger.info("=" * 60)

    for split_name, path in splits.items():
        extract_and_save(split_name, path)

    logger.info("=" * 60)
    logger.info("Discourse coherence extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_discourse_coherence_pipeline()