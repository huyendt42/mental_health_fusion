# scripts/features/affective/emotions.py

import logging
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL,
    EMOTION_MODEL_NAME,          # "SamLowe/roberta-base-go_emotions"
    EMOTION_BATCH_SIZE,          # 16
    AFFECTIVE_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_emotion_pipeline():
    """
    Load the GoEmotions model as a HuggingFace pipeline.

    Returns:
        pipeline object that takes text(s) and returns emotion scores

    Why use pipeline() instead of loading model + tokenizer manually?
    For inference-only tasks where we just want model output, pipeline()
    is simpler — it handles tokenization, batching, device placement, and
    softmax automatically. We only need manual model loading when we
    want fine-grained control like extracting hidden states (which we did
    for roberta_embed.py).

    Why top_k=None?
    By default, text-classification pipeline returns only the top-1 label.
    top_k=None makes it return ALL 28 emotion scores as a probability
    distribution — this is what we want for a feature vector.

    Why truncation=True, max_length=512?
    GoEmotions' RoBERTa backbone accepts max 512 tokens, same as
    MentalRoBERTa. Posts longer than this get truncated. Without this
    setting, the model would crash on long Reddit posts.
    """
    # Use GPU if available, CPU otherwise.
    # device=0 means "first GPU" in HuggingFace pipeline convention.
    # device=-1 means CPU.
    device = 0 if torch.cuda.is_available() else -1
    device_label = "GPU" if device == 0 else "CPU"

    logger.info(f"Loading {EMOTION_MODEL_NAME} on {device_label}...")

    emotion_pipeline = pipeline(
        task        = "text-classification",
        model       = EMOTION_MODEL_NAME,
        top_k       = None,           # Return all 28 emotion scores
        truncation  = True,
        max_length  = 512,
        device      = device,
    )

    logger.info("Emotion pipeline ready.")
    return emotion_pipeline


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_emotion_features(df: pd.DataFrame, emotion_pipeline) -> pd.DataFrame:
    """
    Run the emotion pipeline over all texts and build a 28-dim feature frame.

    Input:
        df                 — processed DataFrame with TEXT_COL column
        emotion_pipeline   — loaded HuggingFace pipeline from setup_emotion_pipeline()

    Output:
        DataFrame of shape (num_samples, 28) with columns named:
        emo_admiration, emo_amusement, emo_anger, ..., emo_sadness,
        emo_surprise, emo_neutral

    Why prefix each column with "emo_"?
    When we later concatenate features from multiple groups (semantic,
    affective, structural, stylistic), having unique column prefixes
    prevents naming collisions. "emo_anger" clearly identifies this
    feature as belonging to the affective group.

    Why sort the emotion labels alphabetically?
    The pipeline may return emotions in different order on different
    batches. Sorting once at the start gives us a stable column order
    so train/val/test feature files have columns in the same positions.
    """
    # Convert text column to list of strings — pipeline expects this format
    # str() is applied defensively in case any row contains a non-string value
    texts = df[TEXT_COL].astype(str).tolist()
    logger.info(f"  Extracting emotions for {len(texts):,} samples...")

    # Run the pipeline in batches. tqdm wraps the generator to show progress.
    # batch_size is passed to the pipeline — it handles batching internally
    # and returns results in the same order as input.
    results = emotion_pipeline(texts, batch_size=EMOTION_BATCH_SIZE)

    # Determine stable column order from the first sample's output.
    # Each result is a list of 28 dicts like: [{"label": "anger", "score": 0.12}, ...]
    first_sample = results[0]
    label_order  = sorted(item["label"] for item in first_sample)

    # Build feature rows one by one.
    # For each sample's list of {label, score} dicts, we convert it into
    # a dict keyed by "emo_{label}" for easy DataFrame construction.
    feature_rows = []
    for result in tqdm(results, desc="  Building feature vectors"):
        score_dict = {item["label"]: item["score"] for item in result}
        row = {f"emo_{label}": score_dict[label] for label in label_order}
        feature_rows.append(row)

    # Build DataFrame — columns will be in the sorted label_order
    return pd.DataFrame(feature_rows)


# =============================================================================
# SAVE
# =============================================================================

def save_features(features: pd.DataFrame, split_name: str) -> None:
    """
    Save emotion features to CSV.

    Input:
        features   — DataFrame of shape (num_samples, 28)
        split_name — "train", "val", or "test"

    Output:
        None — saves results/features/affective/emotions_{split}.csv

    Why CSV instead of .npz here?
    The 28-dim output is small enough (~2MB per split) that CSV is fine
    and gives us readable column names (emo_anger, emo_fear, etc.)
    which is helpful during statistical analysis and debugging.
    """
    save_dir = AFFECTIVE_FEATURES_DIR / "emotions"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"emotions_{split_name}.csv"
    features.to_csv(save_path, index=False)
    logger.info(f"  Saved → {save_path} | shape: {features.shape}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_emotions_pipeline() -> None:
    """
    Extract and save GoEmotions features for all three splits.
    Output per split: 28-dimensional probability vector per sample.
    """
    splits = {
        "train": TRAIN_PATH,
        "val"  : VAL_PATH,
        "test" : TEST_PATH,
    }

    logger.info("=" * 60)
    logger.info("Starting GoEmotions feature extraction")
    logger.info("=" * 60)

    # Load the emotion model ONCE, outside the loop.
    # Loading it inside the loop would re-download/reload for every split.
    emotion_pipeline = setup_emotion_pipeline()

    for split_name, path in splits.items():
        logger.info(f"\nProcessing {split_name} split...")

        df = pd.read_csv(path)
        features = extract_emotion_features(df, emotion_pipeline)
        save_features(features, split_name)

        # Sanity check — show the top 3 emotions for the first sample
        sample_emotions = features.iloc[0].sort_values(ascending=False).head(3)
        logger.info(f"  Sample (row 0) top 3 emotions: {sample_emotions.to_dict()}")

    logger.info("=" * 60)
    logger.info("GoEmotions extraction complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_emotions_pipeline()