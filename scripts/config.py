# scripts/config.py

from pathlib import Path

# =============================================================================
# ROOT
# =============================================================================
# Path(__file__) gives the absolute path of this config.py file itself.
# .parent gives the folder containing it, which is scripts/.
# .parent again gives the project root folder.
# Everything else is built relative to this so the project works on any machine.

ROOT_DIR = Path(__file__).parent.parent


# =============================================================================
# DATA PATHS
# =============================================================================
# Original raw CSVs — your code will ONLY read from here, never write.
# Processed CSVs — output of preprocessing.py, input to everything else.

DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "original"
PROCESSED_DIR   = DATA_DIR / "processed"

# Raw files (as downloaded)
RAW_TRAIN_PATH  = RAW_DIR / "both_train.csv"
RAW_VAL_PATH    = RAW_DIR / "both_val.csv"
RAW_TEST_PATH   = RAW_DIR / "both_test.csv"

# Processed files (output of preprocessing.py)
TRAIN_PATH      = PROCESSED_DIR / "train.csv"
VAL_PATH        = PROCESSED_DIR / "val.csv"
TEST_PATH       = PROCESSED_DIR / "test.csv"


# =============================================================================
# RESULTS PATHS
# =============================================================================
# Everything your code produces goes under results/.
# Sub-folders are organized by what kind of output they hold.

RESULTS_DIR     = ROOT_DIR / "results"
FEATURES_DIR    = RESULTS_DIR / "features"
MODELS_DIR      = RESULTS_DIR / "models"
EVAL_DIR        = RESULTS_DIR / "evaluation"
PLOTS_DIR       = RESULTS_DIR / "plots"
LOGS_DIR        = RESULTS_DIR / "logs"


# =============================================================================
# FEATURE SUB-DIRECTORIES
# =============================================================================
# Each feature group saves its output into its own sub-folder under features/.
# This keeps feature files organized by group and easy to load selectively
# during ablation studies (you can drop one group just by not loading its folder).

SEMANTIC_FEATURES_DIR    = FEATURES_DIR / "semantic"
AFFECTIVE_FEATURES_DIR   = FEATURES_DIR / "affective"
STRUCTURAL_FEATURES_DIR  = FEATURES_DIR / "structural"
STYLISTIC_FEATURES_DIR   = FEATURES_DIR / "stylistic"


# =============================================================================
# MODEL SUB-DIRECTORIES
# =============================================================================

ROBERTA_MODEL_DIR   = MODELS_DIR / "roberta"
TOKENIZED_DIR       = ROBERTA_MODEL_DIR / "tokenized"  
FUSION_MODEL_DIR    = MODELS_DIR / "fusion"
BASELINE_MODEL_DIR  = MODELS_DIR / "baselines"


# =============================================================================
# COLUMN NAMES
# =============================================================================
# Centralizing column names means if your CSV ever has a different column name
# you fix it here once, not in every file that reads the data.

RAW_TITLE_COL   = "title"
RAW_POST_COL    = "post"
TEXT_COL        = "text"        # The merged column created in preprocessing
LABEL_COL       = "class_id"   # Integer label used for model training
CLASS_NAME_COL  = "class_name" # Human-readable label used for analysis/plots


# =============================================================================
# CLASS LABELS
# =============================================================================
# Mapping between integer class IDs and human-readable names.
# Used in evaluation and plots so your confusion matrix shows
# "Depression" instead of "3".

NUM_LABELS = 6

ID_TO_CLASS = {
    0: "ADHD",
    1: "Anxiety",
    2: "Bipolar",
    3: "Depression",
    4: "PTSD",
    5: "None",
}

CLASS_TO_ID = {v: k for k, v in ID_TO_CLASS.items()}


# =============================================================================
# PREPROCESSING SETTINGS
# =============================================================================

# Separator used when merging title and post into a single text field.
# ": " is chosen because it reads naturally: "Title: post body..."
MERGE_SEPARATOR = ": "


# =============================================================================
# ROBERTA SETTINGS
# =============================================================================

MENTAL_ROBERTA_NAME = "mental/mental-roberta-base"  # domain-adapted variant
MAX_LENGTH          = 512       # Maximum token length RoBERTa accepts
BATCH_SIZE          = 32        # Number of samples processed per training step
LEARNING_RATE       = 2e-5      # Standard fine-tuning LR for transformer models
NUM_EPOCHS          = 2         # Number of full passes over the training data
WEIGHT_DECAY        = 0.01      # Regularization to prevent overfitting


# =============================================================================
# EMOTION MODEL SETTINGS
# =============================================================================

EMOTION_MODEL_NAME  = "SamLowe/roberta-base-go_emotions"
EMOTION_BATCH_SIZE  = 16


# =============================================================================
# TFIDF SETTINGS
# =============================================================================

TFIDF_MAX_FEATURES  = 10000     # Vocabulary size limit
TFIDF_NGRAM_RANGE   = (1, 2)    # Use both unigrams and bigrams


# =============================================================================
# FUSION MODEL SETTINGS
# =============================================================================

PROJECTION_DIM  = 256   # Every feature group is projected to this dimension
FUSION_DIM      = PROJECTION_DIM * 4   # 4 groups x 256 = 1024 input to classifier
FUSION_LR       = 1e-3
FUSION_EPOCHS   = 15

# GradNorm settings
GRADNORM_ALPHA  = 1.5   # Controls strength of gradient rebalancing


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# A fixed seed ensures your results are the same every run.
# Required for any research paper claiming reproducible results.

SEED = 42