# scripts/features/semantic/roberta_embed.py

import random
import logging
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,          # NEW — enables dynamic padding
)
import evaluate

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TEXT_COL, LABEL_COL,
    MENTAL_ROBERTA_NAME,
    MAX_LENGTH, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY,
    ROBERTA_MODEL_DIR,
    SEMANTIC_FEATURES_DIR,
    LOGS_DIR,
    NUM_LABELS,
    SEED
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# STEP 1 — TOKENIZATION (with dynamic padding)
# =============================================================================

def tokenize_and_save(tokenizer) -> None:
    """
    Tokenize with truncation but NO padding.
    Padding is now handled dynamically at batch creation time by
    DataCollatorWithPadding — padding each batch only to the length of
    the longest post in that batch, not a fixed 512.

    Why this matters:
    - Median post length: ~180 tokens
    - Old behavior: every batch padded to 512 → wasted compute on padding
    - New behavior: most batches padded to ~200 → ~2.5x faster
    """
    raw = DatasetDict({
        "train"     : Dataset.from_pandas(pd.read_csv(TRAIN_PATH)),
        "validation": Dataset.from_pandas(pd.read_csv(VAL_PATH)),
        "test"      : Dataset.from_pandas(pd.read_csv(TEST_PATH)),
    })

    def tokenize_batch(batch):
        # CHANGED: removed padding here — it's now done per-batch at training time
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH,
            # NO padding argument — leaves each post its natural length
        )

    logger.info("Tokenizing all splits (no padding)...")
    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
    )

    tokenized = tokenized.rename_column(LABEL_COL, "labels")

    # Note: we don't set_format to torch here because dynamic padding
    # needs the raw lists. The collator handles tensor conversion.
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    save_path = ROBERTA_MODEL_DIR / "tokenized"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(save_path))
    logger.info(f"Tokenized dataset saved → {save_path}")


# =============================================================================
# STEP 2 — FINE-TUNING (with fp16 + dynamic padding + larger batch)
# =============================================================================

def fine_tune(tokenized_path: Path) -> None:
    logger.info("Loading tokenized dataset...")
    tokenized = load_from_disk(str(tokenized_path))

    logger.info(f"Loading {MENTAL_ROBERTA_NAME} for sequence classification...")
    model = RobertaForSequenceClassification.from_pretrained(
        MENTAL_ROBERTA_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(MENTAL_ROBERTA_NAME)

    # DataCollatorWithPadding handles dynamic padding at batch creation time.
    # It pads each batch only to the max length WITHIN that batch.
    # If a batch has posts with 150, 200, 180 tokens, it pads all to 200
    # instead of to a fixed 512.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            **acc_metric.compute(predictions=preds, references=labels),
            **f1_metric.compute(predictions=preds, references=labels, average="macro"),
        }

    # Only enable fp16 if GPU is available (it has no effect on CPU anyway
    # but setting True on CPU causes a crash with some PyTorch versions).
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir                  = str(ROBERTA_MODEL_DIR),
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,   # eval can use larger batch
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        save_total_limit            = 2,                # only keep 2 best checkpoints
        logging_dir                 = str(LOGS_DIR),
        logging_steps               = 50,
        fp16                        = use_fp16,         # RE-ENABLED
        bf16                        = False,
        seed                        = SEED,
        warmup_ratio                = 0.1,
        dataloader_num_workers      = 2,                # parallel data loading
        gradient_accumulation_steps = 1,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = tokenized["train"],
        eval_dataset    = tokenized["validation"],
        processing_class = tokenizer,
        data_collator    = data_collator,               # NEW — enables dynamic padding
        compute_metrics  = compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=True)

    ROBERTA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ROBERTA_MODEL_DIR))
    tokenizer.save_pretrained(str(ROBERTA_MODEL_DIR))
    logger.info(f"Fine-tuned model saved → {ROBERTA_MODEL_DIR}")


# =============================================================================
# STEP 3 — EMBEDDING EXTRACTION (unchanged except fp16)
# =============================================================================

def extract_cls_embeddings(
    split_name: str,
    tokenized_path: Path,
    model,
    device: torch.device,
) -> np.ndarray:
    tokenized = load_from_disk(str(tokenized_path))
    dataset   = tokenized[split_name]
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # For embedding extraction we also benefit from dynamic padding via collator.
    # But since each post has a different length and we're calling model.eval,
    # we use the simpler DataLoader approach and handle padding manually
    # by keeping per-batch variable length in the dataset itself.
    tokenizer = RobertaTokenizerFast.from_pretrained(str(ROBERTA_MODEL_DIR))
    collator  = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE * 2,      # larger batch for inference
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()
    model.to(device)

    all_embeddings = []

    logger.info(f"Extracting [CLS] embeddings for {split_name}...")
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            cls_vectors = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_vectors.cpu().numpy())

    return np.vstack(all_embeddings)


def save_embeddings(embeddings: np.ndarray, split_name: str) -> None:
    save_dir = SEMANTIC_FEATURES_DIR / "roberta"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"roberta_{split_name}.npz"
    np.savez_compressed(save_path, embeddings=embeddings)
    logger.info(f"Saved embeddings → {save_path} | shape: {embeddings.shape}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_roberta_pipeline() -> None:
    set_seed(SEED)

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_path = ROBERTA_MODEL_DIR / "tokenized"

    logger.info("=" * 60)
    logger.info("Stage: MentalRoBERTa — Tokenization")
    logger.info("=" * 60)
    tokenizer = RobertaTokenizerFast.from_pretrained(MENTAL_ROBERTA_NAME)

    # CHANGED: always re-tokenize since we switched from static to dynamic padding.
    # If you already tokenized with the old (max_length) padding, delete
    # results/models/roberta/tokenized/ before running so it uses the new format.
    if not tokenized_path.exists():
        tokenize_and_save(tokenizer)
    else:
        logger.info("Tokenized dataset already exists — skipping tokenization.")
        logger.info(
            "NOTE: If this is your first run with dynamic padding, "
            "delete the tokenized folder and re-run to regenerate."
        )

    logger.info("=" * 60)
    logger.info("Stage: MentalRoBERTa — Fine-tuning")
    logger.info("=" * 60)
    fine_tune(tokenized_path)

    logger.info("=" * 60)
    logger.info("Stage: MentalRoBERTa — Embedding Extraction")
    logger.info("=" * 60)
    model = RobertaForSequenceClassification.from_pretrained(str(ROBERTA_MODEL_DIR))

    SPLIT_NAME_MAP = {
        "train"     : "train",
        "validation": "val",
        "test"      : "test",
    }

    for hf_split, file_split in SPLIT_NAME_MAP.items():
        embeddings = extract_cls_embeddings(hf_split, tokenized_path, model, device)
        save_embeddings(embeddings, file_split)

    logger.info("=" * 60)
    logger.info("RoBERTa pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_roberta_pipeline()