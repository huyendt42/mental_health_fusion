"""
Standalone evaluation script — test any trained fusion model on any split.

Loads a saved checkpoint + scaler, runs inference, and reports:
  - Overall accuracy
  - Macro / weighted F1
  - Per-class precision, recall, F1
  - Confusion matrix
  - Side-by-side comparison when multiple models are available

Usage (from project root):
    python -m scripts.models.evaluate_fusion                 # all available models on test
    python -m scripts.models.evaluate_fusion --split val     # evaluate on val split
    python -m scripts.models.evaluate_fusion --model concat  # one model only
    python -m scripts.models.evaluate_fusion --model concat --model gated  # explicit list

Results saved to:
    results/evaluation/eval_{suffix}_{split}.json
    results/evaluation/eval_{suffix}_{split}.txt
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import load_fusion_feature_tensors
from scripts.models.train_fusion import load_labels


# ---------------------------------------------------------------------------
# Available checkpoints  (auto-detected from disk)
# ---------------------------------------------------------------------------

def _discover_checkpoints() -> list[str]:
    """Return all model suffixes that have both a .pt and a .joblib saved."""
    model_dir = config.FUSION_MODEL_DIR
    if not model_dir.exists():
        return []
    suffixes = []
    for pt in model_dir.glob("*_best.pt"):
        suffix = pt.stem.replace("_best", "")
        scaler = model_dir / f"{suffix}_scaler.joblib"
        if scaler.exists():
            suffixes.append(suffix)
    return sorted(suffixes)


def _suffix_to_model_type(suffix: str) -> str:
    """'concat_gradnorm' → 'concat',  'gated' → 'gated'."""
    if suffix.startswith("concat"):
        return "concat"
    if suffix.startswith("gated"):
        return "gated"
    raise ValueError(f"Cannot infer model type from suffix: {suffix}")


# ---------------------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------------------

def load_model_and_scaler(suffix: str):
    ckpt_path   = config.FUSION_MODEL_DIR / f"{suffix}_best.pt"
    scaler_path = config.FUSION_MODEL_DIR / f"{suffix}_scaler.joblib"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model_type = _suffix_to_model_type(suffix)
    model = build_fusion_model(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler


def load_split(suffix: str, split: str, scaler):
    """Return a DataLoader for the given split using the fitted scaler."""
    semantic, affective, handcrafted, _ = load_fusion_feature_tensors(split=split)
    labels = load_labels(split)

    hc_scaled = torch.from_numpy(
        scaler.transform(handcrafted.numpy()).astype(np.float32)
    )
    ds     = TensorDataset(semantic, affective, hc_scaled, labels)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    return loader, labels.numpy()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(suffix: str, split: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {suffix}  |  split: {split}")
    print(f"{'='*60}")

    model, scaler = load_model_and_scaler(suffix)
    loader, true_labels = load_split(suffix, split, scaler)

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]

    all_preds  = []
    all_logits = []

    with torch.no_grad():
        for sem, aff, hc, _ in loader:
            logits = model(sem, aff, hc)
            all_preds.append(logits.argmax(dim=1).numpy())
            all_logits.append(logits.numpy())

    preds  = np.concatenate(all_preds)
    logits = np.concatenate(all_logits)

    # --- Metrics ---
    acc      = float((preds == true_labels).mean())
    macro_f1 = float(f1_score(true_labels, preds, average="macro",    zero_division=0))
    wt_f1    = float(f1_score(true_labels, preds, average="weighted", zero_division=0))

    report_dict = classification_report(
        true_labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str  = classification_report(
        true_labels, preds,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, preds)

    # --- Print ---
    print(f"\nOverall accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro  F1        : {macro_f1:.4f}")
    print(f"Weighted F1      : {wt_f1:.4f}")
    print(f"\n{report_str}")

    print("Confusion matrix (rows=true, cols=predicted):")
    header = f"{'':>12}" + "".join(f"{c[:6]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:<10}" + "".join(f"{v:>8}" for v in row))

    return {
        "suffix":      suffix,
        "split":       split,
        "accuracy":    round(acc,      6),
        "macro_f1":    round(macro_f1, 6),
        "weighted_f1": round(wt_f1,    6),
        "per_class":   {
            cls: {
                "precision": round(report_dict[cls]["precision"], 6),
                "recall":    round(report_dict[cls]["recall"],    6),
                "f1":        round(report_dict[cls]["f1-score"],  6),
                "support":   int(report_dict[cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "class_names":      class_names,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_evaluation(result: dict) -> None:
    eval_dir = config.EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    suffix = result["suffix"]
    split  = result["split"]

    json_path = eval_dir / f"eval_{suffix}_{split}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_dir / f"eval_{suffix}_{split}.txt"
    class_names = result["class_names"]
    lines = [
        f"Evaluation — {suffix}  |  split: {split}",
        "=" * 50,
        f"Accuracy    : {result['accuracy']:.4f}  ({result['accuracy']*100:.2f}%)",
        f"Macro F1    : {result['macro_f1']:.4f}",
        f"Weighted F1 : {result['weighted_f1']:.4f}",
        "",
        f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}",
        "-" * 52,
    ]
    for cls in class_names:
        pc = result["per_class"][cls]
        lines.append(
            f"{cls:<12} {pc['precision']:>10.4f} {pc['recall']:>8.4f} "
            f"{pc['f1']:>8.4f} {pc['support']:>9}"
        )
    lines += ["", "Confusion matrix (rows=true, cols=predicted):"]
    lines.append(f"{'':>12}" + "".join(f"{c[:6]:>8}" for c in class_names))
    for i, row in enumerate(result["confusion_matrix"]):
        lines.append(f"  {class_names[i]:<10}" + "".join(f"{v:>8}" for v in row))

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {json_path}")
    print(f"Saved → {txt_path}")


# ---------------------------------------------------------------------------
# Comparison table (when multiple models evaluated)
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    if len(results) < 2:
        return
    print(f"\n{'='*60}")
    print("Model comparison")
    print(f"{'='*60}")
    print(f"{'Model':<22} {'Accuracy':>9} {'Macro F1':>9} {'Wtd F1':>8}")
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        marker = " ← best" if r == max(results, key=lambda x: x["accuracy"]) else ""
        print(f"{r['suffix']:<22} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
              f"{r['weighted_f1']:>8.4f}{marker}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", dest="models", metavar="SUFFIX",
        help="Model suffix to evaluate (e.g. concat, gated, concat_gradnorm). "
             "Can be repeated. Defaults to all available checkpoints.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate on (default: test).",
    )
    args = parser.parse_args()

    targets = args.models or _discover_checkpoints()

    if not targets:
        print("No trained models found in", config.FUSION_MODEL_DIR)
        print("Train a model first: python -m scripts.models.train_fusion")
        raise SystemExit(1)

    print(f"Models to evaluate: {targets}  |  split: {args.split}")

    all_results = []
    for suffix in targets:
        try:
            result = evaluate(suffix, args.split)
            save_evaluation(result)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {suffix}: {e}")

    print_comparison(all_results)
