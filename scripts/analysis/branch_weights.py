"""
Branch importance analysis for trained fusion models.

GatedFusion:
    Runs the test set through the model with return_gates=True and averages
    the softmax gate values across all samples.  Each gate value is the
    fraction of the fused representation contributed by that branch
    (semantic / affective / handcrafted).  Values sum to 1.0 per dimension;
    we report the mean across all 256 hidden dimensions.

LateConcatFusion:
    Has no explicit gate.  Branch importance is approximated from the first
    linear layer of the ClassifierHead (448 → 256).  We slice the weight
    matrix by branch (semantic: cols 0-255, affective: 256-383, handcrafted:
    384-447) and compute mean absolute weight per branch, then normalise to
    sum to 1.0 so the three numbers are directly comparable.

Usage (from project root):
    python -m scripts.analysis.branch_weights --model concat
    python -m scripts.analysis.branch_weights --model gated
    python -m scripts.analysis.branch_weights --model concat --model gated
    python -m scripts.analysis.branch_weights          # runs both

Results are saved to:
    results/evaluation/branch_weights_{model_type}.json
    results/evaluation/branch_weights_{model_type}.txt
"""

import argparse
import json
import logging

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import load_fusion_feature_tensors
from scripts.models.train_fusion import load_labels

logger = logging.getLogger(__name__)

BRANCHES = ["semantic", "affective", "handcrafted"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_test_loader(model_type: str, batch_size: int = 256):
    """Load and scale the test split, return a DataLoader."""
    semantic, affective, handcrafted, _ = load_fusion_feature_tensors(split="test")
    labels = load_labels("test")

    scaler_path = config.FUSION_MODEL_DIR / f"{model_type}_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Train the model first with train_fusion.py"
        )
    scaler = joblib.load(scaler_path)
    hc_scaled = torch.from_numpy(
        scaler.transform(handcrafted.numpy()).astype(np.float32)
    )
    ds = TensorDataset(semantic, affective, hc_scaled, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _load_model(model_type: str) -> torch.nn.Module:
    ckpt_path = config.FUSION_MODEL_DIR / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first with train_fusion.py"
        )
    model = build_fusion_model(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GatedFusion — extract gate values
# ---------------------------------------------------------------------------

def analyze_gated(batch_size: int = 256) -> dict:
    """
    Average softmax gate weights across the entire test set.

    Gate tensor shape per batch: (B, 3, 256)
      dim 0 = batch
      dim 1 = branch  (0=semantic, 1=affective, 2=handcrafted)
      dim 2 = hidden dimension

    We average over batch and hidden dims to get one scalar per branch.
    """
    print("\n" + "=" * 60)
    print("GatedFusion — gate weight analysis")
    print("=" * 60)

    model  = _load_model("gated")
    loader = _load_test_loader("gated", batch_size=batch_size)

    all_gates = []   # list of (B, 3, 256) tensors

    with torch.no_grad():
        for semantic, affective, handcrafted, _ in loader:
            _, gates = model(semantic, affective, handcrafted, return_gates=True)
            all_gates.append(gates.cpu())

    # (N, 3, 256)
    all_gates = torch.cat(all_gates, dim=0).numpy()

    # Mean over samples and hidden dimensions → shape (3,)
    mean_per_branch = all_gates.mean(axis=(0, 2))

    print(f"\n{'Branch':<14} {'Mean gate':>10}  {'Contribution':>13}")
    print("-" * 42)
    for name, val in zip(BRANCHES, mean_per_branch):
        print(f"  {name:<12} {val:>10.4f}  {val*100:>11.1f}%")

    print(f"\n  (gates are softmaxed across branches, so they sum to 1.0)")
    print(f"  Total: {mean_per_branch.sum():.4f}")

    # Per-class gate breakdown
    print(f"\n{'Branch':<14}", end="")
    id_to_class = config.ID_TO_CLASS
    for cls in id_to_class.values():
        print(f"  {cls:<10}", end="")
    print()
    print("-" * (14 + 12 * len(id_to_class)))

    # Reload with labels for per-class breakdown
    semantic, affective, handcrafted, _ = load_fusion_feature_tensors(split="test")
    labels = load_labels("test").numpy()
    scaler = joblib.load(config.FUSION_MODEL_DIR / "gated_scaler.joblib")
    hc_scaled = torch.from_numpy(scaler.transform(handcrafted.numpy()).astype(np.float32))

    all_gates_arr = []
    all_labels    = []
    ds = TensorDataset(semantic, affective, hc_scaled, torch.from_numpy(labels))
    loader2 = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for sem, aff, hc, lbl in loader2:
            _, gates = model(sem, aff, hc, return_gates=True)
            all_gates_arr.append(gates.cpu().numpy())
            all_labels.append(lbl.numpy())

    all_gates_arr = np.concatenate(all_gates_arr, axis=0)   # (N, 3, 256)
    all_labels    = np.concatenate(all_labels)               # (N,)

    per_class = {}
    for branch_idx, branch_name in enumerate(BRANCHES):
        print(f"  {branch_name:<12}", end="")
        per_class[branch_name] = {}
        for cls_id, cls_name in id_to_class.items():
            mask = all_labels == cls_id
            if mask.sum() == 0:
                val = float("nan")
            else:
                val = float(all_gates_arr[mask, branch_idx, :].mean())
            per_class[branch_name][cls_name] = round(val, 6)
            print(f"  {val:.4f}    ", end="")
        print()

    result = {
        "model_type": "gated",
        "overall": {
            name: round(float(val), 6)
            for name, val in zip(BRANCHES, mean_per_branch)
        },
        "per_class": per_class,
    }
    return result


# ---------------------------------------------------------------------------
# LateConcatFusion — weight-based branch importance
# ---------------------------------------------------------------------------

def analyze_concat() -> dict:
    """
    Approximate branch importance from the classifier head's weight matrix.

    The first linear layer of ClassifierHead is (448 → 256).
    Weight matrix shape: (256, 448).
    Slices by branch:
        semantic:     cols   0 – 255   (256 dims, SEMANTIC_PROJECTION_DIM)
        affective:    cols 256 – 383   (128 dims, AFFECTIVE_PROJECTION_DIM)
        handcrafted:  cols 384 – 447   ( 64 dims, HANDCRAFTED_PROJECTION_DIM)

    Mean absolute weight per branch, then normalised to sum to 1.0.
    """
    print("\n" + "=" * 60)
    print("LateConcatFusion — classifier weight-based branch importance")
    print("=" * 60)

    model = _load_model("concat")

    # classifier.layers[0] is the first nn.Linear inside ClassifierHead
    W = model.classifier.layers[0].weight.detach().numpy()  # (256, 448)

    s_end = config.SEMANTIC_PROJECTION_DIM                              # 256
    a_end = s_end + config.AFFECTIVE_PROJECTION_DIM                    # 384
    h_end = a_end + config.HANDCRAFTED_PROJECTION_DIM                  # 448

    slices = {
        "semantic":    W[:, :s_end],
        "affective":   W[:, s_end:a_end],
        "handcrafted": W[:, a_end:h_end],
    }

    raw = {name: float(np.abs(arr).mean()) for name, arr in slices.items()}
    total = sum(raw.values())
    normalised = {name: round(val / total, 6) for name, val in raw.items()}

    print(f"\n{'Branch':<14} {'Mean |w|':>10}  {'Normalised':>11}  {'Note'}")
    print("-" * 58)
    dims = {"semantic": s_end, "affective": config.AFFECTIVE_PROJECTION_DIM,
            "handcrafted": config.HANDCRAFTED_PROJECTION_DIM}
    for name in BRANCHES:
        print(f"  {name:<12} {raw[name]:>10.5f}  {normalised[name]:>10.4f}   "
              f"({dims[name]} projection dims)")

    print("\n  Note: LateConcatFusion has no explicit gate. This is an")
    print("  approximation based on the classifier head's weight magnitudes.")

    result = {
        "model_type":  "concat",
        "method":      "mean_abs_weight_of_classifier_head_layer0",
        "raw_mean_abs_weight": raw,
        "normalised":  normalised,
        "projection_dims": dims,
    }
    return result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(result: dict, model_type: str) -> None:
    eval_dir = config.EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / f"branch_weights_{model_type}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_dir / f"branch_weights_{model_type}.txt"
    lines = [f"Branch importance — {model_type.upper()}", "=" * 50]

    if model_type == "gated":
        lines.append("\nOverall mean gate (averaged across test set):")
        for name, val in result["overall"].items():
            lines.append(f"  {name:<14} {val:.4f}  ({val*100:.1f}%)")
        lines.append("\nPer-class mean gate:")
        header = f"  {'Branch':<14}" + "".join(
            f"  {c:<10}" for c in config.ID_TO_CLASS.values()
        )
        lines.append(header)
        for branch in BRANCHES:
            row = f"  {branch:<14}" + "".join(
                f"  {result['per_class'][branch][c]:.4f}    "
                for c in config.ID_TO_CLASS.values()
            )
            lines.append(row)
    else:
        lines.append("\nNormalised branch importance (from classifier head weights):")
        for name, val in result["normalised"].items():
            lines.append(f"  {name:<14} {val:.4f}  ({val*100:.1f}%)")
        lines.append("\n(approximation — no explicit gate in LateConcatFusion)")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {json_path}")
    print(f"Saved → {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["concat", "gated"], action="append",
                        dest="models",
                        help="Which model to analyse. Can be passed twice. "
                             "Defaults to both if omitted.")
    args = parser.parse_args()
    models_to_run = args.models or ["concat", "gated"]

    for model_type in models_to_run:
        try:
            if model_type == "gated":
                result = analyze_gated()
            else:
                result = analyze_concat()
            save_results(result, model_type)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {model_type}: {e}")
