"""
Fusion model training script.

Usage (from project root):
    python -m scripts.models.train_fusion                        # LateConcatFusion
    python -m scripts.models.train_fusion --model gated          # GatedFusion
    python -m scripts.models.train_fusion --model concat --gradnorm          # + GradNorm
    python -m scripts.models.train_fusion --model concat --epochs 20 --lr 5e-4

GradNorm (--gradnorm flag):
    Delegates to scripts.models.fusion.gradnorm.GradNormTrainer.
    Adds auxiliary classifier heads (one per branch) and a learnable loss-weight
    vector so all three branches learn at a similar rate.  Auxiliary heads are
    discarded after training; only the main model checkpoint is saved.

    Per-epoch gradient norms per branch are always printed as a diagnostic.

Results saved to:
    results/evaluation/fusion_{suffix}_results.json
    results/evaluation/fusion_{suffix}_summary.txt
    results/models/fusion/{suffix}_best.pt
    results/models/fusion/{suffix}_scaler.joblib

where suffix = model_type, or model_type_gradnorm when --gradnorm is used.
"""

import argparse
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import load_fusion_feature_tensors
from scripts.models.fusion.gradnorm import GradNormTrainer


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(split: str) -> torch.Tensor:
    path_map = {"train": config.TRAIN_PATH, "val": config.VAL_PATH, "test": config.TEST_PATH}
    df = pd.read_csv(path_map[split])
    return torch.tensor(df[config.LABEL_COL].values, dtype=torch.long)


def _load_split_tensors(split: str):
    semantic, affective, handcrafted, _ = load_fusion_feature_tensors(split=split)
    labels = load_labels(split)
    assert len(semantic) == len(labels), (
        f"{split}: feature count {len(semantic)} != label count {len(labels)}"
    )
    return semantic, affective, handcrafted, labels


def _scale_handcrafted(scaler, tensor: torch.Tensor, fit: bool = False) -> torch.Tensor:
    arr    = tensor.numpy()
    scaled = scaler.fit_transform(arr) if fit else scaler.transform(arr)
    return torch.from_numpy(scaled.astype(np.float32))


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Gradient norm diagnostic (always printed, independent of GradNorm training)
# ---------------------------------------------------------------------------

def _log_grad_norms(model: nn.Module) -> dict[str, float]:
    """Mean gradient norm per branch — call after a training backward pass."""
    norms = {}
    for name, attr in [
        ("semantic",    "semantic_branch"),
        ("affective",   "affective_branch"),
        ("handcrafted", "handcrafted_branch"),
    ]:
        branch = getattr(model, attr)
        grads  = [p.grad.norm().item() for p in branch.parameters() if p.grad is not None]
        norms[name] = float(np.mean(grads)) if grads else 0.0
    return norms


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    gradnorm_trainer: GradNormTrainer | None = None,
) -> tuple[float, float]:
    """
    One full pass over *loader*.
    optimizer=None       → evaluation mode, no weight updates.
    gradnorm_trainer!=None → use GradNormTrainer.step() instead of standard step.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for semantic, affective, handcrafted, labels in loader:
            semantic    = semantic.to(device)
            affective   = affective.to(device)
            handcrafted = handcrafted.to(device)
            labels      = labels.to(device)

            if training and gradnorm_trainer is not None:
                loss, acc, _ = gradnorm_trainer.step(
                    semantic, affective, handcrafted, labels, criterion, optimizer
                )
            else:
                logits = model(semantic, affective, handcrafted)
                loss_t = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss_t.backward()
                    optimizer.step()
                loss = loss_t.item()
                acc  = accuracy(logits.detach(), labels)

            total_loss += loss
            total_acc  += acc
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model_type: str       = "concat",
    epochs: int           = config.FUSION_EPOCHS,
    lr: float             = config.FUSION_LR,
    batch_size: int       = config.BATCH_SIZE,
    seed: int             = config.SEED,
    use_gradnorm: bool    = False,
    gradnorm_alpha: float = 1.5,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gn_tag = " + GradNorm" if use_gradnorm else ""
    print(f"\n{'='*60}")
    print(f"Fusion Training — {model_type.upper()}{gn_tag}  device={device}")
    print(f"epochs={epochs}  lr={lr}  batch={batch_size}  seed={seed}")
    if use_gradnorm:
        print(f"GradNorm alpha={gradnorm_alpha}")
    print(f"{'='*60}\n")

    # ---- load & scale features ----------------------------------------------
    print("Loading features …")
    t0 = time.time()
    sem_tr, aff_tr, hc_tr, lbl_tr = _load_split_tensors("train")
    sem_va, aff_va, hc_va, lbl_va = _load_split_tensors("val")
    sem_te, aff_te, hc_te, lbl_te = _load_split_tensors("test")
    print(f"  train={len(lbl_tr)}  val={len(lbl_va)}  test={len(lbl_te)}  ({time.time()-t0:.1f}s)")

    scaler = StandardScaler()
    hc_tr  = _scale_handcrafted(scaler, hc_tr, fit=True)
    hc_va  = _scale_handcrafted(scaler, hc_va)
    hc_te  = _scale_handcrafted(scaler, hc_te)
    print(f"  StandardScaler fitted on handcrafted (mean≈{scaler.mean_.mean():.3f})\n")

    train_ds = TensorDataset(sem_tr, aff_tr, hc_tr, lbl_tr)
    val_ds   = TensorDataset(sem_va, aff_va, hc_va, lbl_va)
    test_ds  = TensorDataset(sem_te, aff_te, hc_te, lbl_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- model --------------------------------------------------------------
    model     = build_fusion_model(model_type).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}  params={trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- GradNorm trainer (optional) ----------------------------------------
    gradnorm_trainer = None
    if use_gradnorm:
        gradnorm_trainer = GradNormTrainer(model, device, alpha=gradnorm_alpha, lr=lr)
        dims = gradnorm_trainer.proj_dims
        print(f"GradNormTrainer: aux heads {dims[0]}→6  {dims[1]}→6  {dims[2]}→6")
    print()

    # ---- training loop ------------------------------------------------------
    history      = []
    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    header = (f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7}")
    if use_gradnorm:
        header += "  | w_sem  w_aff  w_hc"
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, gradnorm_trainer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        grad_norms = _log_grad_norms(model)

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "grad_norms": {k: round(v, 6) for k, v in grad_norms.items()},
        }

        line = (f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.4f}% | "
                f"{val_loss:>8.4f} | {val_acc:>6.4f}%")

        if use_gradnorm and gradnorm_trainer is not None:
            w = gradnorm_trainer.current_weights()
            row["loss_weights"] = w
            line += f"  | {w[0]:.3f}  {w[1]:.3f}  {w[2]:.3f}"

        print(line)
        gn_str = "  ".join(f"{k}={v:.4f}" for k, v in grad_norms.items())
        print(f"       grad norms → {gn_str}")

        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- test evaluation ----------------------------------------------------
    print(f"\nBest val acc {best_val_acc:.4f}% at epoch {best_epoch} — evaluating on test …")
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.4f}%")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sem, aff, hc, lbl in test_loader:
            logits = model(sem.to(device), aff.to(device), hc.to(device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(lbl)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    per_class_acc = {}
    for cls_id, cls_name in config.ID_TO_CLASS.items():
        mask = all_labels == cls_id
        per_class_acc[cls_name] = (
            round(float((all_preds[mask] == cls_id).mean()), 6) if mask.sum() > 0 else None
        )
    print("\nPer-class accuracy on test set:")
    for name, v in per_class_acc.items():
        print(f"  {name:<12} {f'{v:.4f}%' if v is not None else 'N/A'}")

    # ---- save checkpoint + scaler ------------------------------------------
    model_dir = config.FUSION_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    suffix      = f"{model_type}_gradnorm" if use_gradnorm else model_type
    ckpt_path   = model_dir / f"{suffix}_best.pt"
    scaler_path = model_dir / f"{suffix}_scaler.joblib"

    torch.save(best_state, ckpt_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nCheckpoint → {ckpt_path}")
    print(f"Scaler     → {scaler_path}")

    return {
        "model_type":      model_type,
        "use_gradnorm":    use_gradnorm,
        "gradnorm_alpha":  gradnorm_alpha if use_gradnorm else None,
        "epochs_trained":  epochs,
        "lr":              lr,
        "batch_size":      batch_size,
        "seed":            seed,
        "best_epoch":      best_epoch,
        "best_val_acc":    round(best_val_acc, 6),
        "test_loss":       round(test_loss, 6),
        "test_acc":        round(test_acc, 6),
        "per_class_acc":   per_class_acc,
        "history":         history,
        "checkpoint_path": str(ckpt_path),
        "scaler_path":     str(scaler_path),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, suffix: str) -> None:
    eval_dir = config.EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / f"fusion_{suffix}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    gn_tag = " + GradNorm" if results.get("use_gradnorm") else ""
    txt_path = eval_dir / f"fusion_{suffix}_summary.txt"
    lines = [
        f"Fusion Training Summary — {results['model_type'].upper()}{gn_tag}",
        "=" * 50,
        f"Epochs trained  : {results['epochs_trained']}",
        f"Learning rate   : {results['lr']}",
        f"Batch size      : {results['batch_size']}",
        f"Seed            : {results['seed']}",
        f"Best epoch      : {results['best_epoch']}",
        f"Best val acc    : {results['best_val_acc']:.4f}%",
        f"Test loss       : {results['test_loss']:.4f}",
        f"Test accuracy   : {results['test_acc']:.4f}%",
        "",
        "Per-class accuracy (test):",
    ]
    for name, v in results["per_class_acc"].items():
        lines.append(f"  {name:<12} {f'{v:.4f}%' if v is not None else 'N/A'}")

    lines += ["", "Epoch history:"]
    has_gn = results.get("use_gradnorm")
    hdr = (f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6} | "
           f"{'sem_gn':>6} {'aff_gn':>6} {'hc_gn':>6}")
    if has_gn:
        hdr += "  | w_sem  w_aff  w_hc"
    lines += [hdr, "-" * len(hdr)]

    for row in results["history"]:
        gn = row.get("grad_norms", {})
        line = (f"{row['epoch']:>3} | {row['train_loss']:>7.4f} | {row['train_acc']:>5.3f}% | "
                f"{row['val_loss']:>7.4f} | {row['val_acc']:>5.3f}% | "
                f"{gn.get('semantic',0):>6.4f} {gn.get('affective',0):>6.4f} {gn.get('handcrafted',0):>6.4f}")
        if has_gn and "loss_weights" in row:
            w = row["loss_weights"]
            line += f"  | {w[0]:.3f}  {w[1]:.3f}  {w[2]:.3f}"
        lines.append(line)

    lines += ["", f"Checkpoint : {results['checkpoint_path']}",
              f"Scaler     : {results['scaler_path']}"]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Results JSON    → {json_path}")
    print(f"Results summary → {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="concat", choices=["concat", "gated"])
    parser.add_argument("--epochs",   type=int,   default=config.FUSION_EPOCHS)
    parser.add_argument("--lr",       type=float, default=config.FUSION_LR)
    parser.add_argument("--batch",    type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--seed",     type=int,   default=config.SEED)
    parser.add_argument("--gradnorm", action="store_true",
                        help="Enable GradNorm branch balancing (see fusion/gradnorm.py)")
    parser.add_argument("--alpha",    type=float, default=1.5,
                        help="GradNorm alpha — asymmetry parameter (default 1.5)")
    args = parser.parse_args()

    results = train(
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
        use_gradnorm=args.gradnorm,
        gradnorm_alpha=args.alpha,
    )
    suffix = f"{args.model}_gradnorm" if args.gradnorm else args.model
    save_results(results, suffix)
