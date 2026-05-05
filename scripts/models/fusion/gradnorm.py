"""
GradNorm branch-balancing trainer (Chen et al., 2018).

This module is the GradNorm counterpart to late_concat.py and gated.py.
It wraps any fusion model with lightweight auxiliary classifier heads and
a learnable loss-weight vector so that all three input branches
(semantic, affective, handcrafted) learn at a similar rate during training.

The auxiliary heads and loss weights are training-only artifacts — they are
NOT saved in the model checkpoint and are NOT used at inference time.

Algorithm summary
-----------------
For each training step:
  1. Run the model's get_branch_representations() to get per-branch projections.
  2. Pass each projection through its auxiliary head to produce a branch loss L_i.
  3. Compute gradient norms G_i = ||grad(w_i * L_i) w.r.t. W|| where W is the
     first linear layer of the classifier head (the last shared layer).
  4. Compute relative inverse training rates r_i = (L_i / L_i_0) / mean(L_j / L_j_0).
  5. GradNorm loss = sum |G_i - G_avg * r_i^alpha|.
  6. Update model weights from total_loss (main + weighted aux).
  7. Update loss weights w_i from GradNorm loss only.
  8. Renormalise w_i so they sum to n_branches.

Usage
-----
    from scripts.models.fusion.gradnorm import GradNormTrainer

    trainer = GradNormTrainer(model, device, alpha=1.5, lr=1e-3)
    loss, acc, weights = trainer.step(semantic, affective, handcrafted, labels,
                                      criterion, optimizer)
    print(trainer.weight_str())   # "w=[sem=1.24 aff=0.89 hc=0.87]"
"""

import torch
import torch.nn as nn

from scripts import config


BRANCH_NAMES = ["semantic", "affective", "handcrafted"]


class GradNormTrainer:
    """
    Wraps a fusion model with GradNorm auxiliary heads and loss weights.

    Parameters
    ----------
    model   : a LateConcatFusion or GatedFusion instance (must expose
              get_branch_representations() and classifier.layers[0])
    device  : torch.device
    alpha   : GradNorm asymmetry parameter (paper default 1.5;
              lower values → softer balancing)
    lr      : learning rate for the loss weights + aux heads optimiser
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        alpha: float = 1.5,
        lr: float = config.FUSION_LR,
    ):
        self.model  = model
        self.device = device
        self.alpha  = alpha

        # Branch projection output dims — works for both LateConcatFusion and GatedFusion
        self.proj_dims = [
            model.semantic_branch.linear.out_features,
            model.affective_branch.linear.out_features,
            model.handcrafted_branch.linear.out_features,
        ]

        # One lightweight classifier per branch (training-only)
        self.aux_heads = nn.ModuleList([
            nn.Linear(d, config.NUM_LABELS).to(device)
            for d in self.proj_dims
        ])

        # Learnable loss weights — one per branch, kept positive, sum ≈ n_branches
        self.loss_weights = nn.Parameter(
            torch.ones(len(BRANCH_NAMES), device=device)
        )

        # Separate optimiser: updates loss weights + aux head params only
        self.gn_optimizer = torch.optim.Adam(
            list(self.aux_heads.parameters()) + [self.loss_weights],
            lr=lr,
        )

        # L_i(t=0): set on the first step, used for relative training rate
        self._initial_losses: list[float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        semantic: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        model_optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, list[float]]:
        """
        One GradNorm training step.

        Returns
        -------
        main_loss : float   — cross-entropy loss from the fusion classifier
        accuracy  : float   — fraction correct on this batch
        weights   : list[float] — current normalised loss weights [sem, aff, hc]
        """
        # --- Forward: main fusion ---
        logits    = self.model(semantic, affective, handcrafted)
        main_loss = criterion(logits, labels)

        # --- Forward: per-branch auxiliary losses ---
        sem_r, aff_r, hc_r = self.model.get_branch_representations(
            semantic, affective, handcrafted
        )
        branch_losses = [
            criterion(head(r), labels)
            for head, r in zip(self.aux_heads, [sem_r, aff_r, hc_r])
        ]

        if self._initial_losses is None:
            self._initial_losses = [l.item() for l in branch_losses]

        # Normalised weights: positive, sum = n_branches
        n = len(BRANCH_NAMES)
        w = self.loss_weights.clamp(min=1e-4)
        w = w * n / w.sum()

        weighted_aux = sum(wi * li for wi, li in zip(w, branch_losses))
        total_loss   = main_loss + 0.1 * weighted_aux

        # --- GradNorm loss ---
        ref_W = self.model.classifier.layers[0].weight
        G = [
            torch.autograd.grad(
                wi * li, ref_W,
                retain_graph=True, create_graph=True,
            )[0].norm()
            for wi, li in zip(w, branch_losses)
        ]
        G_stack = torch.stack(G)
        G_avg   = G_stack.mean().detach()

        r = torch.tensor(
            [li.item() / l0 for li, l0 in zip(branch_losses, self._initial_losses)],
            device=self.device,
        )
        r_hat   = (r / r.mean()).detach() ** self.alpha
        gn_loss = (G_stack - G_avg * r_hat).abs().sum()

        # --- Update model weights (retain graph for gn_loss.backward()) ---
        model_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        model_optimizer.step()

        # --- Update loss weights via GradNorm ---
        self.gn_optimizer.zero_grad()
        gn_loss.backward()
        self.gn_optimizer.step()

        # Renormalise
        with torch.no_grad():
            self.loss_weights.clamp_(min=1e-4)
            self.loss_weights.data.mul_(n / self.loss_weights.sum())

        acc     = (logits.detach().argmax(dim=1) == labels).float().mean().item()
        w_final = [round(float(wi), 4) for wi in w.detach()]
        return main_loss.item(), acc, w_final

    def grad_norms(self) -> dict[str, float]:
        """Mean gradient norm per branch (call after a training step)."""
        result = {}
        for name, branch_attr in zip(
            BRANCH_NAMES,
            ["semantic_branch", "affective_branch", "handcrafted_branch"],
        ):
            branch = getattr(self.model, branch_attr)
            norms  = [p.grad.norm().item() for p in branch.parameters() if p.grad is not None]
            result[name] = float(sum(norms) / len(norms)) if norms else 0.0
        return result

    def weight_str(self) -> str:
        """Human-readable current loss weights for logging."""
        w = self.loss_weights.detach().clamp(min=1e-4)
        w = w * len(BRANCH_NAMES) / w.sum()
        parts = [f"{n}={float(v):.3f}" for n, v in zip(BRANCH_NAMES, w)]
        return "w=[" + "  ".join(parts) + "]"

    def current_weights(self) -> list[float]:
        w = self.loss_weights.detach().clamp(min=1e-4)
        w = w * len(BRANCH_NAMES) / w.sum()
        return [round(float(v), 6) for v in w]
