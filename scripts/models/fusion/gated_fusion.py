# scripts/models/fusion/gated_fusion.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from scripts.config import (
    PROJECTION_DIM,     # 256
    NUM_LABELS,         # 6
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# GROUP PROJECTION MODULE
# =============================================================================

class GroupProjection(nn.Module):
    """
    Projects one feature group from its native dimension to PROJECTION_DIM.

    Architecture:
        Linear(input_dim → PROJECTION_DIM)
          → LayerNorm(PROJECTION_DIM)
          → GELU activation
          → Dropout

    Why this exact sequence?
        1. Linear: learns the projection matrix from raw features to 256-dim
        2. LayerNorm: normalizes so no group has systematically larger activations
        3. GELU: smooth nonlinearity (used in BERT, RoBERTa); better than ReLU
           for features that may be already normalized
        4. Dropout: regularization to prevent any group from dominating

    Why GELU and not ReLU?
        ReLU zeroes out all negative values, which can lose signal when our
        inputs may already be in a normalized range (like -2 to +2 from
        LayerNorm upstream in our feature extractors). GELU is smoother
        and preserves small negative signals. RoBERTa and similar models
        use GELU for exactly this reason.

    Why Linear (not MLP) for projection?
        The job of projection is dimensional alignment, not complex learning.
        A single linear layer is sufficient to find a good projection.
        Adding more layers here risks overfitting on small feature groups
        (like the 12-dim stylistic group projecting to 256-dim).
    """

    def __init__(self, input_dim: int, projection_dim: int = PROJECTION_DIM,
                 dropout: float = 0.1):
        """
        Input:
            input_dim      — native dimensionality of the group (e.g. 775 for semantic)
            projection_dim — target common dimension (default 256 from config)
            dropout        — dropout probability (default 0.1 — mild regularization)
        """
        super().__init__()

        self.projection = nn.Linear(input_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.activation = nn.GELU()
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x — tensor of shape (batch_size, input_dim)

        Output:
            tensor of shape (batch_size, projection_dim) — 256-dim by default
        """
        x = self.projection(x)    # Linear: input_dim → projection_dim
        x = self.layer_norm(x)    # LayerNorm: zero mean, unit variance per sample
        x = self.activation(x)    # GELU nonlinearity
        x = self.dropout(x)       # Dropout for regularization
        return x


# =============================================================================
# GATED FUSION MODULE
# =============================================================================

class GatedFusion(nn.Module):
    """
    Adaptive gating over 4 projected feature groups.

    Input:
        4 tensors of shape (batch_size, projection_dim)

    Output:
        (fused_features, gate_values)
        fused_features: (batch_size, 4 * projection_dim) — for classification
        gate_values:    (batch_size, num_groups)         — for interpretability

    How the gating works:
        1. Stack the 4 group embeddings into shape (batch, 4, 256)
        2. Apply a learned linear layer to produce 4 gate logits per sample
        3. Softmax over groups so gates sum to 1 per sample
        4. Multiply each group's embedding by its gate value
        5. Concatenate the 4 gated embeddings into (batch, 1024)

    Why softmax instead of sigmoid for gates?
        Sigmoid gates each group independently in [0, 1] but they can all be
        near 1 simultaneously — meaning no adaptive selection happens.
        Softmax forces the gates to sum to 1, so emphasizing one group
        necessarily de-emphasizes others. This is the correct semantics
        for adaptive selection.

    Why return gate_values?
        For interpretability — your thesis will analyze which group the
        model relies on most for each mental health class. Exposing the
        gate values makes this analysis trivial later.
    """

    def __init__(self, projection_dim: int = PROJECTION_DIM, num_groups: int = 4):
        """
        Input:
            projection_dim — dimension per group after projection (256)
            num_groups     — number of feature groups (4)
        """
        super().__init__()

        self.num_groups     = num_groups
        self.projection_dim = projection_dim

        # Gate network: takes concatenated groups → produces one gate per group
        # Input:  (batch, num_groups * projection_dim)  = (batch, 1024)
        # Output: (batch, num_groups)                    = (batch, 4)
        self.gate_network = nn.Linear(
            num_groups * projection_dim,
            num_groups
        )

    def forward(self, group_embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            group_embeddings — list of 4 tensors, each shape (batch, projection_dim)

        Output:
            fused: tensor of shape (batch, num_groups * projection_dim)
            gates: tensor of shape (batch, num_groups)
        """
        # Stack all groups along a new axis: shape (batch, num_groups, projection_dim)
        # torch.stack creates a new dimension at position 1
        stacked = torch.stack(group_embeddings, dim=1)

        # Flatten for gate computation: shape (batch, num_groups * projection_dim)
        # reshape(batch, -1) flattens the last two dims into one
        batch_size = stacked.size(0)
        flattened  = stacked.reshape(batch_size, -1)

        # Compute gate logits: shape (batch, num_groups)
        gate_logits = self.gate_network(flattened)

        # Softmax over groups (dim=1) — gates sum to 1 per sample
        gates = F.softmax(gate_logits, dim=1)   # shape (batch, num_groups)

        # Apply gates: multiply each group's embedding by its gate value
        # gates.unsqueeze(-1) shape: (batch, num_groups, 1)
        # stacked shape:              (batch, num_groups, projection_dim)
        # Broadcasting multiplies each group's full 256-dim vector by a scalar gate
        gated = stacked * gates.unsqueeze(-1)

        # Flatten back to (batch, num_groups * projection_dim) for the classifier
        fused = gated.reshape(batch_size, -1)

        return fused, gates


# =============================================================================
# COMPLETE FUSION NETWORK
# =============================================================================

class GatedFusionNetwork(nn.Module):
    """
    Full architecture combining projection + fusion + classification.

    Architecture flow:
        semantic   (batch, 775) ─→ projection ─→ (batch, 256) ┐
        affective  (batch, 34)  ─→ projection ─→ (batch, 256) │
        structural (batch, 15)  ─→ projection ─→ (batch, 256) ├─→ gated fusion ─→ (batch, 1024) ─→ classifier ─→ (batch, 6)
        stylistic  (batch, 12)  ─→ projection ─→ (batch, 256) ┘

    The classifier itself is a 2-layer MLP:
        Linear(1024 → 512) → GELU → Dropout → Linear(512 → 6)

    Why a 2-layer MLP classifier instead of a single Linear?
        The fused 1024-dim representation is rich. A single linear layer
        would only learn linearly-separable patterns in that space.
        A 2-layer MLP adds nonlinear capacity, which matters for the
        interactions between feature groups (e.g., "high fear AND low
        coherence together indicate anxiety more than either alone").
    """

    def __init__(
        self,
        semantic_dim:   int,
        affective_dim:  int,
        structural_dim: int,
        stylistic_dim:  int,
        projection_dim: int = PROJECTION_DIM,
        num_labels:     int = NUM_LABELS,
        classifier_hidden: int = 512,
        dropout:        float = 0.3,
    ):
        """
        Input:
            semantic_dim, affective_dim, structural_dim, stylistic_dim
                — native dimensions of each group (from config.combine.py).
            projection_dim      — common projection dimension (256).
            num_labels          — number of classes (6).
            classifier_hidden   — hidden dim in the classifier MLP (512).
            dropout             — dropout rate used in classifier (0.3).

        Why 0.1 dropout in projection but 0.3 in classifier?
            The projection step is shallow (just one linear layer per group)
            so heavy dropout would hurt learning. The classifier is deeper
            and closer to the output where overfitting risk is highest,
            so we use stronger regularization.
        """
        super().__init__()

        # One projection module per feature group.
        # Each knows its own input dimension but outputs 256.
        self.projections = nn.ModuleDict({
            "semantic":   GroupProjection(semantic_dim,   projection_dim),
            "affective":  GroupProjection(affective_dim,  projection_dim),
            "structural": GroupProjection(structural_dim, projection_dim),
            "stylistic":  GroupProjection(stylistic_dim,  projection_dim),
        })

        # Gated fusion module
        self.fusion = GatedFusion(projection_dim, num_groups=4)

        # Classification head: 2-layer MLP
        # Input: 4 * 256 = 1024 (fused vector)
        # Hidden: 512
        # Output: 6 (number of mental health classes)
        self.classifier = nn.Sequential(
            nn.Linear(4 * projection_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

    def forward(
        self,
        semantic:   torch.Tensor,
        affective:  torch.Tensor,
        structural: torch.Tensor,
        stylistic:  torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            4 feature tensors, one per group.
            return_gates — if True, also return the gate values for interpretability.

        Output:
            logits — tensor of shape (batch_size, num_labels)
            gates  — tensor of shape (batch_size, 4), only if return_gates=True

        Why the return_gates flag?
            During normal training we only need logits for loss computation.
            During evaluation and interpretability analysis we also need
            the gate values. One flag keeps the forward signature flexible
            without forcing every caller to handle the gates.
        """
        # Step 1: project each group to common dimension
        sem_proj = self.projections["semantic"](semantic)
        aff_proj = self.projections["affective"](affective)
        str_proj = self.projections["structural"](structural)
        sty_proj = self.projections["stylistic"](stylistic)

        # Step 2: gated fusion
        fused, gates = self.fusion([sem_proj, aff_proj, str_proj, sty_proj])

        # Step 3: classification
        logits = self.classifier(fused)

        if return_gates:
            return logits, gates
        return logits

    def get_projections(
        self,
        semantic:   torch.Tensor,
        affective:  torch.Tensor,
        structural: torch.Tensor,
        stylistic:  torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return the projected embeddings for each group separately.

        Used by GradNorm, which needs per-group gradient information
        to rebalance training weights. Not used during normal inference.

        Output:
            dict with keys "semantic", "affective", "structural", "stylistic"
            each mapping to a (batch_size, projection_dim) tensor.
        """
        return {
            "semantic":   self.projections["semantic"](semantic),
            "affective":  self.projections["affective"](affective),
            "structural": self.projections["structural"](structural),
            "stylistic":  self.projections["stylistic"](stylistic),
        }


# =============================================================================
# PARAMETER COUNT UTILITY
# =============================================================================

def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count trainable and total parameters in the model.

    Useful for your thesis — reviewers want to know model size.
    Also useful for debugging — if a layer is accidentally frozen,
    the trainable count will be wrong.

    Returns dict with:
        trainable — parameters that require gradients
        total     — all parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================

def _smoke_test():
    """
    Instantiate the model with expected dimensions and run a dummy forward pass.
    Verifies the architecture is wired correctly.

    Not run automatically — call manually for debugging:
        python -c "from scripts.models.fusion.gated_fusion import _smoke_test; _smoke_test()"
    """
    batch_size = 8

    model = GatedFusionNetwork(
        semantic_dim   = 775,
        affective_dim  = 34,
        structural_dim = 15,
        stylistic_dim  = 12,
    )

    semantic   = torch.randn(batch_size, 775)
    affective  = torch.randn(batch_size, 34)
    structural = torch.randn(batch_size, 15)
    stylistic  = torch.randn(batch_size, 12)

    # Forward pass without gates
    logits = model(semantic, affective, structural, stylistic)
    assert logits.shape == (batch_size, 6), f"Expected (8, 6), got {logits.shape}"

    # Forward pass with gates
    logits, gates = model(semantic, affective, structural, stylistic, return_gates=True)
    assert gates.shape == (batch_size, 4), f"Expected (8, 4), got {gates.shape}"
    # Gates should sum to 1 per sample (due to softmax)
    assert torch.allclose(gates.sum(dim=1), torch.ones(batch_size), atol=1e-5), \
        "Gates should sum to 1 per sample"

    param_counts = count_parameters(model)

    print("=" * 50)
    print("GatedFusionNetwork smoke test passed!")
    print(f"  logits shape: {logits.shape}")
    print(f"  gates shape:  {gates.shape}")
    print(f"  Gate values (first sample): {gates[0].tolist()}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Total parameters:     {param_counts['total']:,}")
    print("=" * 50)


if __name__ == "__main__":
    _smoke_test()