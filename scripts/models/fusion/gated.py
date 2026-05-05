import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock


class GatedFusion(nn.Module):
    """
    Three-branch gated baseline for comparison with LateConcatFusion.

    Each branch is projected to 256 with Linear -> LayerNorm -> tanh. Each
    projected branch has its own Linear -> sigmoid gate. Gates are stacked and
    softmaxed across branches per hidden dimension, then the fused vector is
    the weighted sum of the three 256-dim projected branches.
    """

    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        projection_dim: int = config.SEMANTIC_PROJECTION_DIM,
        num_labels: int = config.NUM_LABELS,
    ):
        super().__init__()
        self.semantic_branch = ProjectionBlock(
            semantic_dim, projection_dim, activation="tanh", dropout=0.0
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, projection_dim, activation="tanh", dropout=0.0
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim, projection_dim, activation="tanh", dropout=0.0
        )

        self.semantic_gate = nn.Linear(projection_dim, projection_dim)
        self.affective_gate = nn.Linear(projection_dim, projection_dim)
        self.handcrafted_gate = nn.Linear(projection_dim, projection_dim)
        self.classifier = ClassifierHead(projection_dim, hidden_dim=256, num_labels=num_labels)

    def get_branch_representations(
        self,
        semantic: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.semantic_branch(semantic),
            self.affective_branch(affective),
            self.handcrafted_branch(handcrafted),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        semantic_repr, affective_repr, handcrafted_repr = self.get_branch_representations(
            semantic, affective, handcrafted
        )

        gate_logits = torch.stack(
            [
                torch.sigmoid(self.semantic_gate(semantic_repr)),
                torch.sigmoid(self.affective_gate(affective_repr)),
                torch.sigmoid(self.handcrafted_gate(handcrafted_repr)),
            ],
            dim=1,
        )
        gates = F.softmax(gate_logits, dim=1)
        stacked = torch.stack([semantic_repr, affective_repr, handcrafted_repr], dim=1)
        fused = (gates * stacked).sum(dim=1)
        logits = self.classifier(fused)
        if return_gates:
            return logits, gates
        return logits
