import torch
import torch.nn as nn

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock


class LateConcatFusion(nn.Module):
    """
    Primary late-concatenation fusion architecture.

    Inputs:
        semantic: (B, 768)
        affective: (B, 34)
        handcrafted: (B, 26), where handcrafted is lexical(11) +
            syntactic(8) + structural(7).

    Architecture:
        semantic -> Linear(768, 256) -> LayerNorm -> GELU -> Dropout(0.1)
        affective -> Linear(34, 128) -> LayerNorm -> GELU -> Dropout(0.1)
        handcrafted -> Linear(26, 64) -> LayerNorm -> GELU -> Dropout(0.1)
        concat -> (B, 448) -> ClassifierHead(448 -> 256 -> 6)
    """

    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        semantic_projection_dim: int = config.SEMANTIC_PROJECTION_DIM,
        affective_projection_dim: int = config.AFFECTIVE_PROJECTION_DIM,
        handcrafted_projection_dim: int = config.HANDCRAFTED_PROJECTION_DIM,
        num_labels: int = config.NUM_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_branch = ProjectionBlock(
            semantic_dim, semantic_projection_dim, activation="gelu", dropout=dropout
        )
        self.affective_branch = ProjectionBlock(
            affective_dim, affective_projection_dim, activation="gelu", dropout=dropout
        )
        self.handcrafted_branch = ProjectionBlock(
            handcrafted_dim,
            handcrafted_projection_dim,
            activation="gelu",
            dropout=dropout,
        )

        fusion_dim = (
            semantic_projection_dim
            + affective_projection_dim
            + handcrafted_projection_dim
        )
        self.classifier = ClassifierHead(fusion_dim, hidden_dim=256, num_labels=num_labels)

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
    ) -> torch.Tensor:
        semantic_repr, affective_repr, handcrafted_repr = self.get_branch_representations(
            semantic, affective, handcrafted
        )
        fused = torch.cat([semantic_repr, affective_repr, handcrafted_repr], dim=1)
        return self.classifier(fused)
