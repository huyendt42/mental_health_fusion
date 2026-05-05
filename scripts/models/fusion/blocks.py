import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import config


class ProjectionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.linear(x))
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return self.dropout(x)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_labels: int = config.NUM_LABELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
