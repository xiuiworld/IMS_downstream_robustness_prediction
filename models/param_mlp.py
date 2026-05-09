from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ._validation import validate_type_id


class ParamMLP(nn.Module):
    def __init__(
        self,
        num_types: int,
        type_embed_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        super().__init__()
        self.type_embed = nn.Embedding(
            num_embeddings=max(1, num_types),
            embedding_dim=type_embed_dim,
            padding_idx=0,
        )
        self.net = nn.Sequential(
            nn.Linear(type_embed_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        type_id = validate_type_id(batch["type_id"], self.type_embed.num_embeddings)
        severity = batch["severity"].float().unsqueeze(-1)
        severity = torch.where(type_id.unsqueeze(-1) == 0, torch.zeros_like(severity), severity)
        x = torch.cat([self.type_embed(type_id), severity], dim=-1)
        return self.net(x)
