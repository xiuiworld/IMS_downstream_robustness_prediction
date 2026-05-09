from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ._validation import validate_type_id
from .visual_baseline import VisualBaseline


class FusionMultiTask(nn.Module):
    def __init__(
        self,
        num_types: int,
        type_embed_dim: int = 16,
        param_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 2,
        visual_backbone: str = "swin_tiny",
        visual_feature_dim: int = 256,
        allow_simple_fallback: bool = False,
        swin_pretrained: bool = False,
        freeze_early_layers: bool = False,
        swin_input_norm: bool | None = None,
    ):
        super().__init__()
        self.type_embed = nn.Embedding(
            num_embeddings=max(1, num_types),
            embedding_dim=type_embed_dim,
            padding_idx=0,
        )
        self.param_encoder = nn.Sequential(
            nn.Linear(type_embed_dim + 1, param_hidden_dim),
            nn.LayerNorm(param_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.visual_encoder = VisualBaseline(
            output_dim=visual_feature_dim,
            backbone=visual_backbone,
            feature_dim=visual_feature_dim,
            dropout=dropout,
            fallback_to_simple=allow_simple_fallback,
            swin_pretrained=swin_pretrained,
            freeze_early_layers=freeze_early_layers,
            swin_input_norm=swin_input_norm,
            with_head=False,
        )
        visual_dim = self.visual_encoder.feature_dim
        self.head = nn.Sequential(
            nn.Linear(param_hidden_dim + visual_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        type_id = validate_type_id(batch["type_id"], self.type_embed.num_embeddings)
        severity = batch["severity"].float().unsqueeze(-1)
        severity = torch.where(type_id.unsqueeze(-1) == 0, torch.zeros_like(severity), severity)
        param_feat = self.param_encoder(torch.cat([self.type_embed(type_id), severity], dim=-1))
        visual_feat = self.visual_encoder.forward_features(batch)
        return self.head(torch.cat([param_feat, visual_feat], dim=-1))
