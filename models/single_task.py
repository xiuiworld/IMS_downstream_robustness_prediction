from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ._validation import validate_type_id
from .visual_baseline import VisualBaseline


TASK_TO_INDEX = {
    "delta_map": 0,
    "delta_hota": 1,
}


def task_index(task: str) -> int:
    if task not in TASK_TO_INDEX:
        raise RuntimeError(f"Unsupported single-task target: {task}")
    return TASK_TO_INDEX[task]


def single_task_output(value: torch.Tensor, task: str) -> dict[str, Any]:
    idx = task_index(task)
    delta = value.new_zeros((value.shape[0], 2))
    delta[:, idx] = value.squeeze(-1)
    return {"delta": delta, "task": task, "task_index": idx}


class VisualSingleTask(nn.Module):
    def __init__(
        self,
        task: str,
        backbone: str = "swin_tiny",
        feature_dim: int = 256,
        dropout: float = 0.2,
        allow_simple_fallback: bool = False,
        swin_pretrained: bool = False,
        freeze_early_layers: bool = False,
        swin_input_norm: bool | None = None,
    ):
        super().__init__()
        self.task = task
        self.visual_encoder = VisualBaseline(
            output_dim=feature_dim,
            backbone=backbone,
            feature_dim=feature_dim,
            dropout=dropout,
            fallback_to_simple=allow_simple_fallback,
            swin_pretrained=swin_pretrained,
            freeze_early_layers=freeze_early_layers,
            swin_input_norm=swin_input_norm,
            with_head=False,
        )
        self.feature_dim = self.visual_encoder.feature_dim
        self.backbone_name = self.visual_encoder.backbone_name
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        feat = self.visual_encoder.forward_features(batch)
        return single_task_output(self.head(feat), self.task)


class FusionSingleTask(nn.Module):
    def __init__(
        self,
        task: str,
        num_types: int,
        type_embed_dim: int = 16,
        param_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        visual_backbone: str = "swin_tiny",
        visual_feature_dim: int = 256,
        allow_simple_fallback: bool = False,
        swin_pretrained: bool = False,
        freeze_early_layers: bool = False,
        swin_input_norm: bool | None = None,
    ):
        super().__init__()
        self.task = task
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
        self.backbone_name = self.visual_encoder.backbone_name
        self.head = nn.Sequential(
            nn.Linear(param_hidden_dim + visual_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        type_id = validate_type_id(batch["type_id"], self.type_embed.num_embeddings)
        severity = batch["severity"].float().unsqueeze(-1)
        severity = torch.where(type_id.unsqueeze(-1) == 0, torch.zeros_like(severity), severity)
        param_feat = self.param_encoder(torch.cat([self.type_embed(type_id), severity], dim=-1))
        visual_feat = self.visual_encoder.forward_features(batch)
        value = self.head(torch.cat([param_feat, visual_feat], dim=-1))
        return single_task_output(value, self.task)
