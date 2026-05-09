from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SimpleVideoEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.net(video)


def _video_to_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise RuntimeError(f"Expected video tensor [B, T, C, H, W] or [B, C, T, H, W], got {tuple(video.shape)}")
    if video.shape[1] == 3:
        return video.float()
    if video.shape[2] == 3:
        return video.permute(0, 2, 1, 3, 4).contiguous().float()
    raise RuntimeError(f"Cannot infer channel dimension for video shape {tuple(video.shape)}")


def _get_swin_kinetics_norm() -> tuple[tuple[float, ...], tuple[float, ...]]:
    from torchvision.models.video import Swin3D_T_Weights

    transforms = Swin3D_T_Weights.KINETICS400_V1.transforms()
    mean = getattr(transforms, "mean", None)
    std = getattr(transforms, "std", None)
    if mean is None or std is None:
        raise RuntimeError("Cannot read mean/std from Swin3D pretrained weights transforms.")
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


class VisualBaseline(nn.Module):
    def __init__(
        self,
        output_dim: int = 2,
        backbone: str = "swin_tiny",
        feature_dim: int = 256,
        dropout: float = 0.2,
        fallback_to_simple: bool = False,
        swin_pretrained: bool = False,
        freeze_early_layers: bool = False,
        swin_input_norm: bool | None = None,
        with_head: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.uses_torchvision_swin = False
        self.normalize_for_swin = False

        if backbone == "swin_tiny":
            try:
                from torchvision.models.video import swin3d_t

                weights = None
                if swin_pretrained:
                    from torchvision.models.video import Swin3D_T_Weights

                    weights = Swin3D_T_Weights.KINETICS400_V1
                use_pretrained_norm = bool(swin_pretrained) if swin_input_norm is None else bool(swin_input_norm)

                self.encoder = swin3d_t(weights=weights)
                in_features = self.encoder.head.in_features
                self.encoder.head = nn.Identity()
                if use_pretrained_norm:
                    mean, std = _get_swin_kinetics_norm()
                    self.register_buffer(
                        "input_mean",
                        torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1, 1),
                        persistent=False,
                    )
                    self.register_buffer(
                        "input_std",
                        torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1, 1),
                        persistent=False,
                    )
                    self.normalize_for_swin = True
                if freeze_early_layers:
                    frozen_params = freeze_early_swin_layers(self.encoder)
                    if frozen_params == 0:
                        raise RuntimeError(
                            "freeze_early_layers=True, but no Swin parameters were frozen. "
                            "Check torchvision Swin parameter names and freeze_prefixes."
                        )
                self.proj = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, feature_dim),
                    nn.ReLU(inplace=True),
                )
                self.head = nn.Linear(feature_dim, output_dim) if with_head else None
                self.backbone_name = "swin_tiny"
                self.uses_torchvision_swin = True
                return
            except Exception as exc:
                if not fallback_to_simple:
                    raise RuntimeError(
                        "Failed to build Video Swin Tiny. Install a compatible torchvision build, "
                        "or explicitly use `--visual_backbone simple3d` for smoke/debug runs."
                    ) from exc

        if backbone != "simple3d" and not fallback_to_simple:
            raise RuntimeError(f"Unsupported visual backbone: {backbone}")

        self.encoder = SimpleVideoEncoder(feature_dim=feature_dim, dropout=dropout)
        self.proj = nn.Identity()
        self.head = nn.Linear(feature_dim, output_dim) if with_head else None
        self.backbone_name = "simple3d"

    def _normalize_video_if_needed(self, video: torch.Tensor) -> torch.Tensor:
        if not self.normalize_for_swin:
            return video
        mean = self.input_mean.to(device=video.device, dtype=video.dtype)
        std = self.input_std.to(device=video.device, dtype=video.dtype)
        return (video - mean) / std

    def forward_features(self, batch: dict[str, Any]) -> torch.Tensor:
        video = _video_to_bcthw(batch["video"])
        video = self._normalize_video_if_needed(video)
        return self.proj(self.encoder(video))

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("VisualBaseline was created with with_head=False; use forward_features().")
        return self.head(self.forward_features(batch))


def freeze_early_swin_layers(
    encoder: nn.Module,
    freeze_prefixes: tuple[str, ...] = ("patch_embed", "features.0", "features.1"),
) -> int:
    frozen = 0
    for name, param in encoder.named_parameters():
        if name.startswith(freeze_prefixes):
            param.requires_grad = False
            frozen += param.numel()
    return frozen
