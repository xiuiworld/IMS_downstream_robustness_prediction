from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def extract_delta_prediction(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "delta" not in outputs:
            raise RuntimeError("Model output dict must contain `delta`.")
        pred = outputs["delta"]
    else:
        pred = outputs
    if not isinstance(pred, torch.Tensor):
        raise RuntimeError("Model delta prediction must be a tensor.")
    if pred.ndim != 2 or pred.shape[-1] != 2:
        raise RuntimeError(f"Expected prediction shape [B, 2], got {tuple(pred.shape)}")
    return pred


def task_index_from_outputs(outputs: Any) -> int | None:
    if not isinstance(outputs, dict) or "task_index" not in outputs:
        return None
    task_index = int(outputs["task_index"])
    if task_index not in (0, 1):
        raise RuntimeError(f"Invalid task_index: {task_index}")
    return task_index


def task_name_from_index(task_index: int | None) -> str | None:
    if task_index is None:
        return None
    return ("delta_map", "delta_hota")[task_index]


def _select_task_if_needed(pred: torch.Tensor, target: torch.Tensor, task_index: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if task_index is None:
        return pred, target
    return pred[:, task_index : task_index + 1], target[:, task_index : task_index + 1]


def compute_zero_auxiliary_loss(
    outputs: Any,
    batch: dict[str, Any],
    zero_epsilon: float = 1e-8,
) -> torch.Tensor | None:
    if not isinstance(outputs, dict) or "zero_logits" not in outputs:
        return None
    pred = extract_delta_prediction(outputs)
    logits = outputs["zero_logits"]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Model zero_logits must be a tensor.")
    task_index = task_index_from_outputs(outputs)
    y_raw = batch["y_raw"].to(device=pred.device, dtype=pred.dtype)

    if task_index is not None:
        zero_target = (y_raw[:, task_index].abs() <= zero_epsilon).to(dtype=pred.dtype)
        if logits.ndim == 1:
            zero_logits = logits
        elif logits.ndim == 2 and logits.shape[1] == 1:
            zero_logits = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] == 2:
            zero_logits = logits[:, task_index]
        else:
            raise RuntimeError(
                f"Invalid zero_logits shape for single-task output: {tuple(logits.shape)}. "
                "Expected [B], [B, 1], or [B, 2]."
            )
    else:
        if logits.ndim == 1:
            zero_logits = logits
            zero_target = (y_raw.abs() <= zero_epsilon).all(dim=1).to(dtype=pred.dtype)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            zero_logits = logits[:, 0]
            zero_target = (y_raw.abs() <= zero_epsilon).all(dim=1).to(dtype=pred.dtype)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            zero_logits = logits
            zero_target = (y_raw.abs() <= zero_epsilon).to(dtype=pred.dtype)
        else:
            raise RuntimeError(
                f"Invalid zero_logits shape for multi-task output: {tuple(logits.shape)}. "
                "Expected [B], [B, 1], or [B, 2]."
            )

    if zero_logits.shape != zero_target.shape:
        raise RuntimeError(
            "zero_logits and zero_target shape mismatch: "
            f"logits={tuple(zero_logits.shape)}, target={tuple(zero_target.shape)}"
        )
    return F.binary_cross_entropy_with_logits(zero_logits, zero_target)


def compute_loss(
    outputs: Any,
    batch: dict[str, Any],
    loss_name: str = "huber",
    huber_beta: float = 1.0,
    zero_epsilon: float = 1e-8,
    zero_weight: float = 1.0,
    nonzero_weight: float = 2.0,
    zero_aux_weight: float = 0.0,
) -> torch.Tensor:
    pred = extract_delta_prediction(outputs)
    target = batch["y"].to(device=pred.device, dtype=pred.dtype)
    task_index = task_index_from_outputs(outputs)
    pred_loss, target_loss = _select_task_if_needed(pred, target, task_index)

    if loss_name == "huber":
        loss = F.smooth_l1_loss(pred_loss, target_loss, beta=huber_beta, reduction="mean")
    elif loss_name == "mse":
        loss = F.mse_loss(pred_loss, target_loss, reduction="mean")
    elif loss_name == "mae":
        loss = F.l1_loss(pred_loss, target_loss, reduction="mean")
    elif loss_name == "weighted_huber":
        raw_loss = F.smooth_l1_loss(pred_loss, target_loss, beta=huber_beta, reduction="none").mean(dim=1)
        y_raw = batch["y_raw"].to(device=pred.device, dtype=pred.dtype)
        if task_index is not None:
            y_raw = y_raw[:, task_index : task_index + 1]
        is_zero = (y_raw.abs() <= zero_epsilon).all(dim=1)
        weights = torch.where(
            is_zero,
            torch.full_like(raw_loss, float(zero_weight)),
            torch.full_like(raw_loss, float(nonzero_weight)),
        )
        loss = (raw_loss * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        raise RuntimeError(f"Unsupported loss: {loss_name}")

    if zero_aux_weight > 0:
        zero_aux_loss = compute_zero_auxiliary_loss(outputs, batch, zero_epsilon=zero_epsilon)
        if zero_aux_loss is not None:
            loss = loss + float(zero_aux_weight) * zero_aux_loss

    return loss


class UncertaintyWeightedHuberLoss(nn.Module):
    """Trainable uncertainty weighting for [delta_map, delta_hota] regression."""

    def __init__(self, huber_beta: float = 1.0):
        super().__init__()
        self.huber_beta = huber_beta
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        pred = extract_delta_prediction(outputs)
        target = batch["y"].to(device=pred.device, dtype=pred.dtype)
        task_index = task_index_from_outputs(outputs)
        per_element = F.smooth_l1_loss(pred, target, beta=self.huber_beta, reduction="none")
        if task_index is not None:
            per_task = per_element.mean(dim=0)[task_index : task_index + 1]
            precision = torch.exp(-self.log_vars[task_index : task_index + 1])
            return (precision * per_task + 0.5 * self.log_vars[task_index : task_index + 1]).sum()
        per_task = per_element.mean(dim=0)
        precision = torch.exp(-self.log_vars)
        return (precision * per_task + 0.5 * self.log_vars).sum()
