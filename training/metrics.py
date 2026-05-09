from __future__ import annotations

import math
from typing import Any, Dict

import torch


TARGET_NAMES = ("delta_map", "delta_hota")


def _to_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().float()


def _require_label_stat(normalization_stats: Dict[str, Any], target_name: str, field: str) -> float:
    label_stats = normalization_stats.get("label_stats")
    if not isinstance(label_stats, dict):
        raise RuntimeError("normalization_stats missing `label_stats`.")

    target_stats = label_stats.get(target_name)
    if not isinstance(target_stats, dict):
        raise RuntimeError(f"normalization_stats missing `label_stats.{target_name}`.")

    if field not in target_stats:
        raise RuntimeError(f"normalization_stats missing `{target_name}.{field}`.")

    value = float(target_stats[field])
    if not math.isfinite(value):
        raise RuntimeError(f"Invalid normalization stat: {target_name}.{field}={value}")
    return value


def denormalize_delta(pred: torch.Tensor, normalization_stats: Dict[str, Any], target_mode: str) -> torch.Tensor:
    if target_mode == "raw":
        return pred
    if target_mode != "zscore":
        raise RuntimeError(f"Unsupported target_mode: {target_mode}")
    mean = torch.tensor(
        [
            _require_label_stat(normalization_stats, "delta_map", "mean"),
            _require_label_stat(normalization_stats, "delta_hota", "mean"),
        ],
        device=pred.device,
        dtype=pred.dtype,
    )
    std = torch.tensor(
        [
            _require_label_stat(normalization_stats, "delta_map", "std"),
            _require_label_stat(normalization_stats, "delta_hota", "std"),
        ],
        device=pred.device,
        dtype=pred.dtype,
    )
    if torch.any(std <= 0):
        raise RuntimeError(f"Invalid normalization std: {std.tolist()}")
    return pred * std + mean


def _rankdata_average(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    n = values.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    if torch.allclose(x, x[0]) or torch.allclose(y, y[0]):
        return None
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = torch.sqrt((vx * vx).sum() * (vy * vy).sum())
    if float(denom) == 0.0:
        return None
    value = float((vx * vy).sum() / denom)
    return value if math.isfinite(value) else None


def _float_or_none(value: torch.Tensor) -> float | None:
    out = float(value)
    return out if math.isfinite(out) else None


def _compute_basic(pred: torch.Tensor, target: torch.Tensor, prefix: str) -> Dict[str, float | None]:
    if pred.numel() == 0:
        return {
            f"{prefix}count": 0.0,
            f"{prefix}rmse_map": None,
            f"{prefix}rmse_hota": None,
            f"{prefix}rmse_mean": None,
            f"{prefix}mae_map": None,
            f"{prefix}mae_hota": None,
            f"{prefix}mae_mean": None,
            f"{prefix}spearman_map": None,
            f"{prefix}spearman_hota": None,
            f"{prefix}spearman_mean": None,
        }
    error = pred - target
    rmse = torch.sqrt((error * error).mean(dim=0))
    mae = error.abs().mean(dim=0)
    s_map = _spearman(pred[:, 0], target[:, 0])
    s_hota = _spearman(pred[:, 1], target[:, 1])
    s_vals = [x for x in (s_map, s_hota) if x is not None]
    return {
        f"{prefix}count": float(pred.shape[0]),
        f"{prefix}rmse_map": _float_or_none(rmse[0]),
        f"{prefix}rmse_hota": _float_or_none(rmse[1]),
        f"{prefix}rmse_mean": _float_or_none(rmse.mean()),
        f"{prefix}mae_map": _float_or_none(mae[0]),
        f"{prefix}mae_hota": _float_or_none(mae[1]),
        f"{prefix}mae_mean": _float_or_none(mae.mean()),
        f"{prefix}spearman_map": s_map,
        f"{prefix}spearman_hota": s_hota,
        f"{prefix}spearman_mean": sum(s_vals) / len(s_vals) if s_vals else None,
    }


def compute_metrics(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    ood_mask: torch.Tensor | None = None,
    zero_epsilon: float = 1e-8,
    zero_pred_threshold: float = 1e-3,
    prefix: str = "",
) -> Dict[str, float | None]:
    pred = _to_cpu_float(pred_raw)
    target = _to_cpu_float(target_raw)
    out = _compute_basic(pred, target, prefix)

    zero_target = (target.abs() <= zero_epsilon).all(dim=1)
    nonzero_target = ~zero_target
    zero_pred = (pred.abs() <= zero_pred_threshold).all(dim=1)
    out[f"{prefix}zero_target_ratio"] = _float_or_none(zero_target.float().mean()) if len(zero_target) else None
    out[f"{prefix}zero_pred_ratio"] = _float_or_none(zero_pred.float().mean()) if len(zero_pred) else None

    subset_metric_suffixes = (
        "count",
        "rmse_map",
        "rmse_hota",
        "rmse_mean",
        "mae_map",
        "mae_hota",
        "mae_mean",
    )
    for label, mask in (("zero", zero_target), ("nonzero", nonzero_target)):
        subset = _compute_basic(pred[mask], target[mask], f"{prefix}{label}_")
        out.update({k: v for k, v in subset.items() if k.endswith(subset_metric_suffixes)})

    if ood_mask is not None:
        ood = _to_cpu_float(ood_mask).bool()
        if ood.any():
            ood_metrics = compute_metrics(
                pred[ood],
                target[ood],
                ood_mask=None,
                zero_epsilon=zero_epsilon,
                zero_pred_threshold=zero_pred_threshold,
                prefix=f"{prefix}ood_",
            )
            out.update(ood_metrics)
        else:
            out.update(_compute_basic(pred[:0], target[:0], f"{prefix}ood_"))
            out[f"{prefix}ood_zero_target_ratio"] = None
            out[f"{prefix}ood_zero_pred_ratio"] = None
    return out
