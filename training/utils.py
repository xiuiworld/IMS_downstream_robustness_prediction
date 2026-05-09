from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def load_state_dict_head_compatible(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as strict_error:
        result = model.load_state_dict(state_dict, strict=False)

    allowed_unexpected = {
        "visual_encoder.head.weight",
        "visual_encoder.head.bias",
    }
    unexpected = set(result.unexpected_keys)
    missing = set(result.missing_keys)
    if missing or not unexpected.issubset(allowed_unexpected):
        raise strict_error


def require_non_empty_loader(loader: Any, split_name: str, allow_empty: bool = False) -> None:
    dataset = getattr(loader, "dataset", None)
    dataset_count = len(dataset) if dataset is not None else None
    try:
        loader_count = len(loader)
    except TypeError:
        loader_count = None
    if (dataset_count == 0 or loader_count == 0) and not allow_empty:
        raise RuntimeError(
            f"{split_name} split has no usable batches. "
            "Use the explicit allow-empty option only for smoke/debug runs."
        )
