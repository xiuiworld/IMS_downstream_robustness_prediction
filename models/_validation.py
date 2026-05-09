from __future__ import annotations

import torch


def validate_type_id(type_id: torch.Tensor, num_embeddings: int) -> torch.Tensor:
    type_id = type_id.long()
    invalid = (type_id < 0) | (type_id >= num_embeddings)
    if invalid.any():
        bad_values = type_id[invalid].detach().cpu().unique().tolist()
        raise RuntimeError(
            "type_id out of embedding range. "
            f"num_embeddings={num_embeddings}, bad_values={bad_values}"
        )
    return type_id
