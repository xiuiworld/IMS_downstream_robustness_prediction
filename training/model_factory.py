from __future__ import annotations

from argparse import Namespace
from typing import Any

from models import FusionMultiTask, FusionSingleTask, ParamMLP, VisualBaseline, VisualSingleTask
from training.dataset import type_vocab_size


MODEL_TO_MODALITY = {
    "param_mlp": "param_only",
    "visual_baseline": "visual_only",
    "fusion_multitask": "fusion",
    "visual_single_task_map": "visual_only",
    "visual_single_task_hota": "visual_only",
    "fusion_single_task_map": "fusion",
    "fusion_single_task_hota": "fusion",
}

MODEL_TO_TASK = {
    "visual_single_task_map": "delta_map",
    "visual_single_task_hota": "delta_hota",
    "fusion_single_task_map": "delta_map",
    "fusion_single_task_hota": "delta_hota",
}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def create_model(model_name: str, args: Namespace | dict[str, Any], normalization_stats: dict[str, Any]):
    cfg = vars(args) if isinstance(args, Namespace) else dict(args)
    hidden_dim = int(cfg.get("hidden_dim", 128))
    dropout = float(cfg.get("dropout", 0.2))
    output_dim = int(cfg.get("output_dim", 2))

    if model_name == "param_mlp":
        return ParamMLP(
            num_types=type_vocab_size(normalization_stats),
            type_embed_dim=int(cfg.get("type_embed_dim", 8)),
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
        )
    if model_name == "visual_baseline":
        return VisualBaseline(
            output_dim=output_dim,
            backbone=str(cfg.get("visual_backbone", "swin_tiny")),
            feature_dim=int(cfg.get("visual_feature_dim", 256)),
            dropout=dropout,
            fallback_to_simple=bool(cfg.get("allow_simple_fallback", False)),
            swin_pretrained=bool(cfg.get("swin_pretrained", False)),
            freeze_early_layers=bool(cfg.get("freeze_early_layers", False)),
            swin_input_norm=_optional_bool(cfg.get("swin_input_norm", None)),
        )
    if model_name in {"visual_single_task_map", "visual_single_task_hota"}:
        return VisualSingleTask(
            task=MODEL_TO_TASK[model_name],
            backbone=str(cfg.get("visual_backbone", "swin_tiny")),
            feature_dim=int(cfg.get("visual_feature_dim", 256)),
            dropout=dropout,
            allow_simple_fallback=bool(cfg.get("allow_simple_fallback", False)),
            swin_pretrained=bool(cfg.get("swin_pretrained", False)),
            freeze_early_layers=bool(cfg.get("freeze_early_layers", False)),
            swin_input_norm=_optional_bool(cfg.get("swin_input_norm", None)),
        )
    if model_name == "fusion_multitask":
        return FusionMultiTask(
            num_types=type_vocab_size(normalization_stats),
            type_embed_dim=int(cfg.get("type_embed_dim", 16)),
            param_hidden_dim=int(cfg.get("param_hidden_dim", 64)),
            fusion_hidden_dim=int(cfg.get("fusion_hidden_dim", 256)),
            dropout=dropout,
            output_dim=output_dim,
            visual_backbone=str(cfg.get("visual_backbone", "swin_tiny")),
            visual_feature_dim=int(cfg.get("visual_feature_dim", 256)),
            allow_simple_fallback=bool(cfg.get("allow_simple_fallback", False)),
            swin_pretrained=bool(cfg.get("swin_pretrained", False)),
            freeze_early_layers=bool(cfg.get("freeze_early_layers", False)),
            swin_input_norm=_optional_bool(cfg.get("swin_input_norm", None)),
        )
    if model_name in {"fusion_single_task_map", "fusion_single_task_hota"}:
        return FusionSingleTask(
            task=MODEL_TO_TASK[model_name],
            num_types=type_vocab_size(normalization_stats),
            type_embed_dim=int(cfg.get("type_embed_dim", 16)),
            param_hidden_dim=int(cfg.get("param_hidden_dim", 64)),
            fusion_hidden_dim=int(cfg.get("fusion_hidden_dim", 256)),
            dropout=dropout,
            visual_backbone=str(cfg.get("visual_backbone", "swin_tiny")),
            visual_feature_dim=int(cfg.get("visual_feature_dim", 256)),
            allow_simple_fallback=bool(cfg.get("allow_simple_fallback", False)),
            swin_pretrained=bool(cfg.get("swin_pretrained", False)),
            freeze_early_layers=bool(cfg.get("freeze_early_layers", False)),
            swin_input_norm=_optional_bool(cfg.get("swin_input_norm", None)),
        )
    raise RuntimeError(f"Unsupported model: {model_name}")
