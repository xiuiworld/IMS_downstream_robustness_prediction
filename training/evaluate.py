from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from training.dataset import build_dataloaders, load_normalization_stats
from training.losses import extract_delta_prediction, task_index_from_outputs, task_name_from_index
from training.metrics import compute_metrics, denormalize_delta
from training.model_factory import MODEL_TO_MODALITY, create_model
from training.utils import load_state_dict_head_compatible, move_batch, require_non_empty_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a surrogate robustness model checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_TO_MODALITY), default=None)
    parser.add_argument("--modality", choices=("param_only", "visual_only", "fusion"), default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_integrity", action="store_true", help="Skip cache fingerprint/hash validation for quick smoke tests.")
    parser.add_argument("--allow_empty_eval", action="store_true", help="Allow an empty evaluation split only for smoke/debug runs.")
    parser.add_argument("--allow_run_id_override", action="store_true", help="Allow evaluating a checkpoint against a different run_id artifact.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during evaluation.")
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return checkpoint


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "original_clip_id",
        "split",
        "is_ood_masked",
        "target_delta_map",
        "target_delta_hota",
        "pred_delta_map",
        "pred_delta_hota",
        "abs_error_delta_map",
        "abs_error_delta_hota",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_trained_swin_input_norm(ckpt_model_args: dict[str, Any]) -> bool:
    visual_backbone = str(ckpt_model_args.get("visual_backbone", "swin_tiny"))
    if visual_backbone != "swin_tiny":
        return False
    raw_norm = ckpt_model_args.get("swin_input_norm", None)
    if raw_norm is None:
        return bool(ckpt_model_args.get("swin_pretrained", False))
    return bool(raw_norm)


def normalize_run_id(run_id: Any) -> str | None:
    if run_id is None:
        return None
    value = str(run_id).strip()
    if value == "" or value.lower() == "canonical":
        return None
    return value


def resolve_eval_run_id(cli_run_id: str | None, checkpoint_run_id: Any, allow_override: bool) -> str | None:
    ckpt_run_id = normalize_run_id(checkpoint_run_id)
    if cli_run_id is None:
        return ckpt_run_id

    requested_run_id = normalize_run_id(cli_run_id)
    if requested_run_id != ckpt_run_id and not allow_override:
        raise RuntimeError(
            "run_id override is unsafe: "
            f"checkpoint run_id={ckpt_run_id or 'canonical'}, "
            f"cli run_id={requested_run_id or 'canonical'}. "
            "Use --allow_run_id_override only for explicit smoke/debug checks."
        )
    return requested_run_id


def resolve_eval_modality(model_name: str, cli_modality: str | None, checkpoint_modality: Any) -> str:
    expected = MODEL_TO_MODALITY[model_name]
    modality = str(cli_modality or checkpoint_modality or expected)
    if modality != expected:
        raise RuntimeError(
            f"Model/modality mismatch during evaluation: model={model_name} "
            f"requires modality={expected}, but got modality={modality}."
        )
    return modality


def effective_model_args_for_config(model_args: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "visual_backbone",
        "visual_feature_dim",
        "swin_pretrained",
        "swin_input_norm",
        "freeze_early_layers",
        "allow_simple_fallback",
        "type_embed_dim",
        "hidden_dim",
        "dropout",
        "output_dim",
    )
    return {key: model_args.get(key) for key in keys}


def active_subset_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_index: int,
    zero_epsilon: float = 1e-8,
) -> dict[str, float | None]:
    pred_active = pred[:, task_index]
    target_active = target[:, task_index]
    zero_mask = target_active.abs() <= zero_epsilon
    nonzero_mask = ~zero_mask

    def subset(prefix: str, mask: torch.Tensor) -> dict[str, float | None]:
        count = int(mask.sum().item())
        if count == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_rmse": None,
                f"{prefix}_mae": None,
            }
        error = pred_active[mask] - target_active[mask]
        return {
            f"{prefix}_count": float(count),
            f"{prefix}_rmse": float(torch.sqrt((error * error).mean())),
            f"{prefix}_mae": float(error.abs().mean()),
        }

    return {
        **subset("active_zero", zero_mask),
        **subset("active_nonzero", nonzero_mask),
    }


def hide_inactive_task_metrics(metrics: dict[str, Any], active_task_index: int) -> None:
    inactive_suffix = "hota" if active_task_index == 0 else "map"
    for prefix in ("", "ood_", "zero_", "nonzero_"):
        for name in ("rmse", "mae", "spearman"):
            key = f"{prefix}{name}_{inactive_suffix}"
            if key in metrics:
                metrics[key] = None
    metrics["rmse_mean"] = None
    metrics["mae_mean"] = None
    metrics["spearman_mean"] = None


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    ckpt_model_args = checkpoint.get("model_args", {})
    if not isinstance(ckpt_model_args, dict):
        ckpt_model_args = {}
    model_name = args.model or checkpoint.get("model_name")
    if model_name not in MODEL_TO_MODALITY:
        raise RuntimeError(f"Invalid or missing model_name in checkpoint: {model_name}")
    model_name = str(model_name)
    modality = resolve_eval_modality(
        model_name=model_name,
        cli_modality=args.modality,
        checkpoint_modality=checkpoint.get("modality"),
    )
    target_mode = str(checkpoint.get("target_mode", ckpt_model_args.get("target_mode", "zscore")))
    checkpoint_run_id = ckpt_model_args.get("run_id")
    run_id = resolve_eval_run_id(args.run_id, checkpoint_run_id, args.allow_run_id_override)
    model_args = dict(ckpt_model_args)
    model_args.update({"model": model_name, "modality": modality})
    trained_swin_input_norm = infer_trained_swin_input_norm(ckpt_model_args)
    # Evaluation immediately loads checkpoint weights, so avoid network/cache-dependent
    # pretrained weight loading while reconstructing the architecture. Keep the
    # training-time input normalization setting because it affects the forward pass.
    model_args["swin_pretrained"] = False
    model_args["swin_input_norm"] = trained_swin_input_norm
    model_args["freeze_early_layers"] = False

    normalization_stats = checkpoint.get("normalization_stats")
    if not isinstance(normalization_stats, dict):
        normalization_stats = load_normalization_stats(config_path=args.config, run_id=run_id)

    device = torch.device(args.device)
    model = create_model(model_name, model_args, normalization_stats).to(device)
    load_state_dict_head_compatible(model, checkpoint["model_state"])
    model.eval()

    loader = build_dataloaders(
        modality=modality,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode=target_mode,
        run_id=run_id,
        config_path=args.config,
        validate_integrity=not args.skip_integrity,
        shuffle=False,
    )
    require_non_empty_loader(loader, args.split, allow_empty=args.allow_empty_eval)

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    active_task_index: int | None = None
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_device = move_batch(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(batch_device)
            batch_task_index = task_index_from_outputs(outputs)
            if batch_task_index is not None:
                if active_task_index is None:
                    active_task_index = batch_task_index
                elif active_task_index != batch_task_index:
                    raise RuntimeError("Single-task model changed task_index during evaluation.")
            pred = denormalize_delta(extract_delta_prediction(outputs).float(), normalization_stats, target_mode).cpu()
            target = batch["y_raw"].cpu()
            preds.append(pred)
            targets.append(target)
            if "is_ood_masked" in batch:
                masks.append(batch["is_ood_masked"].cpu())
            else:
                masks.append(torch.zeros(target.shape[0], dtype=torch.long))

            for i in range(target.shape[0]):
                is_ood = int(masks[-1][i].item())
                rows.append(
                    {
                        "clip_id": batch["clip_id"][i],
                        "original_clip_id": batch["original_clip_id"][i],
                        "split": batch["split"][i],
                        "is_ood_masked": is_ood,
                        "target_delta_map": float(target[i, 0]),
                        "target_delta_hota": float(target[i, 1]),
                        "pred_delta_map": float(pred[i, 0]),
                        "pred_delta_hota": float(pred[i, 1]),
                        "abs_error_delta_map": float((pred[i, 0] - target[i, 0]).abs()),
                        "abs_error_delta_hota": float((pred[i, 1] - target[i, 1]).abs()),
                    }
                )
            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break

    if preds:
        pred_all = torch.cat(preds, dim=0)
        target_all = torch.cat(targets, dim=0)
        mask_all = torch.cat(masks, dim=0)
        metrics = compute_metrics(pred_all, target_all, ood_mask=mask_all)
        if active_task_index is not None:
            active_task = task_name_from_index(active_task_index)
            suffix = "map" if active_task_index == 0 else "hota"
            metrics.update(
                {
                    "active_task": active_task,
                    "active_task_index": float(active_task_index),
                    "active_rmse": metrics.get(f"rmse_{suffix}"),
                    "active_mae": metrics.get(f"mae_{suffix}"),
                    "active_spearman": metrics.get(f"spearman_{suffix}"),
                    "active_ood_rmse": metrics.get(f"ood_rmse_{suffix}"),
                    "active_zero_rmse": metrics.get(f"zero_rmse_{suffix}"),
                    "active_nonzero_rmse": metrics.get(f"nonzero_rmse_{suffix}"),
                    **active_subset_metrics(pred_all, target_all, active_task_index),
                }
            )
            hide_inactive_task_metrics(metrics, active_task_index)
    else:
        metrics = {"count": 0.0, "rmse_mean": None, "mae_mean": None, "spearman_mean": None}

    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"eval_{args.split}"
    save_json(out_dir / f"{args.split}_metrics.json", metrics)
    save_json(
        out_dir / "eval_config.json",
        {
            "checkpoint": str(checkpoint_path),
            "model": model_name,
            "modality": modality,
            "split": args.split,
            "target_mode": target_mode,
            "run_id": run_id,
            "checkpoint_run_id": normalize_run_id(checkpoint_run_id),
            "cli_run_id": normalize_run_id(args.run_id),
            "run_id_override_allowed": bool(args.allow_run_id_override),
            "run_id_override_used": normalize_run_id(args.run_id) != normalize_run_id(checkpoint_run_id)
            if args.run_id is not None
            else False,
            "active_task": task_name_from_index(active_task_index),
            "amp": bool(args.amp),
            "max_batches": args.max_batches,
            "effective_model_args": effective_model_args_for_config(model_args),
        },
    )
    write_predictions(out_dir / f"{args.split}_predictions.csv", rows)
    print(f"[DONE] metrics: {out_dir / f'{args.split}_metrics.json'}")
    print(f"[DONE] predictions: {out_dir / f'{args.split}_predictions.csv'}")


if __name__ == "__main__":
    main()
