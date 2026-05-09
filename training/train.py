from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from training.dataset import build_dataloaders, load_normalization_stats
from training.losses import (
    UncertaintyWeightedHuberLoss,
    compute_loss,
    compute_zero_auxiliary_loss,
    extract_delta_prediction,
    task_index_from_outputs,
    task_name_from_index,
)
from training.metrics import compute_metrics, denormalize_delta
from training.model_factory import MODEL_TO_MODALITY, create_model
from training.utils import move_batch, require_non_empty_loader, save_json, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_MODEL_NAMES = {
    "visual_baseline",
    "fusion_multitask",
    "visual_single_task_map",
    "visual_single_task_hota",
    "fusion_single_task_map",
    "fusion_single_task_hota",
}
PARAM_MODEL_NAMES = {"param_mlp", "fusion_multitask", "fusion_single_task_map", "fusion_single_task_hota"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a surrogate robustness model.")
    parser.add_argument("--model", choices=tuple(MODEL_TO_MODALITY), required=True)
    parser.add_argument("--modality", choices=("param_only", "visual_only", "fusion"), default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--target_mode", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--loss", choices=("huber", "weighted_huber", "mse", "mae", "uncertainty_huber"), default="huber")
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--zero_weight", type=float, default=1.0)
    parser.add_argument("--nonzero_weight", type=float, default=2.0)
    parser.add_argument("--zero_aux_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--visual_backbone", choices=("swin_tiny", "simple3d"), default="swin_tiny")
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--allow_simple_fallback", action="store_true", help="Allow fallback from Video Swin to Simple3D only for smoke/debug runs.")
    parser.add_argument("--swin_pretrained", action="store_true", help="Use torchvision Kinetics-400 pretrained weights for Video Swin Tiny.")
    parser.add_argument("--freeze_early_layers", action="store_true", help="Freeze early Video Swin layers when using the Swin backbone.")
    parser.add_argument(
        "--swin_input_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply Kinetics-400 Swin input normalization. Defaults to --swin_pretrained.",
    )
    parser.add_argument("--type_embed_dim", type=int, default=16)
    parser.add_argument("--param_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--skip_integrity", action="store_true", help="Skip cache fingerprint/hash validation for quick smoke tests.")
    parser.add_argument("--allow_empty_val", action="store_true", help="Allow an empty validation split only for smoke/debug runs.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def default_experiment_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id or "canonical"
    parts = [
        args.model,
        f"run-{run_id}",
        f"seed-{args.seed}",
        f"target-{args.target_mode}",
        f"loss-{args.loss}",
    ]
    if args.model in VISUAL_MODEL_NAMES:
        parts.extend(
            [
                f"backbone-{args.visual_backbone}",
                f"vfeat-{args.visual_feature_dim}",
                f"pretrained-{int(args.swin_pretrained)}",
                f"freeze-{int(args.freeze_early_layers)}",
                f"norm-{int(bool(args.swin_input_norm))}",
            ]
        )
    if args.model in PARAM_MODEL_NAMES:
        parts.append(f"typeemb-{args.type_embed_dim}")
    name = "__".join(parts)
    return PROJECT_ROOT / "experiments" / name


def get_actual_visual_backbone(model: torch.nn.Module) -> str | None:
    if hasattr(model, "backbone_name"):
        return str(getattr(model, "backbone_name"))
    visual_encoder = getattr(model, "visual_encoder", None)
    if visual_encoder is not None and hasattr(visual_encoder, "backbone_name"):
        return str(getattr(visual_encoder, "backbone_name"))
    return None


def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total_params": sum(p.numel() for p in module.parameters()),
        "trainable_params": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }


def model_summary(model: torch.nn.Module) -> dict[str, Any]:
    summary: dict[str, Any] = count_parameters(model)
    visual_encoder = getattr(model, "visual_encoder", None)
    if visual_encoder is not None:
        summary.update({f"visual_{key}": value for key, value in count_parameters(visual_encoder).items()})
        encoder = getattr(visual_encoder, "encoder", None)
        if isinstance(encoder, torch.nn.Module):
            summary.update({f"visual_backbone_{key}": value for key, value in count_parameters(encoder).items()})
    return summary


def finalize_visual_options(args: argparse.Namespace) -> None:
    """Resolve implicit visual options into checkpoint-safe explicit values."""
    if args.model not in VISUAL_MODEL_NAMES:
        return
    if args.visual_backbone != "swin_tiny":
        args.swin_pretrained = False
        args.freeze_early_layers = False
        args.swin_input_norm = False
        return
    if args.swin_input_norm is None:
        args.swin_input_norm = bool(args.swin_pretrained)


def resolve_and_validate_modality(args: argparse.Namespace) -> None:
    expected = MODEL_TO_MODALITY[args.model]
    if args.modality is None:
        args.modality = expected
        return
    if args.modality != expected:
        raise RuntimeError(
            f"Model/modality mismatch: model={args.model} requires modality={expected}, "
            f"but got modality={args.modality}. Do not override modality in canonical experiments."
        )


def compute_training_loss(outputs: Any, batch: dict[str, Any], args: argparse.Namespace, loss_module: torch.nn.Module | None) -> torch.Tensor:
    if loss_module is not None:
        loss = loss_module(outputs, batch)
        if args.zero_aux_weight > 0:
            zero_aux_loss = compute_zero_auxiliary_loss(outputs, batch)
            if zero_aux_loss is not None:
                loss = loss + float(args.zero_aux_weight) * zero_aux_loss
        return loss
    return compute_loss(
        outputs,
        batch,
        loss_name=args.loss,
        huber_beta=args.huber_beta,
        zero_weight=args.zero_weight,
        nonzero_weight=args.nonzero_weight,
        zero_aux_weight=args.zero_aux_weight,
    )


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


def evaluate_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    normalization_stats: dict[str, Any],
    loss_module: torch.nn.Module | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float | None]]:
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    active_task_index: int | None = None
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(batch)
                loss = compute_training_loss(outputs, batch, args, loss_module)
            batch_task_index = task_index_from_outputs(outputs)
            if batch_task_index is not None:
                if active_task_index is None:
                    active_task_index = batch_task_index
                elif active_task_index != batch_task_index:
                    raise RuntimeError("Single-task model changed task_index during validation.")
            pred = extract_delta_prediction(outputs).float()
            preds.append(denormalize_delta(pred, normalization_stats, args.target_mode).cpu())
            targets.append(batch["y_raw"].detach().cpu())
            if "is_ood_masked" in batch:
                masks.append(batch["is_ood_masked"].detach().cpu())
            losses.append(float(loss.detach().cpu()))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    if not preds:
        return float("nan"), {"count": 0.0, "loss": None, "rmse_mean": None}

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    mask_all = torch.cat(masks, dim=0) if masks else None
    metrics = compute_metrics(pred_all, target_all, ood_mask=mask_all)
    if active_task_index is not None:
        suffix = "map" if active_task_index == 0 else "hota"
        metrics.update(
            {
                "active_task": task_name_from_index(active_task_index),
                "active_task_index": float(active_task_index),
                "active_rmse": metrics.get(f"rmse_{suffix}"),
                "active_mae": metrics.get(f"mae_{suffix}"),
                "active_spearman": metrics.get(f"spearman_{suffix}"),
                "active_ood_rmse": metrics.get(f"ood_rmse_{suffix}"),
            }
        )
        hide_inactive_task_metrics(metrics, active_task_index)
    metrics["loss"] = mean(losses)
    return mean(losses), metrics


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    loss_module: torch.nn.Module | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    use_amp = bool(args.amp and device.type == "cuda")
    clip_params = list(model.parameters())
    if loss_module is not None:
        clip_params += list(loss_module.parameters())
    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(batch)
            if args.zero_aux_weight > 0 and not getattr(args, "_zero_aux_warning_printed", False):
                if not isinstance(outputs, dict) or "zero_logits" not in outputs:
                    print(
                        "[WARNING] --zero_aux_weight > 0 was set, but this model output "
                        "does not contain `zero_logits`; zero auxiliary loss is inactive."
                    )
                setattr(args, "_zero_aux_warning_printed", True)
            loss = compute_training_loss(outputs, batch, args, loss_module)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return mean(losses)

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    normalization_stats: dict[str, Any],
    epoch: int,
    val_metrics: dict[str, float | None],
    loss_module: torch.nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "modality": args.modality,
            "target_mode": args.target_mode,
            "model_args": vars(args),
            "normalization_stats": normalization_stats,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_name": args.loss,
            "loss_state": loss_module.state_dict() if loss_module is not None else None,
            "val_metrics": val_metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    resolve_and_validate_modality(args)
    finalize_visual_options(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    normalization_stats = load_normalization_stats(config_path=args.config, run_id=args.run_id)
    loaders = build_dataloaders(
        modality=args.modality,
        split=("train", "val"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode=args.target_mode,
        run_id=args.run_id,
        config_path=args.config,
        validate_integrity=not args.skip_integrity,
    )
    require_non_empty_loader(loaders["train"], "train", allow_empty=False)
    require_non_empty_loader(loaders["val"], "val", allow_empty=args.allow_empty_val)
    model = create_model(args.model, args, normalization_stats).to(device)
    actual_backbone = get_actual_visual_backbone(model)
    if actual_backbone is not None and actual_backbone != args.visual_backbone:
        if not args.allow_simple_fallback:
            raise RuntimeError(f"Backbone mismatch: requested={args.visual_backbone}, actual={actual_backbone}")
        args.visual_backbone = actual_backbone
        args.allow_simple_fallback = False
        if actual_backbone == "simple3d":
            args.swin_pretrained = False
            args.freeze_early_layers = False
            args.swin_input_norm = False

    exp_dir = Path(args.experiment_dir) if args.experiment_dir else default_experiment_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)

    loss_module = (
        UncertaintyWeightedHuberLoss(huber_beta=args.huber_beta).to(device)
        if args.loss == "uncertainty_huber"
        else None
    )
    optim_params = list(model.parameters())
    if loss_module is not None:
        optim_params += list(loss_module.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    save_json(exp_dir / "train_config.json", vars(args))
    save_json(exp_dir / "model_summary.json", model_summary(model))
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            args,
            loss_module=loss_module,
            scaler=scaler,
            max_batches=args.max_train_batches,
        )
        val_loss, val_metrics = evaluate_epoch(
            model,
            loaders["val"],
            device,
            args,
            normalization_stats,
            loss_module=loss_module,
            max_batches=args.max_val_batches,
        )
        monitor_value = val_metrics.get("active_rmse")
        if monitor_value is None or not math.isfinite(float(monitor_value)):
            monitor_value = val_metrics.get("rmse_mean")
        if monitor_value is None or not math.isfinite(float(monitor_value)):
            monitor_value = val_loss
        if monitor_value is None or not math.isfinite(float(monitor_value)):
            monitor_value = train_loss
        monitor = float(monitor_value)
        scheduler.step(float(monitor))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss if math.isfinite(val_loss) else None,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        save_json(exp_dir / "train_history.json", {"history": history})
        save_checkpoint(exp_dir / "latest.pt", model, optimizer, args, normalization_stats, epoch, val_metrics, loss_module=loss_module)

        if float(monitor) < best_metric - args.min_delta:
            best_metric = float(monitor)
            stale_epochs = 0
            save_checkpoint(exp_dir / "best.pt", model, optimizer, args, normalization_stats, epoch, val_metrics, loss_module=loss_module)
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_rmse_mean={val_metrics.get('rmse_mean')}"
        )
        if stale_epochs >= args.patience:
            break


if __name__ == "__main__":
    main()
