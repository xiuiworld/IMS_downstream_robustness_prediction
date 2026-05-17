from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.run_seed_sweep import DEFAULT_MODELS, VISUAL_MODELS, experiment_dir, parse_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proposal-compliant surrogate benchmark jobs over canonical checkpoints.")
    parser.add_argument("--experiments_root", default="experiments/canonical_swin_v1")
    parser.add_argument("--reference_metrics", default="experiments/canonical_swin_v1/reference_yolo_deepsort/reference_metrics.json")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target_mode", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--loss", choices=("huber", "weighted_huber", "mse", "mae", "uncertainty_huber"), default="uncertainty_huber")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size used in run_seed_sweep naming.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--visual_backbone", choices=("swin_tiny", "simple3d"), default="swin_tiny")
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--no_swin_pretrained", action="store_true")
    parser.add_argument("--no_freeze_early_layers", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--include_param_cached",
        action="store_true",
        help="Also benchmark param_mlp using cached metadata tensors. This is not used for raw-frame speedup claims.",
    )
    return parser.parse_args()


def load_reference_time_ms(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    value = payload.get("reference_time_ms_for_surrogate_speedup", payload.get("clip_time_ms_median"))
    if value is None or float(value) <= 0:
        raise RuntimeError(f"Invalid reference time in {path}: {value}")
    return float(value)


def naming_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        run_id=args.run_id,
        target_mode=args.target_mode,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        visual_backbone=args.visual_backbone,
        visual_feature_dim=args.visual_feature_dim,
        no_swin_pretrained=args.no_swin_pretrained,
        no_freeze_early_layers=args.no_freeze_early_layers,
        experiments_root=args.experiments_root,
        train_extra=None,
    )


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def benchmark_exists(checkpoint: Path, split: str) -> bool:
    return (checkpoint.parent / f"benchmark_{split}" / "benchmark_metrics.json").exists()


def main() -> None:
    args = parse_args()
    reference_metrics = Path(args.reference_metrics)
    if not reference_metrics.is_absolute():
        reference_metrics = PROJECT_ROOT / reference_metrics
    reference_time_ms = load_reference_time_ms(reference_metrics)
    models = parse_csv(args.models)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    sweep_args = naming_args(args)

    for model in models:
        for seed in seeds:
            checkpoint = experiment_dir(sweep_args, model, seed) / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            if args.skip_existing and benchmark_exists(checkpoint, args.split):
                print(f"[SKIP] existing benchmark: {checkpoint.parent / f'benchmark_{args.split}' / 'benchmark_metrics.json'}")
                continue

            command = [
                sys.executable,
                "-m",
                "training.benchmark_inference",
                "--checkpoint",
                str(checkpoint),
                "--config",
                args.config,
                "--split",
                args.split,
                "--num_workers",
                str(args.num_workers),
                "--num_warmup",
                str(args.num_warmup),
                "--num_batches",
                str(args.num_batches),
                "--device",
                args.device,
            ]
            if args.amp:
                command.append("--amp")

            if model in VISUAL_MODELS:
                command.extend(
                    [
                        "--input_source",
                        "raw_frames",
                        "--batch_size",
                        "1",
                        "--reference_time_ms",
                        str(reference_time_ms),
                    ]
                )
                run_command(command, args.dry_run)
            elif args.include_param_cached:
                command.extend(["--input_source", "cached", "--batch_size", "1"])
                run_command(command, args.dry_run)
            else:
                print(f"[SKIP] metadata-only benchmark not requested for {model} seed={seed}")


if __name__ == "__main__":
    main()
