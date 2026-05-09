from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RQ1_MODELS = ("param_mlp", "visual_baseline", "fusion_multitask")
ABLATION_MODELS = (
    "visual_single_task_map",
    "visual_single_task_hota",
    "fusion_single_task_map",
    "fusion_single_task_hota",
)
DEFAULT_MODELS = (*RQ1_MODELS, *ABLATION_MODELS)
SUPPORTED_MODELS = DEFAULT_MODELS
VISUAL_MODELS = {
    "visual_baseline",
    "fusion_multitask",
    "visual_single_task_map",
    "visual_single_task_hota",
    "fusion_single_task_map",
    "fusion_single_task_hota",
}


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical train/eval sweeps for surrogate models.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated models to run. Defaults to full RQ1+RQ2 sweep.")
    parser.add_argument("--rq1_only", action="store_true", help="Run only param/visual/fusion RQ1 models from the selected model list.")
    parser.add_argument("--include_ablations", action="store_true", help="Legacy option; ablations are included by default unless --rq1_only or --models excludes them.")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target_mode", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loss", choices=("huber", "weighted_huber", "mse", "mae", "uncertainty_huber"), default="uncertainty_huber")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--visual_backbone", choices=("swin_tiny", "simple3d"), default="swin_tiny")
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--no_swin_pretrained", action="store_true", help="Disable canonical Kinetics-400 pretrained Swin initialization.")
    parser.add_argument("--no_freeze_early_layers", action="store_true", help="Disable canonical early Swin layer freezing.")
    parser.add_argument("--experiments_root", default="experiments")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--train_extra", nargs=argparse.REMAINDER, default=None, help="Extra args appended to training.train.")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_id_args(run_id: str | None) -> list[str]:
    return ["--run_id", run_id] if run_id not in (None, "") else []


def train_extra_hash(extra: list[str] | None) -> str | None:
    if not extra:
        return None
    payload = "\x1f".join(extra)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:10]


def experiment_dir(args: argparse.Namespace, model: str, seed: int) -> Path:
    run_token = args.run_id or "canonical"
    parts = [
        model,
        f"run-{run_token}",
        f"seed-{seed}",
        f"target-{args.target_mode}",
        f"loss-{args.loss}",
        f"epochs-{args.epochs}",
        f"bs-{args.batch_size}",
        f"lr-{args.lr:.3g}",
        f"wd-{args.weight_decay:.3g}",
    ]
    if model in VISUAL_MODELS:
        use_pretrained = not args.no_swin_pretrained and args.visual_backbone == "swin_tiny"
        freeze_early = not args.no_freeze_early_layers and args.visual_backbone == "swin_tiny"
        parts.extend(
            [
                f"backbone-{args.visual_backbone}",
                f"vfeat-{args.visual_feature_dim}",
                f"pretrained-{int(use_pretrained)}",
                f"freeze-{int(freeze_early)}",
            ]
        )
    extra_hash = train_extra_hash(args.train_extra)
    if extra_hash is not None:
        parts.append(f"extra-{extra_hash}")
    return Path(args.experiments_root) / "__".join(parts)


def train_command(args: argparse.Namespace, model: str, seed: int, exp_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--model",
        model,
        "--config",
        args.config,
        *run_id_args(args.run_id),
        "--target_mode",
        args.target_mode,
        "--loss",
        args.loss,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--experiment_dir",
        str(exp_dir),
    ]
    if model in VISUAL_MODELS:
        command.extend(
            [
                "--visual_backbone",
                args.visual_backbone,
                "--visual_feature_dim",
                str(args.visual_feature_dim),
            ]
        )
        if not args.no_swin_pretrained and args.visual_backbone == "swin_tiny":
            command.append("--swin_pretrained")
        if not args.no_freeze_early_layers and args.visual_backbone == "swin_tiny":
            command.append("--freeze_early_layers")
    if args.train_extra:
        command.extend(args.train_extra)
    return command


def eval_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--config",
        args.config,
        "--split",
        args.split,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]


def aggregate_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.aggregate_results",
        "--experiments_root",
        args.experiments_root,
        "--split",
        args.split,
    ]


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    if args.rq1_only:
        models = [model for model in models if model in RQ1_MODELS]
    if args.include_ablations:
        models = [*models, *[model for model in ABLATION_MODELS if model not in models]]
    seeds = [int(seed) for seed in parse_csv(args.seeds)]

    for model in models:
        if model not in SUPPORTED_MODELS:
            raise RuntimeError(f"Unsupported sweep model: {model}")
        for seed in seeds:
            exp_dir = experiment_dir(args, model, seed)
            checkpoint = exp_dir / "best.pt"
            if not (args.skip_existing and checkpoint.exists()):
                run_command(train_command(args, model, seed, exp_dir), args.dry_run)
            else:
                print(f"[SKIP] existing checkpoint: {checkpoint}")
            run_command(eval_command(args, checkpoint), args.dry_run)

    run_command(aggregate_command(args), args.dry_run)


if __name__ == "__main__":
    main()
