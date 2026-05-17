from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.run_seed_sweep import DEFAULT_MODELS, VISUAL_MODELS, experiment_dir, parse_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRIC_KEYS = (
    "count",
    "rmse_mean",
    "mae_mean",
    "spearman_mean",
    "zero_target_ratio",
    "zero_pred_ratio",
    "ood_rmse_mean",
    "ood_mae_mean",
)
REQUIRED_SINGLE_TASK_KEYS = (
    "active_rmse",
    "active_mae",
    "active_spearman",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate expected canonical training/evaluation/benchmark outputs.")
    parser.add_argument("--experiments_root", default="experiments/canonical_swin_v1")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--target_mode", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--loss", choices=("huber", "weighted_huber", "mse", "mae", "uncertainty_huber"), default="uncertainty_huber")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--visual_backbone", choices=("swin_tiny", "simple3d"), default="swin_tiny")
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--no_swin_pretrained", action="store_true")
    parser.add_argument("--no_freeze_early_layers", action="store_true")
    parser.add_argument("--require_benchmarks", action="store_true")
    parser.add_argument("--require_reference", action="store_true")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


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


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def csv_row_count(path: Path) -> int | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except Exception:
        return None


def check_metric_keys(model: str, metrics: dict[str, Any]) -> list[str]:
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if "single_task" in model:
        missing.extend(key for key in REQUIRED_SINGLE_TASK_KEYS if key not in metrics)
    return missing


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    sweep_args = naming_args(args)
    failures: list[dict[str, Any]] = []
    checked_runs = 0

    for model in models:
        for seed in seeds:
            checked_runs += 1
            exp_dir = experiment_dir(sweep_args, model, seed)
            required_paths = {
                "best_checkpoint": exp_dir / "best.pt",
                "model_summary": exp_dir / "model_summary.json",
                "train_history": exp_dir / "train_history.json",
                "eval_metrics": exp_dir / f"eval_{args.split}" / f"{args.split}_metrics.json",
            }
            for label, path in required_paths.items():
                if not path.exists():
                    failures.append({"model": model, "seed": seed, "kind": "missing_file", "label": label, "path": str(path)})

            metric_path = required_paths["eval_metrics"]
            if metric_path.exists():
                metrics = load_json(metric_path)
                if metrics is None:
                    failures.append({"model": model, "seed": seed, "kind": "invalid_json", "path": str(metric_path)})
                else:
                    missing_keys = check_metric_keys(model, metrics)
                    if missing_keys:
                        failures.append(
                            {
                                "model": model,
                                "seed": seed,
                                "kind": "missing_metric_keys",
                                "path": str(metric_path),
                                "keys": missing_keys,
                            }
                        )

            if args.require_benchmarks:
                benchmark_path = exp_dir / f"benchmark_{args.split}" / "benchmark_metrics.json"
                should_exist = model in VISUAL_MODELS or model == "param_mlp"
                if should_exist and not benchmark_path.exists():
                    failures.append(
                        {
                            "model": model,
                            "seed": seed,
                            "kind": "missing_file",
                            "label": "benchmark_metrics",
                            "path": str(benchmark_path),
                        }
                    )

    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = PROJECT_ROOT / experiments_root
    aggregate_dir = experiments_root / f"aggregate_{args.split}"
    aggregate_required = {
        "results_summary": aggregate_dir / "results_summary.csv",
        "summary_by_model": aggregate_dir / "summary_by_model.csv",
        "rq_summary": aggregate_dir / "rq_summary.csv",
    }
    if args.require_benchmarks:
        aggregate_required["benchmark_summary"] = aggregate_dir / "benchmark_summary.csv"
        aggregate_required["rq2_deployment_summary"] = aggregate_dir / "rq2_deployment_summary.csv"
    for label, path in aggregate_required.items():
        if not path.exists():
            failures.append({"kind": "missing_file", "label": label, "path": str(path)})
        elif path.suffix == ".csv" and (csv_row_count(path) or 0) == 0:
            failures.append({"kind": "empty_csv", "label": label, "path": str(path)})

    if args.require_reference:
        reference_path = experiments_root / "reference_yolo_deepsort" / "reference_metrics.json"
        payload = load_json(reference_path) if reference_path.exists() else None
        if payload is None:
            failures.append({"kind": "missing_or_invalid_reference", "path": str(reference_path)})
        elif not payload.get("reference_time_ms_for_surrogate_speedup"):
            failures.append({"kind": "invalid_reference_time", "path": str(reference_path)})

    result = {
        "ok": not failures,
        "checked_runs": checked_runs,
        "expected_models": models,
        "seeds": seeds,
        "split": args.split,
        "failure_count": len(failures),
        "failures": failures,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print(f"[OK] canonical outputs validated: runs={checked_runs}, split={args.split}")


if __name__ == "__main__":
    main()
