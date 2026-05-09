from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = (
    "rmse_map",
    "rmse_hota",
    "rmse_mean",
    "mae_map",
    "mae_hota",
    "mae_mean",
    "spearman_map",
    "spearman_hota",
    "spearman_mean",
    "ood_rmse_map",
    "ood_rmse_hota",
    "ood_rmse_mean",
    "ood_mae_map",
    "ood_mae_hota",
    "ood_mae_mean",
    "zero_target_ratio",
    "zero_pred_ratio",
    "zero_count",
    "zero_rmse_map",
    "zero_rmse_hota",
    "zero_rmse_mean",
    "zero_mae_map",
    "zero_mae_hota",
    "zero_mae_mean",
    "nonzero_count",
    "nonzero_rmse_map",
    "nonzero_rmse_hota",
    "nonzero_rmse_mean",
    "nonzero_mae_map",
    "nonzero_mae_hota",
    "nonzero_mae_mean",
    "active_rmse",
    "active_mae",
    "active_spearman",
    "active_ood_rmse",
    "active_zero_rmse",
    "active_nonzero_rmse",
    "active_zero_count",
    "active_zero_mae",
    "active_nonzero_count",
    "active_nonzero_mae",
)

SUMMARY_METRICS = (
    "rmse_map",
    "rmse_hota",
    "rmse_mean",
    "mae_mean",
    "spearman_mean",
    "ood_rmse_mean",
    "zero_rmse_mean",
    "nonzero_rmse_mean",
    "active_rmse",
    "active_mae",
    "active_spearman",
    "active_ood_rmse",
    "active_zero_rmse",
    "active_nonzero_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate surrogate evaluation metrics across runs/seeds.")
    parser.add_argument("--experiments_root", default="experiments")
    parser.add_argument("--glob", default=None, help="Metric file glob. Defaults to **/eval_{split}/{split}_metrics.json.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--group_by",
        default=(
            "run_id,model,active_task,loss,target_mode,epochs,batch_size,visual_backbone,visual_feature_dim,"
            "swin_pretrained,swin_input_norm,freeze_early_layers"
        ),
        help="Comma-separated result columns used for summary grouping.",
    )
    parser.add_argument(
        "--rq_duplicate_policy",
        choices=("error", "first"),
        default="error",
        help="How rq_summary handles duplicate rows for the same model/comparison key.",
    )
    return parser.parse_args()


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_metric_files(experiments_root: Path, pattern: str) -> list[Path]:
    search_pattern = pattern
    if not Path(search_pattern).is_absolute():
        search_pattern = str(experiments_root / search_pattern)
    return sorted(Path(p) for p in globlib.glob(search_pattern, recursive=True))


def find_benchmark_files(experiments_root: Path, split: str) -> list[Path]:
    return sorted(Path(p) for p in globlib.glob(str(experiments_root / f"**/benchmark_{split}/benchmark_metrics.json"), recursive=True))


def find_train_config(metric_path: Path, experiments_root: Path) -> dict[str, Any]:
    for directory in (metric_path.parent, *metric_path.parents):
        config_path = directory / "train_config.json"
        if config_path.exists():
            return load_json_or_empty(config_path)
        if directory == experiments_root or directory.parent == directory:
            break
    return {}


def metric_split_from_path(metric_path: Path, fallback: str) -> str:
    name = metric_path.name
    suffix = "_metrics.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return fallback


def normalize_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def row_from_metric(metric_path: Path, experiments_root: Path, split: str) -> dict[str, Any]:
    metrics = load_json_or_empty(metric_path)
    eval_config = load_json_or_empty(metric_path.parent / "eval_config.json")
    train_config = find_train_config(metric_path, experiments_root)

    model = eval_config.get("model") or train_config.get("model")
    run_id = eval_config.get("run_id", train_config.get("run_id"))
    row: dict[str, Any] = {
        "metric_path": str(metric_path),
        "experiment_dir": str(metric_path.parent.parent if metric_path.parent.name.startswith("eval_") else metric_path.parent),
        "model": model,
        "seed": train_config.get("seed"),
        "run_id": "canonical" if run_id in (None, "") else run_id,
        "split": eval_config.get("split") or metric_split_from_path(metric_path, split),
        "target_mode": eval_config.get("target_mode") or train_config.get("target_mode"),
        "active_task": eval_config.get("active_task"),
        "loss": train_config.get("loss"),
        "epochs": train_config.get("epochs"),
        "batch_size": train_config.get("batch_size"),
        "visual_backbone": train_config.get("visual_backbone"),
        "visual_feature_dim": train_config.get("visual_feature_dim"),
        "swin_pretrained": train_config.get("swin_pretrained"),
        "swin_input_norm": train_config.get("swin_input_norm"),
        "freeze_early_layers": train_config.get("freeze_early_layers"),
        "checkpoint": eval_config.get("checkpoint"),
    }
    for key in METRIC_COLUMNS:
        row[key] = normalize_value(metrics.get(key))
    return row


BENCHMARK_METRIC_COLUMNS = (
    "num_samples",
    "data_time_ms_mean",
    "transfer_time_ms_mean",
    "forward_time_ms_mean",
    "preprocess_time_ms_mean",
    "total_time_ms_mean",
    "per_sample_preprocess_time_ms_mean",
    "per_sample_forward_time_ms_mean",
    "per_sample_total_time_ms_mean",
    "fps",
    "reference_time_ms",
    "speedup_ratio",
)


def benchmark_row_from_metric(path: Path, experiments_root: Path) -> dict[str, Any]:
    metrics = load_json_or_empty(path)
    train_config = find_train_config(path, experiments_root)
    model = metrics.get("model") or train_config.get("model")
    run_id = metrics.get("run_id", train_config.get("run_id"))
    row: dict[str, Any] = {
        "benchmark_path": str(path),
        "experiment_dir": str(path.parent.parent if path.parent.name.startswith("benchmark_") else path.parent),
        "model": model,
        "seed": train_config.get("seed"),
        "run_id": "canonical" if run_id in (None, "") else run_id,
        "split": metrics.get("split"),
        "target_mode": metrics.get("target_mode") or train_config.get("target_mode"),
        "loss": train_config.get("loss"),
        "epochs": train_config.get("epochs"),
        "batch_size": metrics.get("batch_size", train_config.get("batch_size")),
        "visual_backbone": train_config.get("visual_backbone"),
        "visual_feature_dim": train_config.get("visual_feature_dim"),
        "swin_pretrained": train_config.get("swin_pretrained"),
        "swin_input_norm": train_config.get("swin_input_norm"),
        "freeze_early_layers": train_config.get("freeze_early_layers"),
        "benchmark_input_source": metrics.get("benchmark_input_source"),
        "proposal_efficiency_compliant": metrics.get("proposal_efficiency_compliant"),
        "proposal_raw_preprocessing_compliant": metrics.get("proposal_raw_preprocessing_compliant"),
        "proposal_batch_size_compliant": metrics.get("proposal_batch_size_compliant"),
    }
    for key in BENCHMARK_METRIC_COLUMNS:
        row[key] = normalize_value(metrics.get(key))
    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def parse_group_by(raw: str) -> list[str]:
    fields = [field.strip() for field in raw.split(",") if field.strip()]
    return fields or ["model"]


def group_key(row: dict[str, Any], group_by: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in group_by)


def summarize_grouped(rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row, group_by), []).append(row)

    summary_rows = []
    for key_values, model_rows in sorted(grouped.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])):
        summary: dict[str, Any] = {field: value for field, value in zip(group_by, key_values)}
        summary["group_key"] = "__".join("" if value is None else str(value) for value in key_values)
        summary["run_count"] = len(model_rows)
        seeds = sorted({str(row.get("seed")) for row in model_rows if row.get("seed") not in (None, "")})
        summary["seed_count"] = len(seeds)
        summary["seeds"] = ",".join(seeds)
        for key in SUMMARY_METRICS:
            values = numeric_values(model_rows, key)
            summary[f"{key}_count"] = len(values)
            summary[f"{key}_mean"] = statistics.mean(values) if values else None
            summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        summary_rows.append(summary)
    return summary_rows


def comparable_key(row: dict[str, Any], key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in key_fields)


def first_rows_by_key(
    rows: list[dict[str, Any]],
    model: str,
    key_fields: tuple[str, ...],
    duplicate_policy: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates: list[tuple[Any, ...]] = []
    for row in rows:
        if row.get("model") != model:
            continue
        key = comparable_key(row, key_fields)
        if key in out:
            duplicates.append(key)
            if duplicate_policy == "error":
                continue
        out.setdefault(key, row)
    if duplicates and duplicate_policy == "error":
        sample = duplicates[:5]
        raise RuntimeError(
            f"Duplicate comparable rows for model={model}, key_fields={key_fields}: {sample}. "
            "Use a clean experiments_root, narrow --glob, or pass --rq_duplicate_policy first for exploratory aggregation."
        )
    return out


def numeric_or_none(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def paired_comparison(
    rows: list[dict[str, Any]],
    rq: str,
    metric: str,
    baseline_model: str,
    baseline_metric: str,
    proposed_model: str,
    proposed_metric: str,
    key_fields: tuple[str, ...],
    duplicate_policy: str,
) -> dict[str, Any]:
    baseline_rows = first_rows_by_key(rows, baseline_model, key_fields, duplicate_policy)
    proposed_rows = first_rows_by_key(rows, proposed_model, key_fields, duplicate_policy)
    common_keys = sorted(set(baseline_rows) & set(proposed_rows), key=lambda key: tuple("" if value is None else str(value) for value in key))

    baseline_values: list[float] = []
    proposed_values: list[float] = []
    wins = 0
    seeds: list[str] = []
    for key in common_keys:
        baseline_value = numeric_or_none(baseline_rows[key], baseline_metric)
        proposed_value = numeric_or_none(proposed_rows[key], proposed_metric)
        if baseline_value is None or proposed_value is None:
            continue
        baseline_values.append(baseline_value)
        proposed_values.append(proposed_value)
        if proposed_value < baseline_value:
            wins += 1
        key_map = dict(zip(key_fields, key))
        seeds.append(str(key_map.get("seed", "")))

    baseline_mean = statistics.mean(baseline_values) if baseline_values else None
    proposed_mean = statistics.mean(proposed_values) if proposed_values else None
    return {
        "rq": rq,
        "metric": metric,
        "baseline": baseline_model,
        "baseline_metric": baseline_metric,
        "proposed": proposed_model,
        "proposed_metric": proposed_metric,
        "pair_key": ",".join(key_fields),
        "seed_count": len(baseline_values),
        "seeds": ",".join(seeds),
        "proposed_win_count": wins,
        "baseline_mean": baseline_mean,
        "proposed_mean": proposed_mean,
        "mean_delta": proposed_mean - baseline_mean if baseline_mean is not None and proposed_mean is not None else None,
        "baseline_std": statistics.stdev(baseline_values) if len(baseline_values) > 1 else (0.0 if baseline_values else None),
        "proposed_std": statistics.stdev(proposed_values) if len(proposed_values) > 1 else (0.0 if proposed_values else None),
    }


def build_rq_summary(rows: list[dict[str, Any]], duplicate_policy: str) -> list[dict[str, Any]]:
    base_key = ("seed", "run_id", "loss", "target_mode", "epochs", "batch_size")
    visual_key = (
        "seed",
        "run_id",
        "loss",
        "target_mode",
        "epochs",
        "batch_size",
        "visual_backbone",
        "visual_feature_dim",
        "swin_pretrained",
        "swin_input_norm",
        "freeze_early_layers",
    )
    return [
        paired_comparison(rows, "RQ1", "rmse_mean", "param_mlp", "rmse_mean", "fusion_multitask", "rmse_mean", base_key, duplicate_policy),
        paired_comparison(rows, "RQ1", "rmse_mean", "visual_baseline", "rmse_mean", "fusion_multitask", "rmse_mean", visual_key, duplicate_policy),
        paired_comparison(rows, "RQ1_OOD", "ood_rmse_mean", "param_mlp", "ood_rmse_mean", "fusion_multitask", "ood_rmse_mean", base_key, duplicate_policy),
        paired_comparison(rows, "RQ1_OOD", "ood_rmse_mean", "visual_baseline", "ood_rmse_mean", "fusion_multitask", "ood_rmse_mean", visual_key, duplicate_policy),
        paired_comparison(rows, "RQ2", "delta_map_rmse", "visual_single_task_map", "active_rmse", "visual_baseline", "rmse_map", visual_key, duplicate_policy),
        paired_comparison(rows, "RQ2", "delta_hota_rmse", "visual_single_task_hota", "active_rmse", "visual_baseline", "rmse_hota", visual_key, duplicate_policy),
        paired_comparison(rows, "RQ2", "delta_map_rmse", "fusion_single_task_map", "active_rmse", "fusion_multitask", "rmse_map", visual_key, duplicate_policy),
        paired_comparison(rows, "RQ2", "delta_hota_rmse", "fusion_single_task_hota", "active_rmse", "fusion_multitask", "rmse_hota", visual_key, duplicate_policy),
    ]


def benchmark_key_fields() -> tuple[str, ...]:
    return (
        "seed",
        "run_id",
        "loss",
        "target_mode",
        "epochs",
        "batch_size",
        "visual_backbone",
        "visual_feature_dim",
        "swin_pretrained",
        "swin_input_norm",
        "freeze_early_layers",
        "benchmark_input_source",
    )


def build_rq2_deployment_summary(rows: list[dict[str, Any]], duplicate_policy: str) -> list[dict[str, Any]]:
    key_fields = benchmark_key_fields()
    comparisons = [
        ("visual", "visual_baseline", "visual_single_task_map", "visual_single_task_hota"),
        ("fusion", "fusion_multitask", "fusion_single_task_map", "fusion_single_task_hota"),
    ]
    out: list[dict[str, Any]] = []
    for family, multitask_model, map_model, hota_model in comparisons:
        multitask = first_rows_by_key(rows, multitask_model, key_fields, duplicate_policy)
        single_map = first_rows_by_key(rows, map_model, key_fields, duplicate_policy)
        single_hota = first_rows_by_key(rows, hota_model, key_fields, duplicate_policy)
        common_keys = sorted(
            set(multitask) & set(single_map) & set(single_hota),
            key=lambda key: tuple("" if value is None else str(value) for value in key),
        )
        multitask_values: list[float] = []
        single_sum_values: list[float] = []
        seeds: list[str] = []
        for key in common_keys:
            mt = numeric_or_none(multitask[key], "per_sample_total_time_ms_mean")
            sm = numeric_or_none(single_map[key], "per_sample_total_time_ms_mean")
            sh = numeric_or_none(single_hota[key], "per_sample_total_time_ms_mean")
            if mt is None or sm is None or sh is None:
                continue
            multitask_values.append(mt)
            single_sum_values.append(sm + sh)
            seeds.append(str(dict(zip(key_fields, key)).get("seed", "")))
        mt_mean = statistics.mean(multitask_values) if multitask_values else None
        single_mean = statistics.mean(single_sum_values) if single_sum_values else None
        out.append(
            {
                "rq": "RQ2_DEPLOYMENT",
                "family": family,
                "multitask_model": multitask_model,
                "single_task_models": f"{map_model}+{hota_model}",
                "metric": "per_sample_total_time_ms_mean",
                "pair_key": ",".join(key_fields),
                "seed_count": len(multitask_values),
                "seeds": ",".join(seeds),
                "multitask_mean": mt_mean,
                "single_task_sum_mean": single_mean,
                "latency_delta": mt_mean - single_mean if mt_mean is not None and single_mean is not None else None,
                "single_over_multitask_ratio": single_mean / mt_mean if mt_mean not in (None, 0.0) and single_mean is not None else None,
                "multitask_std": statistics.stdev(multitask_values) if len(multitask_values) > 1 else (0.0 if multitask_values else None),
                "single_task_sum_std": statistics.stdev(single_sum_values) if len(single_sum_values) > 1 else (0.0 if single_sum_values else None),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = PROJECT_ROOT / experiments_root
    pattern = args.glob or f"**/eval_{args.split}/{args.split}_metrics.json"
    metric_files = find_metric_files(experiments_root, pattern)
    group_by = parse_group_by(args.group_by)

    rows = [row_from_metric(path, experiments_root, args.split) for path in metric_files]
    out_dir = Path(args.output_dir) if args.output_dir else experiments_root / f"aggregate_{args.split}"
    result_fields = [
        "metric_path",
        "experiment_dir",
        "model",
        "seed",
        "run_id",
        "split",
        "target_mode",
        "active_task",
        "loss",
        "epochs",
        "batch_size",
        "visual_backbone",
        "visual_feature_dim",
        "swin_pretrained",
        "swin_input_norm",
        "freeze_early_layers",
        "checkpoint",
        *METRIC_COLUMNS,
    ]
    write_csv(out_dir / "results_summary.csv", rows, result_fields)

    summary_rows = summarize_grouped(rows, group_by)
    summary_fields = sorted({key for row in summary_rows for key in row})
    preferred = [*group_by, "group_key", "run_count", "seed_count", "seeds"]
    summary_fields = preferred + [key for key in summary_fields if key not in preferred]
    write_csv(out_dir / "summary_by_model.csv", summary_rows, summary_fields)
    rq_rows = build_rq_summary(rows, args.rq_duplicate_policy)
    rq_fields = [
        "rq",
        "metric",
        "baseline",
        "baseline_metric",
        "proposed",
        "proposed_metric",
        "pair_key",
        "seed_count",
        "seeds",
        "proposed_win_count",
        "baseline_mean",
        "proposed_mean",
        "mean_delta",
        "baseline_std",
        "proposed_std",
    ]
    write_csv(out_dir / "rq_summary.csv", rq_rows, rq_fields)

    benchmark_files = find_benchmark_files(experiments_root, args.split)
    benchmark_rows = [benchmark_row_from_metric(path, experiments_root) for path in benchmark_files]
    benchmark_fields = [
        "benchmark_path",
        "experiment_dir",
        "model",
        "seed",
        "run_id",
        "split",
        "target_mode",
        "loss",
        "epochs",
        "batch_size",
        "visual_backbone",
        "visual_feature_dim",
        "swin_pretrained",
        "swin_input_norm",
        "freeze_early_layers",
        "benchmark_input_source",
        "proposal_efficiency_compliant",
        "proposal_raw_preprocessing_compliant",
        "proposal_batch_size_compliant",
        *BENCHMARK_METRIC_COLUMNS,
    ]
    write_csv(out_dir / "benchmark_summary.csv", benchmark_rows, benchmark_fields)

    rq2_deployment_rows = build_rq2_deployment_summary(benchmark_rows, args.rq_duplicate_policy)
    rq2_deployment_fields = [
        "rq",
        "family",
        "multitask_model",
        "single_task_models",
        "metric",
        "pair_key",
        "seed_count",
        "seeds",
        "multitask_mean",
        "single_task_sum_mean",
        "latency_delta",
        "single_over_multitask_ratio",
        "multitask_std",
        "single_task_sum_std",
    ]
    write_csv(out_dir / "rq2_deployment_summary.csv", rq2_deployment_rows, rq2_deployment_fields)
    save_json(
        out_dir / "aggregate_config.json",
        {
            "experiments_root": str(experiments_root),
            "glob": pattern,
            "split": args.split,
            "group_by": group_by,
            "rq_duplicate_policy": args.rq_duplicate_policy,
            "metric_file_count": len(metric_files),
            "benchmark_file_count": len(benchmark_files),
        },
    )
    print(f"[DONE] rows: {out_dir / 'results_summary.csv'}")
    print(f"[DONE] summary: {out_dir / 'summary_by_model.csv'}")
    print(f"[DONE] rq summary: {out_dir / 'rq_summary.csv'}")
    print(f"[DONE] benchmark summary: {out_dir / 'benchmark_summary.csv'}")
    print(f"[DONE] rq2 deployment summary: {out_dir / 'rq2_deployment_summary.csv'}")


if __name__ == "__main__":
    main()
