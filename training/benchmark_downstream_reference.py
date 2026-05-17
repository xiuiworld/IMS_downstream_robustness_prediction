from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the YOLOv8+DeepSORT downstream reference latency on raw clip frames."
    )
    parser.add_argument("--targets", default="data/processed/targets/surrogate_targets.csv")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num_clips", type=int, default=50)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="experiments/canonical_swin_v1/reference_yolo_deepsort")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.7)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=None,
        help="Reference device label. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--yolo_device",
        default=None,
        help="Ultralytics device string. Defaults to 0 for CUDA, otherwise cpu.",
    )
    parser.add_argument("--yolo_agnostic_nms", type=str2bool, default=False)
    parser.add_argument("--ds_max_age", type=int, default=30)
    parser.add_argument("--ds_n_init", type=int, default=3)
    parser.add_argument("--ds_nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--ds_max_iou_distance", type=float, default=0.7)
    parser.add_argument("--ds_max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--ds_nn_budget", type=int, default=None)
    parser.add_argument("--ds_embedder", type=str, default="mobilenet")
    parser.add_argument(
        "--ds_embedder_gpu",
        type=str2bool,
        default=None,
        help="Defaults to true on CUDA devices and false on CPU.",
    )
    parser.add_argument("--ds_half", type=str2bool, default=True)
    parser.add_argument("--ds_bgr", type=str2bool, default=True)
    parser.add_argument("--ds_embedder_model_name", type=str, default=None)
    parser.add_argument("--ds_embedder_wts", type=str, default=None)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "measured_index",
        "is_warmup",
        "clip_id",
        "split",
        "degradation_type",
        "degradation_param",
        "frame_count",
        "detection_count",
        "confirmed_track_count",
        "read_time_ms",
        "yolo_time_ms",
        "deepsort_time_ms",
        "total_time_ms",
        "per_frame_total_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_existing_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_target_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    target_path = resolve_existing_path(args.targets)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing surrogate target file: {target_path}")
    df = pd.read_csv(target_path)
    required = {"clip_id", "split", "file_path", "start_frame", "end_frame", "degradation_type", "degradation_param"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Target file missing required columns: {missing}")
    df = df[df["split"].astype(str) == args.split].copy()
    if df.empty:
        raise RuntimeError(f"No target rows found for split={args.split}")
    records = df.to_dict(orient="records")
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(records)
    needed = args.num_warmup + args.num_clips
    return records[:needed]


def frame_paths_for_row(row: dict[str, Any]) -> list[Path]:
    clip_dir = resolve_existing_path(str(row["file_path"]))
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found for clip_id={row['clip_id']}: {clip_dir}")
    start_frame = int(float(row["start_frame"]))
    end_frame = int(float(row["end_frame"]))
    indexed: list[tuple[int, Path]] = []
    for path in clip_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_FRAME_EXTS:
            continue
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        if start_frame <= frame_idx <= end_frame:
            indexed.append((frame_idx, path))
    indexed.sort(key=lambda item: item[0])
    expected_count = end_frame - start_frame + 1
    if len(indexed) != expected_count:
        raise RuntimeError(
            f"Frame count mismatch for clip_id={row['clip_id']}: expected={expected_count}, found={len(indexed)}"
        )
    if indexed and (indexed[0][0] != start_frame or indexed[-1][0] != end_frame):
        raise RuntimeError(
            f"Frame range mismatch for clip_id={row['clip_id']}: "
            f"expected={start_frame}-{end_frame}, found={indexed[0][0]}-{indexed[-1][0]}"
        )
    return [path for _, path in indexed]


def build_deepsort_config(args: argparse.Namespace, device_label: str) -> dict[str, Any]:
    embedder_gpu = args.ds_embedder_gpu
    if embedder_gpu is None:
        embedder_gpu = str(device_label).startswith("cuda")
    return {
        "max_age": args.ds_max_age,
        "n_init": args.ds_n_init,
        "nms_max_overlap": args.ds_nms_max_overlap,
        "max_iou_distance": args.ds_max_iou_distance,
        "max_cosine_distance": args.ds_max_cosine_distance,
        "nn_budget": args.ds_nn_budget,
        "embedder": args.ds_embedder,
        "embedder_gpu": embedder_gpu,
        "half": args.ds_half,
        "bgr": args.ds_bgr,
        "embedder_model_name": args.ds_embedder_model_name,
        "embedder_wts": args.ds_embedder_wts,
    }


def resolve_deepsort_kwargs(deepsort_cls: type, config: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(deepsort_cls.__init__)
    accepted = {key for key in sig.parameters if key != "self"}
    unsupported = sorted(key for key, value in config.items() if value is not None and key not in accepted)
    if unsupported:
        raise RuntimeError(
            "DeepSort config contains unsupported parameters for installed deep-sort-realtime: "
            f"{unsupported}"
        )
    return {key: value for key, value in config.items() if value is not None and key in accepted}


def import_runtime_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import torch
        from deep_sort_realtime.deepsort_tracker import DeepSort
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing benchmark dependencies. Install with: "
            "pip install ultralytics deep-sort-realtime lap opencv-python pandas numpy pyyaml tqdm scipy"
        ) from exc
    return cv2, torch, YOLO, DeepSort


def normalize_device_args(args: argparse.Namespace, torch_module: Any) -> tuple[str, str]:
    cuda_available = bool(torch_module.cuda.is_available())
    device_label = args.device or ("cuda" if cuda_available else "cpu")
    if args.yolo_device is not None:
        yolo_device = str(args.yolo_device)
    elif str(device_label).startswith("cuda") and cuda_available:
        yolo_device = "0"
    else:
        yolo_device = "cpu"
    return str(device_label), yolo_device


def sync_if_needed(torch_module: Any, device_label: str) -> None:
    if str(device_label).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def gpu_name(torch_module: Any, device_label: str) -> str | None:
    if str(device_label).startswith("cuda") and torch_module.cuda.is_available():
        return torch_module.cuda.get_device_name(0)
    return None


def benchmark_clip(
    row: dict[str, Any],
    frame_paths: list[Path],
    model: Any,
    tracker: Any,
    cv2_module: Any,
    torch_module: Any,
    args: argparse.Namespace,
    device_label: str,
    yolo_device: str,
) -> dict[str, Any]:
    read_ms = 0.0
    yolo_ms = 0.0
    deepsort_ms = 0.0
    detection_count = 0
    confirmed_track_count = 0
    total_start = time.perf_counter()

    for frame_path in frame_paths:
        read_start = time.perf_counter()
        frame = cv2_module.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        read_end = time.perf_counter()
        read_ms += (read_end - read_start) * 1000.0

        sync_if_needed(torch_module, device_label)
        yolo_start = time.perf_counter()
        infer_results = model(
            frame,
            classes=[0],
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            device=yolo_device,
            agnostic_nms=args.yolo_agnostic_nms,
            verbose=False,
        )
        sync_if_needed(torch_module, device_label)
        yolo_end = time.perf_counter()
        yolo_ms += (yolo_end - yolo_start) * 1000.0

        detections = []
        for result in infer_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                detections.append(([x1, y1, w, h], conf, "person"))
        detection_count += len(detections)

        sync_if_needed(torch_module, device_label)
        ds_start = time.perf_counter()
        tracks = tracker.update_tracks(detections, frame=frame)
        sync_if_needed(torch_module, device_label)
        ds_end = time.perf_counter()
        deepsort_ms += (ds_end - ds_start) * 1000.0
        confirmed_track_count += sum(1 for track in tracks if track.is_confirmed())

    total_ms = (time.perf_counter() - total_start) * 1000.0
    frame_count = len(frame_paths)
    return {
        "clip_id": str(row["clip_id"]),
        "split": str(row["split"]),
        "degradation_type": str(row["degradation_type"]),
        "degradation_param": row["degradation_param"],
        "frame_count": frame_count,
        "detection_count": detection_count,
        "confirmed_track_count": confirmed_track_count,
        "read_time_ms": read_ms,
        "yolo_time_ms": yolo_ms,
        "deepsort_time_ms": deepsort_ms,
        "total_time_ms": total_ms,
        "per_frame_total_time_ms": total_ms / frame_count if frame_count else None,
    }


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def summarize(values: list[float]) -> dict[str, float | None]:
    vals = finite(values)
    if not vals:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main() -> None:
    args = parse_args()
    if args.num_clips <= 0:
        raise RuntimeError("--num_clips must be positive.")
    if args.num_warmup < 0:
        raise RuntimeError("--num_warmup must be non-negative.")

    cv2_module, torch_module, yolo_cls, deepsort_cls = import_runtime_deps()
    device_label, yolo_device = normalize_device_args(args, torch_module)
    rows_source = load_target_rows(args)
    if len(rows_source) <= args.num_warmup:
        raise RuntimeError(
            f"Not enough rows to collect measurements: rows={len(rows_source)}, warmup={args.num_warmup}"
        )

    print(f"[INFO] Loading detector: {args.yolo_model}")
    model = yolo_cls(args.yolo_model)
    deepsort_config = build_deepsort_config(args, device_label)
    deepsort_kwargs = resolve_deepsort_kwargs(deepsort_cls, deepsort_config)

    measured_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows_source):
        is_warmup = row_index < args.num_warmup
        frame_paths = frame_paths_for_row(row)
        tracker = deepsort_cls(**deepsort_kwargs)
        result = benchmark_clip(
            row=row,
            frame_paths=frame_paths,
            model=model,
            tracker=tracker,
            cv2_module=cv2_module,
            torch_module=torch_module,
            args=args,
            device_label=device_label,
            yolo_device=yolo_device,
        )
        if not is_warmup:
            measured_rows.append(result)
        result.update(
            {
                "row_index": row_index,
                "measured_index": None if is_warmup else len(measured_rows) - 1,
                "is_warmup": is_warmup,
            }
        )
        print(
            f"[{'WARMUP' if is_warmup else 'MEASURE'}] "
            f"{result['clip_id']} total_ms={float(result['total_time_ms']):.2f}"
        )
        if len(measured_rows) >= args.num_clips:
            break

    if not measured_rows:
        raise RuntimeError("No measured benchmark rows were collected.")

    measured_total = [float(row["total_time_ms"]) for row in measured_rows]
    measured_frame = [float(row["per_frame_total_time_ms"]) for row in measured_rows]
    read_times = [float(row["read_time_ms"]) for row in measured_rows]
    yolo_times = [float(row["yolo_time_ms"]) for row in measured_rows]
    ds_times = [float(row["deepsort_time_ms"]) for row in measured_rows]
    frame_count = sum(int(row["frame_count"]) for row in measured_rows)
    total_time_sum_ms = sum(measured_total)

    metrics = {
        "reference_label": "YOLOv8+DeepSORT",
        "split": args.split,
        "selection": args.selection,
        "seed": args.seed,
        "num_warmup": args.num_warmup,
        "num_clips_requested": args.num_clips,
        "num_clips_measured": len(measured_rows),
        "num_frames_measured": frame_count,
        "clip_time_ms_mean": summarize(measured_total)["mean"],
        "clip_time_ms_median": summarize(measured_total)["median"],
        "clip_time_ms_std": summarize(measured_total)["std"],
        "clip_time_ms_min": summarize(measured_total)["min"],
        "clip_time_ms_max": summarize(measured_total)["max"],
        "per_frame_time_ms_mean": summarize(measured_frame)["mean"],
        "per_frame_time_ms_median": summarize(measured_frame)["median"],
        "read_time_ms_mean": summarize(read_times)["mean"],
        "yolo_time_ms_mean": summarize(yolo_times)["mean"],
        "deepsort_time_ms_mean": summarize(ds_times)["mean"],
        "fps": frame_count * 1000.0 / total_time_sum_ms if total_time_sum_ms > 0 else None,
        "reference_time_ms_for_surrogate_speedup": summarize(measured_total)["median"],
        "reference_time_ms_basis": "median clip total latency over measured rows",
        "device": device_label,
        "yolo_device": yolo_device,
        "gpu_name": gpu_name(torch_module, device_label),
        "yolo": {
            "model": args.yolo_model,
            "conf": args.yolo_conf,
            "iou": args.yolo_iou,
            "imgsz": args.yolo_imgsz,
            "agnostic_nms": args.yolo_agnostic_nms,
        },
        "deepsort_config": deepsort_config,
        "deepsort_effective_kwargs": deepsort_kwargs,
        "latency_scope": "raw frame decode + YOLOv8 inference + DeepSORT update; metric computation is excluded",
    }

    output_dir = resolve_existing_path(args.output_dir)
    save_json(output_dir / "reference_metrics.json", metrics)
    write_rows(output_dir / "reference_rows.csv", measured_rows)
    print(f"[DONE] metrics: {output_dir / 'reference_metrics.json'}")
    print(f"[DONE] rows: {output_dir / 'reference_rows.csv'}")


if __name__ == "__main__":
    main()
