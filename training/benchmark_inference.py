from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset

from training.dataset import (
    _attach_visual_ood_mask_from_fusion,
    _load_runtime_cache_context,
    _masked_severity,
    build_dataloaders,
    get_split_index_path,
    get_surrogate_targets_path,
    load_normalization_stats,
    load_pt_payload,
    resolve_paths,
    resolve_targets,
)
from training.evaluate import (
    effective_model_args_for_config,
    infer_trained_swin_input_norm,
    load_checkpoint,
    normalize_run_id,
    resolve_eval_modality,
    resolve_eval_run_id,
    save_json,
)
from training.losses import extract_delta_prediction
from training.model_factory import MODEL_TO_MODALITY, create_model
from training.utils import load_state_dict_head_compatible, move_batch, require_non_empty_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark surrogate checkpoint inference latency/FPS.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_TO_MODALITY), default=None)
    parser.add_argument("--modality", choices=("param_only", "visual_only", "fusion"), default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--allow_run_id_override", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--input_source", choices=("cached", "raw_frames"), default="cached")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reference_time_ms", type=float, default=None)
    parser.add_argument("--reference_label", default="YOLOv8+DeepSORT")
    parser.add_argument(
        "--allow_cached_speedup",
        action="store_true",
        help="Allow speedup_ratio for cached-tensor surrogate latency. Required with --reference_time_ms.",
    )
    parser.add_argument(
        "--allow_amortized_speedup",
        action="store_true",
        help="Allow speedup_ratio with batch_size > 1. Otherwise speedup requires batch_size=1.",
    )
    parser.add_argument("--skip_integrity", action="store_true", help="Skip cache fingerprint/hash validation for quick smoke tests.")
    parser.add_argument("--allow_empty_eval", action="store_true", help="Allow an empty evaluation split only for smoke/debug runs.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during benchmarking.")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_index",
        "batch_size",
        "data_time_ms",
        "transfer_time_ms",
        "forward_time_ms",
        "preprocess_time_ms",
        "total_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_name(device)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise RuntimeError("--batch_size must be positive.")
    if args.num_warmup < 0:
        raise RuntimeError("--num_warmup must be non-negative.")
    if args.num_batches <= 0:
        raise RuntimeError("--num_batches must be positive.")
    if args.reference_time_ms is not None and args.reference_time_ms <= 0:
        raise RuntimeError("--reference_time_ms must be positive when provided.")
    if args.reference_time_ms is not None and args.input_source == "cached" and not args.allow_cached_speedup:
        raise RuntimeError(
            "This benchmark uses cached shared video tensors and excludes raw frame decode/resize. "
            "Pass --allow_cached_speedup to explicitly compute a cached-tensor speedup ratio, "
            "or add a raw-preprocessing benchmark before making proposal-compliant speedup claims."
        )
    if args.reference_time_ms is not None and args.batch_size != 1 and not args.allow_amortized_speedup:
        raise RuntimeError(
            "speedup_ratio is only comparable to one-sample downstream latency when --batch_size 1. "
            "Use --allow_amortized_speedup only if you intentionally want batch-amortized per-sample speedup."
        )


def read_frame_rgb(path: Path, img_size: int) -> torch.Tensor:
    try:
        import cv2  # type: ignore
        import numpy as np

        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        arr = np.transpose(frame_rgb, (2, 0, 1)).astype("float32") / 255.0
        return torch.from_numpy(arr)
    except ImportError:
        from PIL import Image
        import numpy as np

        with Image.open(path) as im:
            im = im.convert("RGB").resize((img_size, img_size))
            arr = np.transpose(np.array(im), (2, 0, 1)).astype("float32") / 255.0
            return torch.from_numpy(arr)


def ordered_frame_paths(clip_dir: Path, clip_len: int, start_frame: int, end_frame: int) -> list[Path]:
    if not clip_dir.exists():
        raise FileNotFoundError(f"Raw frame clip directory not found: {clip_dir}")
    frame_paths = []
    for path in clip_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_FRAME_EXTS:
            continue
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        if start_frame <= frame_idx <= end_frame:
            frame_paths.append((frame_idx, path))
    frame_paths.sort(key=lambda item: item[0])
    if len(frame_paths) != clip_len:
        raise RuntimeError(
            f"Expected {clip_len} raw frames in {clip_dir} for {start_frame}-{end_frame}, "
            f"found {len(frame_paths)}."
        )
    if frame_paths[0][0] != start_frame or frame_paths[-1][0] != end_frame:
        raise RuntimeError(
            f"Raw frame range mismatch in {clip_dir}: "
            f"expected {start_frame}-{end_frame}, found {frame_paths[0][0]}-{frame_paths[-1][0]}."
        )
    return [path for _, path in frame_paths]


class RawFrameSurrogateDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, Any],
        target_rows: dict[str, dict[str, Any]],
        modality: str,
        target_mode: str,
        clip_len: int,
        img_size: int,
    ) -> None:
        y, y_raw = resolve_targets(payload, target_mode)
        self.modality = modality
        self.y = y
        self.y_raw = y_raw
        self.clip_id = [str(x) for x in payload.get("clip_id", [])]
        self.original_clip_id = [str(x) for x in payload.get("original_clip_id", [])]
        self.split = [str(x) for x in payload.get("split", [])]
        self.target_rows = target_rows
        self.clip_len = clip_len
        self.img_size = img_size
        self.type_id = payload.get("type_id")
        self.severity = payload.get("severity")
        self.is_ood_masked = payload.get("is_ood_masked")
        if modality == "fusion":
            if not isinstance(self.type_id, torch.Tensor) or not isinstance(self.severity, torch.Tensor):
                raise RuntimeError("Raw fusion benchmark payload missing type_id/severity tensors.")
            if not isinstance(self.is_ood_masked, torch.Tensor):
                raise RuntimeError("Raw fusion benchmark payload missing is_ood_masked tensor.")

    def __len__(self) -> int:
        return len(self.clip_id)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id = self.clip_id[idx]
        row = self.target_rows.get(clip_id)
        if row is None:
            raise RuntimeError(f"Missing surrogate target row for clip_id={clip_id}")
        clip_dir_raw = str(row.get("abs_file_path") or row.get("file_path") or "").strip()
        if not clip_dir_raw:
            raise RuntimeError(f"Missing file_path for clip_id={clip_id}")
        clip_dir = Path(clip_dir_raw)
        if not clip_dir.is_absolute():
            clip_dir = PROJECT_ROOT / clip_dir
        start_frame = int(float(row["start_frame"]))
        end_frame = int(float(row["end_frame"]))
        frame_paths = ordered_frame_paths(clip_dir, self.clip_len, start_frame, end_frame)
        video = torch.stack([read_frame_rgb(path, self.img_size) for path in frame_paths], dim=0)
        item: dict[str, Any] = {
            "video": video,
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "clip_id": clip_id,
            "original_clip_id": self.original_clip_id[idx],
            "split": self.split[idx],
        }
        if isinstance(self.is_ood_masked, torch.Tensor):
            item["is_ood_masked"] = self.is_ood_masked[idx]
        if self.modality == "fusion":
            assert isinstance(self.type_id, torch.Tensor)
            assert isinstance(self.severity, torch.Tensor)
            assert isinstance(self.is_ood_masked, torch.Tensor)
            severity = _masked_severity(self.type_id[idx], self.severity[idx], self.is_ood_masked[idx])
            item["type_id"] = self.type_id[idx]
            item["severity"] = severity
            item["param"] = torch.stack((self.type_id[idx].float(), severity.float()), dim=-1)
        return item


def build_raw_frame_loader(
    modality: str,
    split: str,
    target_mode: str,
    run_id: str | None,
    config_path: str,
    batch_size: int,
    num_workers: int,
    validate_integrity: bool,
) -> DataLoader:
    if modality == "param_only":
        raise RuntimeError("--input_source raw_frames is only valid for visual_only/fusion models.")
    paths = resolve_paths(config_path=config_path, run_id=run_id)
    if validate_integrity:
        from training.dataset import validate_runtime_integrity

        validate_runtime_integrity(paths=paths, run_id=run_id, modality=modality, split=split)
    payload = load_pt_payload(get_split_index_path(paths, run_id, modality, split))
    if modality == "visual_only":
        payload = _attach_visual_ood_mask_from_fusion(payload, paths, run_id, split)
    build_cfg = _load_runtime_cache_context(paths, run_id)
    import pandas as pd

    target_df = pd.read_csv(get_surrogate_targets_path(paths, run_id))
    target_rows = {str(row["clip_id"]): row for row in target_df.to_dict(orient="records")}
    dataset = RawFrameSurrogateDataset(
        payload=payload,
        target_rows=target_rows,
        modality=modality,
        target_mode=target_mode,
        clip_len=int(build_cfg["clip_len"]),
        img_size=int(build_cfg["img_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def load_model_for_benchmark(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any], str, str, str | None, dict[str, Any]]:
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
    model_args["swin_pretrained"] = False
    model_args["swin_input_norm"] = infer_trained_swin_input_norm(ckpt_model_args)
    model_args["freeze_early_layers"] = False

    normalization_stats = checkpoint.get("normalization_stats")
    if not isinstance(normalization_stats, dict):
        normalization_stats = load_normalization_stats(config_path=args.config, run_id=run_id)

    device = torch.device(args.device)
    model = create_model(model_name, model_args, normalization_stats).to(device)
    load_state_dict_head_compatible(model, checkpoint["model_state"])
    model.eval()

    metadata = {
        "checkpoint_run_id": normalize_run_id(checkpoint_run_id),
        "cli_run_id": normalize_run_id(args.run_id),
        "run_id_override_allowed": bool(args.allow_run_id_override),
        "run_id_override_used": normalize_run_id(args.run_id) != normalize_run_id(checkpoint_run_id)
        if args.run_id is not None
        else False,
        "effective_model_args": effective_model_args_for_config(model_args),
    }
    return model, metadata, model_name, modality, run_id, {"target_mode": target_mode}


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    model, metadata, model_name, modality, run_id, mode_info = load_model_for_benchmark(args)

    if args.input_source == "raw_frames":
        loader = build_raw_frame_loader(
            modality=modality,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_mode=str(mode_info["target_mode"]),
            run_id=run_id,
            config_path=args.config,
            validate_integrity=not args.skip_integrity,
        )
    else:
        loader = build_dataloaders(
            modality=modality,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_mode=str(mode_info["target_mode"]),
            run_id=run_id,
            config_path=args.config,
            validate_integrity=not args.skip_integrity,
            shuffle=False,
        )
    require_non_empty_loader(loader, args.split, allow_empty=args.allow_empty_eval)

    rows: list[dict[str, Any]] = []
    total_iters = args.num_warmup + args.num_batches
    iterator = iter(loader)
    use_amp = bool(args.amp and device.type == "cuda")

    with torch.no_grad():
        for batch_index in range(total_iters):
            iter_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_end = time.perf_counter()

            transfer_start = time.perf_counter()
            batch_device = move_batch(batch, device)
            sync_if_needed(device)
            transfer_end = time.perf_counter()
            forward_start = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(batch_device)
                extract_delta_prediction(outputs)
            sync_if_needed(device)
            forward_end = time.perf_counter()

            if batch_index < args.num_warmup:
                continue

            batch_count = int(batch["y"].shape[0]) if isinstance(batch.get("y"), torch.Tensor) else args.batch_size
            data_ms = (data_end - iter_start) * 1000.0
            transfer_ms = (transfer_end - transfer_start) * 1000.0
            forward_ms = (forward_end - forward_start) * 1000.0
            total_ms = (forward_end - iter_start) * 1000.0
            rows.append(
                {
                    "batch_index": batch_index - args.num_warmup,
                    "batch_size": batch_count,
                    "data_time_ms": data_ms,
                    "transfer_time_ms": transfer_ms,
                    "forward_time_ms": forward_ms,
                    "preprocess_time_ms": data_ms + transfer_ms,
                    "total_time_ms": total_ms,
                }
            )

    if not rows:
        raise RuntimeError("No benchmark rows were collected. Reduce --num_warmup or check split size.")

    num_samples = sum(int(row["batch_size"]) for row in rows)
    total_time_ms = sum(float(row["total_time_ms"]) for row in rows)
    fps = (num_samples * 1000.0 / total_time_ms) if total_time_ms > 0 else None
    total_time_ms_mean = mean([float(row["total_time_ms"]) for row in rows])
    per_sample_total_time_ms_mean = (total_time_ms / num_samples) if num_samples > 0 else None
    per_sample_forward_time_ms_mean = (
        sum(float(row["forward_time_ms"]) for row in rows) / num_samples
        if num_samples > 0
        else None
    )
    per_sample_preprocess_time_ms_mean = (
        sum(float(row["preprocess_time_ms"]) for row in rows) / num_samples
        if num_samples > 0
        else None
    )
    reference_time_ms = args.reference_time_ms
    speedup_ratio = None
    if reference_time_ms is not None and per_sample_total_time_ms_mean is not None and per_sample_total_time_ms_mean > 0:
        speedup_ratio = reference_time_ms / per_sample_total_time_ms_mean

    metrics = {
        "model": model_name,
        "modality": modality,
        "split": args.split,
        "target_mode": mode_info["target_mode"],
        "run_id": run_id,
        "batch_size": args.batch_size,
        "num_warmup": args.num_warmup,
        "num_batches_requested": args.num_batches,
        "num_batches_measured": len(rows),
        "num_samples": num_samples,
        "data_time_ms_mean": mean([float(row["data_time_ms"]) for row in rows]),
        "transfer_time_ms_mean": mean([float(row["transfer_time_ms"]) for row in rows]),
        "forward_time_ms_mean": mean([float(row["forward_time_ms"]) for row in rows]),
        "preprocess_time_ms_mean": mean([float(row["preprocess_time_ms"]) for row in rows]),
        "total_time_ms_mean": total_time_ms_mean,
        "per_sample_preprocess_time_ms_mean": per_sample_preprocess_time_ms_mean,
        "per_sample_forward_time_ms_mean": per_sample_forward_time_ms_mean,
        "per_sample_total_time_ms_mean": per_sample_total_time_ms_mean,
        "fps": fps if fps is None or math.isfinite(fps) else None,
        "device": str(device),
        "gpu_name": device_name(device),
        "amp": bool(args.amp),
        "reference_time_ms": reference_time_ms,
        "reference_label": args.reference_label,
        "speedup_ratio": speedup_ratio,
        "speedup_ratio_latency_basis": "per_sample_total_time_ms_mean",
        "allow_cached_speedup": bool(args.allow_cached_speedup),
        "allow_amortized_speedup": bool(args.allow_amortized_speedup),
        "benchmark_input_source": "raw_clip_frames" if args.input_source == "raw_frames" else "cached_shared_video_tensor",
        "includes_raw_frame_decode_resize": args.input_source == "raw_frames",
        "includes_cache_load": args.input_source == "cached",
        "includes_device_transfer": True,
        "proposal_batch_size_compliant": args.batch_size == 1,
        "proposal_raw_preprocessing_compliant": args.input_source == "raw_frames",
        "proposal_efficiency_compliant": bool(args.input_source == "raw_frames" and args.batch_size == 1),
        "latency_scope": (
            "raw frame decode/resize + host/device transfer + model forward"
            if args.input_source == "raw_frames"
            else "cached shared video tensor load + host/device transfer + model forward; raw frame decode/resize is excluded"
        ),
        **metadata,
    }

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"benchmark_{args.split}"
    save_json(out_dir / "benchmark_metrics.json", metrics)
    write_rows(out_dir / "benchmark_rows.csv", rows)
    print(f"[DONE] metrics: {out_dir / 'benchmark_metrics.json'}")
    print(f"[DONE] rows: {out_dir / 'benchmark_rows.csv'}")


if __name__ == "__main__":
    main()
