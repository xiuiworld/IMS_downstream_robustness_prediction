#!/usr/bin/env python3
# Colab install example:
# !pip install -q ultralytics deep-sort-realtime lap opencv-python pandas numpy pyyaml tqdm scipy

import argparse
import configparser
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import cv2
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as import_err:
    print("[ERROR] Missing required packages.")
    print(
        "Install with: pip install ultralytics deep-sort-realtime lap "
        "opencv-python pandas numpy pyyaml tqdm scipy"
    )
    raise import_err


TARGET_COLUMNS = [
    "clip_id",
    "original_clip_id",
    "degradation_type",
    "degradation_param",
    "p_orig_map",
    "p_orig_hota",
    "p_anon_map",
    "p_anon_hota",
    "delta_map",
    "delta_hota",
]

CACHE_COLUMNS = ["original_clip_id", "p_orig_map", "p_orig_hota"]

STATS_COLUMNS = [
    "split",
    "total_candidates",
    "included_targets",
    "reused_completed_count",
    "excluded_low_active_trajectories",
    "excluded_low_baseline",
    "excluded_empty_gt",
    "excluded_missing_baseline",
    "excluded_eval_failure",
    "zero_clipped_count",
    "zero_clipped_ratio",
    "exclusion_ratio",
]

FAILURE_COLUMNS = [
    "timestamp",
    "stage",
    "clip_id",
    "original_clip_id",
    "degradation_type",
    "degradation_param",
    "error_type",
    "error_message",
]

GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "class",
    "vis",
]

EMPTY_GT_SENTINEL = "empty_gt_eval_target"
ORIG_STATE_READY = "orig_ready"
ORIG_STATE_LOW_ACTIVE = "orig_low_active"
ORIG_STATE_CONTRACT_VIOLATION = "orig_contract_violation"
ORIG_STATE_EMPTY_GT = "orig_empty_gt"
ORIG_STATE_EVAL_FAILURE = "orig_eval_failure"

ZERO_CLIPPED_POLICY = "include_and_report"
STATS_MODE = "strict_full_run"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Path, path_like: str) -> Path:
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    return (project_root / path_obj).resolve()


def parse_sequence_info(seq_dir: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    seq = parser["Sequence"]
    return {
        "im_ext": seq.get("imExt", ".jpg"),
        "im_width": int(seq.get("imWidth", 1920)),
        "im_height": int(seq.get("imHeight", 1080)),
        "frame_rate": int(seq.get("frameRate", 30)),
    }


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def normalize_param_key(value) -> str:
    v = to_float(value)
    if v is None:
        text = str(value).strip()
        return text.lower()
    return f"{v:.12g}"


def normalize_split_name(value) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "unknown"
    return text


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def safe_error_message(exc: Exception, limit: int = 400) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


def is_empty_gt_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and str(exc) == EMPTY_GT_SENTINEL


def append_failure(
    failure_records: List[dict],
    stage: str,
    clip_id: str,
    original_clip_id: str,
    degradation_type: str,
    degradation_param,
    exc: Exception,
) -> None:
    failure_records.append(
        {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "stage": stage,
            "clip_id": clip_id,
            "original_clip_id": original_clip_id,
            "degradation_type": degradation_type,
            "degradation_param": degradation_param,
            "error_type": type(exc).__name__,
            "error_message": safe_error_message(exc),
        }
    )


def read_csv_or_empty(path: Path, columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]


def atomic_write_csv(df: pd.DataFrame, path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columns]
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def resolve_yolo_weights_path(model_arg: str, model: Optional[YOLO]) -> Optional[Path]:
    # 1) direct local path from CLI arg
    arg_path = Path(str(model_arg)).expanduser()
    if arg_path.is_file():
        return arg_path.resolve()

    # 2) fallback: inspect resolved checkpoint path from loaded YOLO object
    candidates: List[str] = []
    for attr in ["ckpt_path", "weights", "pt_path"]:
        value = getattr(model, attr, None) if model is not None else None
        if isinstance(value, str) and value.strip():
            candidates.append(value)

    inner_model = getattr(model, "model", None) if model is not None else None
    for attr in ["pt_path", "ckpt_path", "weights"]:
        value = getattr(inner_model, attr, None) if inner_model is not None else None
        if isinstance(value, str) and value.strip():
            candidates.append(value)

    for c in candidates:
        c_path = Path(c).expanduser()
        if c_path.is_file():
            return c_path.resolve()
    return None


def build_clip_manifest_identity(clip_manifest_path: Path, clip_df: pd.DataFrame) -> dict:
    key_columns = ["clip_id", "sequence_name", "split", "start_frame", "end_frame", "file_path"]
    missing_columns = [c for c in key_columns if c not in clip_df.columns]
    key_df = clip_df.copy()
    for col in missing_columns:
        key_df[col] = ""

    # Deterministic key-fingerprint over canonical key columns.
    key_df = key_df[key_columns].fillna("")
    hasher = hashlib.sha256()
    for row in key_df.itertuples(index=False):
        line = "\x1f".join([str(v).strip() for v in row]) + "\n"
        hasher.update(line.encode("utf-8", errors="replace"))

    return {
        "clip_manifest_sha256": compute_file_sha256(clip_manifest_path),
        "clip_manifest_row_count": int(len(clip_df)),
        "clip_manifest_key_sha256": hasher.hexdigest(),
        "clip_manifest_missing_key_columns": missing_columns,
    }


def build_provenance_identity(
    script_path: Path,
    yolo_weights_path: Optional[Path],
    deep_sort_version: str,
) -> dict:
    payload = {
        "script_sha256": compute_file_sha256(script_path),
        "yolo_weights_resolved_path": str(yolo_weights_path) if yolo_weights_path is not None else "",
        "yolo_weights_sha256": compute_file_sha256(yolo_weights_path) if yolo_weights_path is not None else "",
        "deep_sort_realtime_version": deep_sort_version,
    }
    return payload


def empty_split_stats() -> dict:
    return {
        "total_candidates": 0,
        "included_targets": 0,
        "reused_completed_count": 0,
        "excluded_low_active_trajectories": 0,
        "excluded_low_baseline": 0,
        "excluded_empty_gt": 0,
        "excluded_missing_baseline": 0,
        "excluded_eval_failure": 0,
        "zero_clipped_count": 0,
    }


def increment_split_stat(split_stats: Dict[str, dict], split_name: str, field: str, amount: int = 1) -> None:
    if split_name not in split_stats:
        split_stats[split_name] = empty_split_stats()
    split_stats[split_name][field] += amount


def build_target_stats_df(split_stats: Dict[str, dict]) -> pd.DataFrame:
    rows: List[dict] = []
    aggregate = empty_split_stats()

    for split in sorted(split_stats.keys()):
        stats = split_stats[split]
        excluded_total = (
            stats["excluded_low_active_trajectories"]
            + stats["excluded_low_baseline"]
            + stats["excluded_empty_gt"]
            + stats["excluded_missing_baseline"]
            + stats["excluded_eval_failure"]
        )
        total_candidates = stats["total_candidates"]
        included_targets = stats["included_targets"]
        reused_completed = stats["reused_completed_count"]
        effective_included = included_targets + reused_completed
        zero_clipped = stats["zero_clipped_count"]

        row = {
            "split": split,
            "total_candidates": total_candidates,
            "included_targets": included_targets,
            "reused_completed_count": reused_completed,
            "excluded_low_active_trajectories": stats["excluded_low_active_trajectories"],
            "excluded_low_baseline": stats["excluded_low_baseline"],
            "excluded_empty_gt": stats["excluded_empty_gt"],
            "excluded_missing_baseline": stats["excluded_missing_baseline"],
            "excluded_eval_failure": stats["excluded_eval_failure"],
            "zero_clipped_count": zero_clipped,
            "zero_clipped_ratio": (zero_clipped / effective_included) if effective_included > 0 else 0.0,
            "exclusion_ratio": (excluded_total / total_candidates) if total_candidates > 0 else 0.0,
        }
        rows.append(row)
        for key in aggregate:
            aggregate[key] += stats[key]

    excluded_total_global = (
        aggregate["excluded_low_active_trajectories"]
        + aggregate["excluded_low_baseline"]
        + aggregate["excluded_empty_gt"]
        + aggregate["excluded_missing_baseline"]
        + aggregate["excluded_eval_failure"]
    )
    total_global = aggregate["total_candidates"]
    included_global = aggregate["included_targets"]
    reused_global = aggregate["reused_completed_count"]
    effective_included_global = included_global + reused_global
    zero_global = aggregate["zero_clipped_count"]

    rows.append(
        {
            "split": "GLOBAL",
            "total_candidates": total_global,
            "included_targets": included_global,
            "reused_completed_count": reused_global,
            "excluded_low_active_trajectories": aggregate["excluded_low_active_trajectories"],
            "excluded_low_baseline": aggregate["excluded_low_baseline"],
            "excluded_empty_gt": aggregate["excluded_empty_gt"],
            "excluded_missing_baseline": aggregate["excluded_missing_baseline"],
            "excluded_eval_failure": aggregate["excluded_eval_failure"],
            "zero_clipped_count": zero_global,
            "zero_clipped_ratio": (zero_global / effective_included_global) if effective_included_global > 0 else 0.0,
            "exclusion_ratio": (excluded_total_global / total_global) if total_global > 0 else 0.0,
        }
    )

    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def validate_clip_frames(clip_dir: Path, start_frame: int, end_frame: int, im_ext: str) -> None:
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")
    if end_frame < start_frame:
        raise ValueError("contract_violation: invalid frame range")

    expected_names = [f"{f:06d}{im_ext}" for f in range(start_frame, end_frame + 1)]
    missing = [name for name in expected_names if not (clip_dir / name).exists()]
    if missing:
        raise ValueError(f"contract_violation: missing expected frame(s): {missing[:5]}")

    actual_names = sorted([p.name for p in clip_dir.glob(f"*{im_ext}") if p.is_file()])
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    if actual_set != expected_set:
        unexpected = sorted(list(actual_set - expected_set))
        missing_again = sorted(list(expected_set - actual_set))
        raise ValueError(
            "contract_violation: frame naming mismatch. "
            f"unexpected={unexpected[:5]}, missing={missing_again[:5]}"
        )


def get_trackeval_commit_hash(trackeval_dir: Path) -> str:
    if not trackeval_dir.exists():
        return "missing"
    if shutil.which("git") is None:
        return "unknown_git_unavailable"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(trackeval_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def with_semantic_run_id(path: Path, semantic_run_id: Optional[str]) -> Path:
    if not semantic_run_id:
        return path
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", semantic_run_id)
    if safe_id == "":
        return path
    return path.with_name(f"{path.stem}_{safe_id}{path.suffix}")


def build_deepsort_config(args: argparse.Namespace) -> dict:
    return {
        "max_age": args.ds_max_age,
        "n_init": args.ds_n_init,
        "nms_max_overlap": args.ds_nms_max_overlap,
        "max_iou_distance": args.ds_max_iou_distance,
        "max_cosine_distance": args.ds_max_cosine_distance,
        "nn_budget": args.ds_nn_budget,
        "embedder": args.ds_embedder,
        "embedder_gpu": args.ds_embedder_gpu,
        "half": args.ds_half,
        "bgr": args.ds_bgr,
        "embedder_model_name": args.ds_embedder_model_name,
        "embedder_wts": args.ds_embedder_wts,
    }


def resolve_deepsort_effective_kwargs(deepsort_config: dict, strict: bool = True) -> dict:
    sig = inspect.signature(DeepSort.__init__)
    accepted = {k for k in sig.parameters.keys() if k != "self"}
    unsupported = [k for k, v in deepsort_config.items() if v is not None and k not in accepted]
    if strict and unsupported:
        raise RuntimeError(
            "DeepSort config contains unsupported parameters for installed deep-sort-realtime: "
            f"{sorted(unsupported)}"
        )

    kwargs = {}
    for key, value in deepsort_config.items():
        if key in accepted and value is not None:
            kwargs[key] = value
    return kwargs


def create_deepsort_tracker(deepsort_effective_kwargs: dict) -> DeepSort:
    kwargs = dict(deepsort_effective_kwargs)
    return DeepSort(**kwargs)


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def select_semantic_snapshot_source(
    canonical_path: Path, partial_path: Path
) -> Tuple[Optional[dict], Optional[Path], Optional[str]]:
    candidates: List[Tuple[float, str, Path, dict]] = []
    for source_name, source_path in [("canonical", canonical_path), ("partial", partial_path)]:
        payload = load_json_if_exists(source_path)
        if payload is None:
            if source_path.exists():
                print(f"[WARNING] Ignoring invalid snapshot JSON: {source_path}")
            continue
        if not isinstance(payload, dict):
            print(f"[WARNING] Ignoring snapshot with non-dict payload: {source_path}")
            continue
        try:
            mtime = source_path.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((mtime, source_name, source_path, payload))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, source_name, source_path, payload = candidates[0]
    return payload, source_path, source_name


def build_semantic_signature(
    args: argparse.Namespace,
    map_definition: str,
    trackeval_commit: str,
    manifest_identity: dict,
    provenance_identity: dict,
    deep_sort_config: dict,
    deep_sort_effective_kwargs: dict,
    execution_scope: str,
    is_partial_run: bool,
    zero_clipped_policy: str,
    stats_mode: str,
) -> dict:
    return {
        "map_definition": map_definition,
        "manifest_identity": manifest_identity,
        "provenance_identity": provenance_identity,
        "execution_scope": execution_scope,
        "is_partial_run": bool(is_partial_run),
        "zero_clipped_policy": zero_clipped_policy,
        "stats_mode": stats_mode,
        "yolo": {
            "model": args.yolo_model,
            "conf": args.yolo_conf,
            "iou": args.yolo_iou,
            "imgsz": args.yolo_imgsz,
            "device": args.yolo_device,
            "agnostic_nms": args.yolo_agnostic_nms,
        },
        "trackeval": {
            "requested_ref": args.trackeval_ref,
            "resolved_commit": trackeval_commit,
        },
        "filters": {
            "min_active_trajectories": args.min_active_trajectories,
            "expected_clip_length": args.expected_clip_length,
            "min_orig_threshold": args.min_orig_threshold,
        },
        "deep_sort_config": deep_sort_config,
        "deep_sort_effective_kwargs": deep_sort_effective_kwargs,
    }


def compare_nested_dict(old_value, new_value, prefix: str = "") -> List[str]:
    diffs: List[str] = []
    if isinstance(old_value, dict) and isinstance(new_value, dict):
        keys = sorted(set(old_value.keys()) | set(new_value.keys()))
        for key in keys:
            p = f"{prefix}.{key}" if prefix else key
            if key not in old_value:
                diffs.append(f"{p}: <missing> -> {new_value[key]}")
            elif key not in new_value:
                diffs.append(f"{p}: {old_value[key]} -> <missing>")
            else:
                diffs.extend(compare_nested_dict(old_value[key], new_value[key], p))
        return diffs
    if old_value != new_value:
        diffs.append(f"{prefix}: {old_value} -> {new_value}")
    return diffs


def write_run_config_snapshot(
    path: Path,
    args: argparse.Namespace,
    trackeval_dir: Path,
    map_definition: str,
    manifest_identity: dict,
    provenance_identity: dict,
    semantic_signature: dict,
    execution_scope: str,
    is_partial_run: bool,
    zero_clipped_policy: str,
    stats_mode: str,
    deep_sort_effective_kwargs: dict,
) -> None:
    ultralytics_module = sys.modules.get("ultralytics")
    snapshot = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "script": "03_generate_targets.py",
        "map_definition": map_definition,
        "map_caveat": "AP is computed on detections that pass YOLO confidence threshold (--yolo_conf).",
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": getattr(cv2, "__version__", "unknown"),
            "ultralytics": getattr(ultralytics_module, "__version__", "unknown"),
        },
        "trackeval": {
            "requested_ref": args.trackeval_ref,
            "resolved_commit": get_trackeval_commit_hash(trackeval_dir),
            "path": str(trackeval_dir),
        },
        "manifest_identity": manifest_identity,
        "provenance_identity": provenance_identity,
        "execution": {
            "execution_scope": execution_scope,
            "is_partial_run": bool(is_partial_run),
            "zero_clipped_policy": zero_clipped_policy,
            "stats_mode": stats_mode,
        },
        "deep_sort_effective_kwargs": deep_sort_effective_kwargs,
        "semantic_signature": semantic_signature,
        "arguments": vars(args),
    }
    atomic_write_json(snapshot, path)


def detect_drive_manifest_dir() -> Optional[Path]:
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return None

    patterns = [
        "downstream_robustness_prediction/data/interim/manifests",
        "*/downstream_robustness_prediction/data/interim/manifests",
        "*/*/downstream_robustness_prediction/data/interim/manifests",
        "*/*/*/downstream_robustness_prediction/data/interim/manifests",
        "*/*/*/*/downstream_robustness_prediction/data/interim/manifests",
    ]
    for pattern in patterns:
        matches = sorted(drive_root.glob(pattern))
        for match in matches:
            if match.is_dir():
                return match.resolve()
    return None


def validate_drive_manifest_dir(candidate: Path, strict_drive_mount: bool) -> Optional[Path]:
    drive_root = Path("/content/drive/MyDrive")
    candidate = candidate.expanduser().resolve()

    if strict_drive_mount:
        if not drive_root.exists():
            print("[WARNING] Drive root '/content/drive/MyDrive' is not mounted. Falling back to local-only mode.")
            return None
        try:
            candidate.relative_to(drive_root)
        except ValueError:
            print(f"[WARNING] Drive sync path is outside mounted Drive: {candidate}. Falling back to local-only mode.")
            return None

    return candidate


def resolve_drive_manifest_dir(args_drive_sync_dir: Optional[str], strict_drive_mount: bool) -> Optional[Path]:
    if args_drive_sync_dir:
        return validate_drive_manifest_dir(Path(args_drive_sync_dir), strict_drive_mount)

    env_dir = os.environ.get("TARGET_MANIFEST_SYNC_DIR", "").strip()
    if env_dir:
        return validate_drive_manifest_dir(Path(env_dir), strict_drive_mount)

    detected = detect_drive_manifest_dir()
    if detected is None:
        return None
    return validate_drive_manifest_dir(detected, strict_drive_mount)


class SyncManager:
    def __init__(
        self,
        local_target_path: Path,
        local_cache_path: Path,
        local_failure_path: Path,
        local_stats_path: Path,
        local_snapshot_path: Path,
        local_snapshot_partial_path: Path,
        drive_manifest_dir: Optional[Path],
    ) -> None:
        self.local_target_path = local_target_path
        self.local_cache_path = local_cache_path
        self.local_failure_path = local_failure_path
        self.local_stats_path = local_stats_path
        self.local_snapshot_path = local_snapshot_path
        self.local_snapshot_partial_path = local_snapshot_partial_path
        self.drive_manifest_dir = drive_manifest_dir
        self.drive_logs_dir = None
        self.drive_target_path = None
        self.drive_cache_path = None
        self.drive_failure_path = None
        self.drive_stats_path = None
        self.drive_snapshot_path = None
        self.drive_snapshot_partial_path = None

        if drive_manifest_dir is not None:
            try:
                self.drive_manifest_dir.mkdir(parents=True, exist_ok=True)
                self.drive_logs_dir = self.drive_manifest_dir.parent / "logs"
                self.drive_logs_dir.mkdir(parents=True, exist_ok=True)
                self.drive_target_path = self.drive_manifest_dir / self.local_target_path.name
                self.drive_cache_path = self.drive_manifest_dir / self.local_cache_path.name
                self.drive_failure_path = self.drive_logs_dir / self.local_failure_path.name
                self.drive_stats_path = self.drive_manifest_dir / self.local_stats_path.name
                self.drive_snapshot_path = self.drive_manifest_dir / self.local_snapshot_path.name
                self.drive_snapshot_partial_path = self.drive_manifest_dir / self.local_snapshot_partial_path.name
            except Exception as sync_dir_err:  # noqa: BLE001
                print(
                    "[WARNING] Failed to initialize Drive sync directories: "
                    f"{safe_error_message(sync_dir_err)}. Falling back to local-only mode."
                )
                self.drive_manifest_dir = None
                self.drive_logs_dir = None
                self.drive_target_path = None
                self.drive_cache_path = None
                self.drive_failure_path = None
                self.drive_stats_path = None
                self.drive_snapshot_path = None
                self.drive_snapshot_partial_path = None

    def _disable_drive_sync(self, reason: str) -> None:
        print(f"[WARNING] Disabling Drive sync and switching to local-only mode: {reason}")
        self.drive_manifest_dir = None
        self.drive_logs_dir = None
        self.drive_target_path = None
        self.drive_cache_path = None
        self.drive_failure_path = None
        self.drive_stats_path = None
        self.drive_snapshot_path = None
        self.drive_snapshot_partial_path = None

    def _copy_if_drive_newer(self, local_path: Path, drive_path: Optional[Path], label: str) -> None:
        if drive_path is None or not drive_path.exists():
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            atomic_copy_file(drive_path, local_path)
            print(f"[INFO] Recovered {label} from Drive.")
            return
        if drive_path.stat().st_mtime > local_path.stat().st_mtime:
            atomic_copy_file(drive_path, local_path)
            print(f"[INFO] Updated local {label} with newer Drive copy.")

    def recover_latest_to_local(self) -> None:
        if self.drive_manifest_dir is None:
            print("[WARNING] Drive sync path not configured. Running in local-only mode.")
            return
        recovery_items = [
            ("target_manifest.csv", self.local_target_path, self.drive_target_path),
            ("original_performance_cache.csv", self.local_cache_path, self.drive_cache_path),
            ("target_failures.csv", self.local_failure_path, self.drive_failure_path),
            ("target_stats.csv", self.local_stats_path, self.drive_stats_path),
        ]
        for label, local_path, drive_path in recovery_items:
            try:
                self._copy_if_drive_newer(local_path, drive_path, label)
            except Exception as recovery_err:  # noqa: BLE001
                self._disable_drive_sync(
                    f"Drive recovery failed for {label}: {safe_error_message(recovery_err)}"
                )
                break

    def recover_snapshot_to_local(self) -> None:
        if self.drive_manifest_dir is None:
            return
        recovery_items = [
            (self.local_snapshot_path, self.drive_snapshot_path),
            (self.local_snapshot_partial_path, self.drive_snapshot_partial_path),
        ]
        for local_path, drive_path in recovery_items:
            try:
                self._copy_if_drive_newer(local_path, drive_path, local_path.name)
            except Exception as recovery_err:  # noqa: BLE001
                self._disable_drive_sync(
                    f"Drive recovery failed for {local_path.name}: {safe_error_message(recovery_err)}"
                )
                break

    def sync_to_drive(self, include_stats: bool = False) -> None:
        if self.drive_manifest_dir is None:
            return

        to_sync = [
            ("target_manifest.csv", self.local_target_path, self.drive_target_path),
            ("original_performance_cache.csv", self.local_cache_path, self.drive_cache_path),
            ("target_failures.csv", self.local_failure_path, self.drive_failure_path),
        ]
        if include_stats:
            to_sync.append(("target_stats.csv", self.local_stats_path, self.drive_stats_path))
        for label, local_path, drive_path in to_sync:
            if drive_path is None or not local_path.exists():
                continue
            try:
                atomic_copy_file(local_path, drive_path)
            except Exception as sync_err:  # noqa: BLE001
                print(f"[WARNING] Drive sync failed for {label}: {safe_error_message(sync_err)}")

    def save_and_sync(
        self,
        target_df: pd.DataFrame,
        cache_df: pd.DataFrame,
        failure_df: pd.DataFrame,
    ) -> None:
        atomic_write_csv(target_df, self.local_target_path, TARGET_COLUMNS)
        atomic_write_csv(cache_df, self.local_cache_path, CACHE_COLUMNS)
        atomic_write_csv(failure_df, self.local_failure_path, FAILURE_COLUMNS)
        self.sync_to_drive(include_stats=False)

    def save_stats_and_sync(self, stats_df: pd.DataFrame) -> None:
        atomic_write_csv(stats_df, self.local_stats_path, STATS_COLUMNS)
        self.sync_to_drive(include_stats=True)

    def sync_snapshot_to_drive(self) -> None:
        if self.drive_manifest_dir is None:
            return
        if not self.local_snapshot_path.exists() or self.drive_snapshot_path is None:
            return
        try:
            atomic_copy_file(self.local_snapshot_path, self.drive_snapshot_path)
        except Exception as sync_err:  # noqa: BLE001
            print(f"[WARNING] Drive sync failed for {self.local_snapshot_path.name}: {safe_error_message(sync_err)}")

    def sync_snapshot_partial_to_drive(self) -> None:
        if self.drive_manifest_dir is None:
            return
        if not self.local_snapshot_partial_path.exists() or self.drive_snapshot_partial_path is None:
            return
        try:
            atomic_copy_file(self.local_snapshot_partial_path, self.drive_snapshot_partial_path)
        except Exception as sync_err:  # noqa: BLE001
            print(
                "[WARNING] Drive sync failed for "
                f"{self.local_snapshot_partial_path.name}: {safe_error_message(sync_err)}"
            )

    def cleanup_partial_snapshot_best_effort(self) -> None:
        try:
            if self.local_snapshot_partial_path.exists():
                self.local_snapshot_partial_path.unlink()
        except Exception as cleanup_err:  # noqa: BLE001
            print(
                "[WARNING] Failed to remove local partial snapshot "
                f"{self.local_snapshot_partial_path}: {safe_error_message(cleanup_err)}"
            )
        if self.drive_snapshot_partial_path is not None:
            try:
                if self.drive_snapshot_partial_path.exists():
                    self.drive_snapshot_partial_path.unlink()
            except Exception as cleanup_err:  # noqa: BLE001
                print(
                    "[WARNING] Failed to remove Drive partial snapshot "
                    f"{self.drive_snapshot_partial_path}: {safe_error_message(cleanup_err)}"
                )

    def sync_manifest_file_best_effort(self, local_file_path: Path) -> None:
        if self.drive_manifest_dir is None:
            return
        if not local_file_path.exists():
            return
        drive_path = self.drive_manifest_dir / local_file_path.name
        try:
            atomic_copy_file(local_file_path, drive_path)
        except Exception as sync_err:  # noqa: BLE001
            print(f"[WARNING] Drive sync failed for {local_file_path.name}: {safe_error_message(sync_err)}")


def ensure_trackeval(trackeval_dir: Path, trackeval_ref: Optional[str] = None) -> None:
    run_script = trackeval_dir / "scripts" / "run_mot_challenge.py"
    if run_script.exists():
        pass
    else:
        if shutil.which("git") is None:
            raise RuntimeError("TrackEval not found and git is unavailable to clone it.")

        print(f"[INFO] Cloning TrackEval into {trackeval_dir} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/JonathonLuiten/TrackEval.git", str(trackeval_dir)],
            check=True,
        )
        if not run_script.exists():
            raise RuntimeError("TrackEval clone completed but run_mot_challenge.py was not found.")

    if trackeval_ref:
        if shutil.which("git") is None:
            raise RuntimeError("git is unavailable to checkout requested --trackeval_ref")
        subprocess.run(["git", "fetch", "--all", "--tags"], cwd=str(trackeval_dir), check=False)
        subprocess.run(["git", "checkout", trackeval_ref], cwd=str(trackeval_dir), check=True)


def compute_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def calculate_voc2007_ap50(predictions: List[dict], ground_truths: List[dict]) -> float:
    if len(ground_truths) == 0 or len(predictions) == 0:
        return 0.0

    gt_by_frame: Dict[int, List[dict]] = {}
    for gt in ground_truths:
        frame = int(gt["frame"])
        gt_by_frame.setdefault(frame, []).append({"bbox": gt["bbox"], "matched": False})

    predictions = sorted(predictions, key=lambda x: float(x["conf"]), reverse=True)
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for i, pred in enumerate(predictions):
        frame = int(pred["frame"])
        pred_box = pred["bbox"]
        candidates = gt_by_frame.get(frame, [])

        best_iou = 0.0
        best_idx = -1
        for j, gt in enumerate(candidates):
            if gt["matched"]:
                continue
            iou = compute_iou_xyxy(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0 and best_iou >= 0.5:
            tp[i] = 1.0
            candidates[best_idx]["matched"] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(float(len(ground_truths)), 1e-12)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = 0.0
    for thresh in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= thresh
        precision_at_recall = np.max(precisions[mask]) if np.any(mask) else 0.0
        ap += precision_at_recall / 11.0
    return float(ap)


def sanitize_seq_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def parse_hota_summary(summary_path: Path) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"HOTA summary file not found: {summary_path}")

    for sep in (r"\s+", ","):
        try:
            df = pd.read_csv(summary_path, sep=sep, engine="python")
            if "HOTA" not in df.columns or len(df) == 0:
                continue
            if df.shape[1] > 0:
                first_col = df.columns[0]
                combined = df[df[first_col].astype(str).str.upper() == "COMBINED"]
                if not combined.empty:
                    value = float(combined["HOTA"].iloc[0])
                else:
                    value = float(df["HOTA"].iloc[-1])
            else:
                    value = float(df["HOTA"].iloc[-1])
            if value > 1.0:
                value /= 100.0
            return max(0.0, min(1.0, value))
        except Exception:
            continue

    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\bHOTA\b[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise ValueError(f"Failed to parse HOTA from summary: {summary_path}")
    value = float(match.group(1))
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def validate_trackeval_frame_range(df: pd.DataFrame, seq_length: int, label: str) -> None:
    if df.empty:
        return
    frame_series = df["frame"].astype(int)
    invalid = df[(frame_series < 1) | (frame_series > seq_length)]
    if not invalid.empty:
        sample = invalid["frame"].astype(int).head(5).tolist()
        raise ValueError(
            f"{label} has frame(s) out of valid range [1, {seq_length}] after normalization. "
            f"Example: {sample}"
        )


def run_trackeval_hota(
    pred_df: pd.DataFrame,
    gt_clip_df: pd.DataFrame,
    eval_clip_id: str,
    start_frame: int,
    end_frame: int,
    seq_info: dict,
    trackeval_dir: Path,
    allow_hota_stdout_fallback: bool,
) -> float:
    if gt_clip_df.empty:
        raise ValueError(EMPTY_GT_SENTINEL)
    if end_frame < start_frame:
        raise ValueError(f"Invalid clip frame range: start={start_frame}, end={end_frame}")

    with tempfile.TemporaryDirectory(prefix="trackeval_clip_") as tmpdir:
        tmp_root = Path(tmpdir)
        benchmark = "MOT17"
        split = "train"
        benchmark_split = f"{benchmark}-{split}"
        tracker_name = "MyTracker"

        gt_root = tmp_root / "gt" / "mot_challenge"
        trk_root = tmp_root / "trackers" / "mot_challenge"
        seqmaps_dir = gt_root / "seqmaps"
        seqmaps_dir.mkdir(parents=True, exist_ok=True)

        seq_name = sanitize_seq_name(eval_clip_id)
        seq_gt_dir = gt_root / benchmark_split / seq_name
        seq_gt_txt_dir = seq_gt_dir / "gt"
        seq_gt_txt_dir.mkdir(parents=True, exist_ok=True)
        seq_trk_dir = trk_root / benchmark_split / tracker_name / "data"
        seq_trk_dir.mkdir(parents=True, exist_ok=True)

        seq_length = int(end_frame) - int(start_frame) + 1

        gt_eval_df = gt_clip_df.copy()
        gt_eval_df["frame"] = gt_eval_df["frame"].astype(int) - int(start_frame) + 1
        gt_eval_df = gt_eval_df[GT_COLUMNS]
        validate_trackeval_frame_range(gt_eval_df, seq_length, "GT")

        pred_eval_df = pred_df.copy()
        if pred_eval_df.empty:
            pred_eval_df = pd.DataFrame(
                columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
            )
        else:
            pred_eval_df["frame"] = pred_eval_df["frame"].astype(int) - int(start_frame) + 1
            pred_eval_df = pred_eval_df[
                ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
            ]
            validate_trackeval_frame_range(pred_eval_df, seq_length, "Tracker prediction")

        gt_path = seq_gt_txt_dir / "gt.txt"
        trk_path = seq_trk_dir / f"{seq_name}.txt"
        gt_eval_df.to_csv(gt_path, index=False, header=False)
        pred_eval_df.to_csv(trk_path, index=False, header=False)

        seqinfo_text = (
            "[Sequence]\n"
            f"name={seq_name}\n"
            "imDir=img1\n"
            f"frameRate={seq_info['frame_rate']}\n"
            f"seqLength={seq_length}\n"
            f"imWidth={seq_info['im_width']}\n"
            f"imHeight={seq_info['im_height']}\n"
            f"imExt={seq_info['im_ext']}\n"
        )
        (seq_gt_dir / "seqinfo.ini").write_text(seqinfo_text, encoding="utf-8")

        seqmap_path = seqmaps_dir / f"{benchmark_split}.txt"
        seqmap_path.write_text(f"name\n{seq_name}\n", encoding="utf-8")

        run_script = trackeval_dir / "scripts" / "run_mot_challenge.py"
        cmd = [
            sys.executable,
            str(run_script),
            "--GT_FOLDER",
            str(gt_root),
            "--TRACKERS_FOLDER",
            str(trk_root),
            "--BENCHMARK",
            benchmark,
            "--SPLIT_TO_EVAL",
            split,
            "--TRACKERS_TO_EVAL",
            tracker_name,
            "--METRICS",
            "HOTA",
            "--USE_PARALLEL",
            "False",
            "--NUM_PARALLEL_CORES",
            "1",
        ]
        # NOTE: Do not pass --SEQMAP_FILE here.
        # TrackEval's CLI parses options with default None using nargs='+', which turns
        # SEQMAP_FILE into a list and can break os.path.isfile checks. We place seqmap
        # at the default location (GT_FOLDER/seqmaps/<BENCHMARK-SPLIT>.txt) instead.

        result = subprocess.run(
            cmd,
            cwd=str(trackeval_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "TrackEval failed with non-zero exit code: "
                f"{result.returncode}, stderr={safe_error_message(RuntimeError(result.stderr))}"
            )

        summary_path = trk_root / benchmark_split / tracker_name / "pedestrian_summary.txt"
        if summary_path.exists():
            return parse_hota_summary(summary_path)

        matches = list((trk_root / benchmark_split / tracker_name).rglob("pedestrian_summary.txt"))
        if matches:
            return parse_hota_summary(matches[0])

        if allow_hota_stdout_fallback:
            for line in (result.stdout or "").splitlines():
                line_u = line.upper()
                if "COMBINED" not in line_u or "HOTA" not in line_u:
                    continue
                nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)
                if not nums:
                    continue
                value = float(nums[0])
                if value > 1.0:
                    value /= 100.0
                return max(0.0, min(1.0, value))
        raise ValueError(f"HOTA summary not found and no parsable HOTA in TrackEval stdout for clip: {eval_clip_id}")


def evaluate_clip(
    clip_dir: Path,
    start_frame: int,
    end_frame: int,
    im_ext: str,
    model: YOLO,
    gt_map_clip_df: pd.DataFrame,
    gt_hota_clip_df: pd.DataFrame,
    eval_clip_id: str,
    seq_info: dict,
    trackeval_dir: Path,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_device: str,
    yolo_agnostic_nms: bool,
    deepsort_effective_kwargs: dict,
    allow_hota_stdout_fallback: bool,
) -> Tuple[float, float]:
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")

    if gt_map_clip_df.empty or gt_hota_clip_df.empty:
        raise ValueError(EMPTY_GT_SENTINEL)

    # Critical rule: new tracker instance per clip to avoid state leakage.
    tracker = create_deepsort_tracker(deepsort_effective_kwargs)

    pred_for_map: List[dict] = []
    pred_for_track: List[List[float]] = []

    for abs_frame in range(start_frame, end_frame + 1):
        frame_path = clip_dir / f"{abs_frame:06d}{im_ext}"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame in clip: {frame_path}")
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")

        infer_results = model(
            frame,
            classes=[0],
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            device=yolo_device,
            agnostic_nms=yolo_agnostic_nms,
            verbose=False,
        )
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
                pred_for_map.append({"frame": abs_frame, "bbox": [x1, y1, x2, y2], "conf": conf})

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue

            det_conf = getattr(track, "det_conf", None)
            if det_conf is None:
                det_conf = 1.0

            pred_for_track.append(
                [abs_frame, int(track.track_id), float(x1), float(y1), float(w), float(h), float(det_conf), -1, -1, -1]
            )

    gt_for_map = []
    for _, row in gt_map_clip_df.iterrows():
        x1 = float(row["bb_left"])
        y1 = float(row["bb_top"])
        w = float(row["bb_width"])
        h = float(row["bb_height"])
        x2 = x1 + w
        y2 = y1 + h
        if w <= 0.0 or h <= 0.0:
            continue
        gt_for_map.append({"frame": int(row["frame"]), "bbox": [x1, y1, x2, y2]})

    pred_track_df = pd.DataFrame(
        pred_for_track,
        columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
    )

    map50 = calculate_voc2007_ap50(pred_for_map, gt_for_map)
    hota = run_trackeval_hota(
        pred_track_df,
        gt_hota_clip_df,
        eval_clip_id,
        start_frame,
        end_frame,
        seq_info,
        trackeval_dir,
        allow_hota_stdout_fallback,
    )
    return map50, hota


def extract_last_directory_name(file_path_value: str) -> str:
    raw = str(file_path_value).strip()
    if raw == "" or raw.lower() == "nan":
        return ""

    path_obj = Path(raw)
    name = path_obj.name.strip()
    if name == "":
        name = path_obj.parent.name.strip()
    if path_obj.suffix:
        parent_name = path_obj.parent.name.strip()
        if parent_name:
            return parent_name
    return name


def build_target_clip_id(file_path_value: str, fallback_clip_id: str, degradation_type: str, degradation_param) -> str:
    # Policy: clip_id should follow the last directory name of file_path.
    dir_name = extract_last_directory_name(file_path_value)
    if dir_name:
        return dir_name

    clip_id_text = str(fallback_clip_id).strip()
    if clip_id_text and clip_id_text.lower() != "nan":
        return clip_id_text
    param_text = normalize_param_key(degradation_param)
    return f"{fallback_clip_id}_{degradation_type}_{param_text}"


def build_original_triplet_index(clip_df: pd.DataFrame) -> Dict[Tuple[str, int, int], str]:
    index: Dict[Tuple[str, int, int], str] = {}
    original_df = clip_df[clip_df["degradation_type"] == "original"]
    for row in original_df.itertuples(index=False):
        clip_id = str(getattr(row, "clip_id", "")).strip()
        if clip_id == "" or clip_id.lower() == "nan":
            continue
        key = (str(row.sequence_name), int(row.start_frame), int(row.end_frame))
        if key not in index:
            index[key] = clip_id
    return index


def resolve_original_clip_id(
    row,
    clip_id_format: str,
    original_triplet_index: Dict[Tuple[str, int, int], str],
    baseline_cache: Dict[str, Tuple[float, float]],
    original_state_map: Dict[str, str],
) -> Tuple[str, str]:
    start_frame = int(row.start_frame)
    end_frame = int(row.end_frame)
    triplet_key = (str(row.sequence_name), start_frame, end_frame)

    candidates: List[str] = []
    row_clip_id = str(getattr(row, "clip_id", "")).strip()
    if row_clip_id and row_clip_id.lower() != "nan":
        candidates.append(row_clip_id)

    indexed_clip_id = original_triplet_index.get(triplet_key)
    if indexed_clip_id:
        candidates.append(indexed_clip_id)

    formatted_clip_id = clip_id_format.format(
        sequence=row.sequence_name,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    candidates.append(formatted_clip_id)

    ordered_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        c = str(candidate).strip()
        if c == "" or c.lower() == "nan":
            continue
        if c in seen:
            continue
        ordered_candidates.append(c)
        seen.add(c)

    for candidate in ordered_candidates:
        if candidate in baseline_cache:
            return candidate, "baseline_cache"
    for candidate in ordered_candidates:
        if candidate in original_state_map:
            return candidate, "original_state_map"
    if indexed_clip_id:
        return indexed_clip_id, "triplet_index"
    if ordered_candidates:
        return ordered_candidates[0], "candidate"
    return formatted_clip_id, "generated"


def make_target_key(clip_id: str, degradation_type: str, degradation_param) -> Tuple[str, str, str]:
    return (str(clip_id), str(degradation_type), normalize_param_key(degradation_param))


def save_state(
    sync_manager: SyncManager,
    target_records: List[dict],
    cache_records: List[dict],
    failure_records: List[dict],
) -> None:
    target_df = pd.DataFrame(target_records, columns=TARGET_COLUMNS)
    cache_df = pd.DataFrame(cache_records, columns=CACHE_COLUMNS)
    failure_df = pd.DataFrame(failure_records, columns=FAILURE_COLUMNS)
    sync_manager.save_and_sync(target_df, cache_df, failure_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target_manifest.csv from clip_manifest.csv")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model weights path.")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo_iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo_device", type=str, default="0", help="YOLO device string, e.g. '0', 'cpu'.")
    parser.add_argument("--yolo_agnostic_nms", type=str2bool, default=False, help="YOLO agnostic NMS flag.")
    parser.add_argument(
        "--allow_hota_stdout_fallback",
        type=str2bool,
        default=False,
        help="Allow parsing HOTA from TrackEval stdout when summary file is missing.",
    )
    parser.add_argument(
        "--allow_semantic_mismatch",
        type=str2bool,
        default=False,
        help="Allow continuing when run semantics differ from existing snapshot.",
    )
    parser.add_argument(
        "--semantic_run_id",
        type=str,
        default=None,
        help="Optional run-id suffix for output files (e.g., 'smoke').",
    )
    parser.add_argument("--save_every", type=int, default=20, help="Save/sync after this many updates.")
    parser.add_argument("--drive_sync_dir", type=str, default=None, help="Drive manifests dir for sync.")
    parser.add_argument(
        "--strict_drive_mount",
        type=str2bool,
        default=True,
        help="Require drive_sync_dir to be under mounted '/content/drive/MyDrive'.",
    )
    parser.add_argument("--min_orig_threshold", type=float, default=0.1, help="Filter threshold for low P_orig.")
    parser.add_argument(
        "--min_active_trajectories",
        type=int,
        default=3,
        help="Exclude clips with active_trajectories lower than this threshold.",
    )
    parser.add_argument(
        "--expected_clip_length",
        type=int,
        default=30,
        help="Expected clip frame length contract (end_frame - start_frame + 1).",
    )
    parser.add_argument(
        "--trackeval_ref",
        type=str,
        default=None,
        help="Optional TrackEval git ref (commit/tag/branch) to checkout for reproducibility.",
    )
    parser.add_argument("--ds_max_age", type=int, default=30, help="DeepSort max_age.")
    parser.add_argument("--ds_n_init", type=int, default=3, help="DeepSort n_init.")
    parser.add_argument("--ds_nms_max_overlap", type=float, default=1.0, help="DeepSort nms_max_overlap.")
    parser.add_argument("--ds_max_iou_distance", type=float, default=0.7, help="DeepSort max_iou_distance.")
    parser.add_argument("--ds_max_cosine_distance", type=float, default=0.2, help="DeepSort max_cosine_distance.")
    parser.add_argument("--ds_nn_budget", type=int, default=None, help="DeepSort nn_budget.")
    parser.add_argument("--ds_embedder", type=str, default="mobilenet", help="DeepSort embedder backend.")
    parser.add_argument("--ds_embedder_gpu", type=str2bool, default=True, help="DeepSort embedder_gpu.")
    parser.add_argument("--ds_half", type=str2bool, default=True, help="DeepSort embedder half-precision.")
    parser.add_argument("--ds_bgr", type=str2bool, default=True, help="DeepSort BGR input flag.")
    parser.add_argument("--ds_embedder_model_name", type=str, default=None, help="DeepSort embedder model name.")
    parser.add_argument("--ds_embedder_wts", type=str, default=None, help="DeepSort embedder weights path.")
    parser.add_argument("--max_original_clips", type=int, default=None, help="Optional cap for original clips.")
    parser.add_argument("--max_obf_clips", type=int, default=None, help="Optional cap for obfuscated clips.")
    args = parser.parse_args()

    is_partial_run = args.max_original_clips is not None or args.max_obf_clips is not None
    execution_scope = "run_scoped" if args.semantic_run_id else "full_run"
    if args.allow_semantic_mismatch and not args.semantic_run_id:
        raise RuntimeError(
            "--allow_semantic_mismatch true requires --semantic_run_id to prevent accidental canonical overwrite."
        )
    if not args.semantic_run_id and is_partial_run:
        raise RuntimeError(
            "Canonical outputs are strict full-run artifacts. Use --semantic_run_id for partial/smoke runs."
        )

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"[INFO] Project root: {project_root}")

    config = load_config(resolve_path(project_root, args.config))
    raw_dir = resolve_path(project_root, config["paths"]["raw_dir"])
    manifest_dir = resolve_path(project_root, config["paths"]["manifest_dir"])
    logs_dir = resolve_path(project_root, "data/interim/logs")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    clip_manifest_path = manifest_dir / "clip_manifest.csv"
    target_manifest_path = with_semantic_run_id(manifest_dir / "target_manifest.csv", args.semantic_run_id)
    cache_path = with_semantic_run_id(manifest_dir / "original_performance_cache.csv", args.semantic_run_id)
    stats_path = with_semantic_run_id(manifest_dir / "target_stats.csv", args.semantic_run_id)
    stats_partial_path = with_semantic_run_id(manifest_dir / "target_stats.partial.csv", args.semantic_run_id)
    snapshot_path = with_semantic_run_id(manifest_dir / "run_config_snapshot.json", args.semantic_run_id)
    snapshot_partial_path = with_semantic_run_id(manifest_dir / "run_config_snapshot.partial.json", args.semantic_run_id)
    failure_path = with_semantic_run_id(logs_dir / "target_failures.csv", args.semantic_run_id)

    if not clip_manifest_path.exists():
        raise FileNotFoundError(f"Missing clip manifest: {clip_manifest_path}")
    clip_df = pd.read_csv(clip_manifest_path)
    manifest_identity = build_clip_manifest_identity(clip_manifest_path, clip_df)

    drive_manifest_dir = resolve_drive_manifest_dir(args.drive_sync_dir, args.strict_drive_mount)
    if drive_manifest_dir is None:
        print("[WARNING] Drive sync dir unresolved. Using local-only persistence.")
    else:
        print(f"[INFO] Drive sync dir: {drive_manifest_dir}")

    sync_manager = SyncManager(
        local_target_path=target_manifest_path,
        local_cache_path=cache_path,
        local_failure_path=failure_path,
        local_stats_path=stats_path,
        local_snapshot_path=snapshot_path,
        local_snapshot_partial_path=snapshot_partial_path,
        drive_manifest_dir=drive_manifest_dir,
    )
    sync_manager.recover_snapshot_to_local()

    trackeval_dir = project_root / "TrackEval"
    ensure_trackeval(trackeval_dir, args.trackeval_ref)
    trackeval_commit = get_trackeval_commit_hash(trackeval_dir)
    deep_sort_config = build_deepsort_config(args)
    deep_sort_effective_kwargs = resolve_deepsort_effective_kwargs(deep_sort_config, strict=True)
    deep_sort_version = safe_package_version("deep-sort-realtime")
    map_definition = "PASCAL VOC2007 11-point AP@0.5 (single class: person)"

    print(f"[INFO] Loading detector: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    yolo_weights_path = resolve_yolo_weights_path(args.yolo_model, model)
    provenance_identity = build_provenance_identity(
        script_path=Path(__file__).resolve(),
        yolo_weights_path=yolo_weights_path,
        deep_sort_version=deep_sort_version,
    )
    if execution_scope == "full_run":
        unresolved = []
        if provenance_identity.get("script_sha256", "") == "":
            unresolved.append("script_sha256")
        if provenance_identity.get("yolo_weights_resolved_path", "") == "":
            unresolved.append("yolo_weights_resolved_path")
        if provenance_identity.get("yolo_weights_sha256", "") == "":
            unresolved.append("yolo_weights_sha256")
        ds_ver = str(provenance_identity.get("deep_sort_realtime_version", "")).strip().lower()
        if ds_ver in {"", "unknown"}:
            unresolved.append("deep_sort_realtime_version")
        if unresolved:
            raise RuntimeError(
                "Canonical full-run requires fully resolved provenance fields. Missing: "
                f"{', '.join(unresolved)}"
            )

    semantic_signature = build_semantic_signature(
        args=args,
        map_definition=map_definition,
        trackeval_commit=trackeval_commit,
        manifest_identity=manifest_identity,
        provenance_identity=provenance_identity,
        deep_sort_config=deep_sort_config,
        deep_sort_effective_kwargs=deep_sort_effective_kwargs,
        execution_scope=execution_scope,
        is_partial_run=is_partial_run,
        zero_clipped_policy=ZERO_CLIPPED_POLICY,
        stats_mode=STATS_MODE,
    )

    existing_snapshot, existing_snapshot_path, existing_snapshot_source = select_semantic_snapshot_source(
        snapshot_path, snapshot_partial_path
    )
    if existing_snapshot_source == "partial" and existing_snapshot_path is not None:
        print(f"[INFO] Using partial snapshot as semantic source for resume: {existing_snapshot_path}")
    semantic_mismatch_diffs: List[str] = []
    should_recover_artifacts = True
    if existing_snapshot and "semantic_signature" in existing_snapshot:
        semantic_mismatch_diffs = compare_nested_dict(existing_snapshot["semantic_signature"], semantic_signature)
        if semantic_mismatch_diffs:
            should_recover_artifacts = False
            if not args.allow_semantic_mismatch:
                diff_text = "\n".join(semantic_mismatch_diffs[:20])
                raise RuntimeError(
                    "Semantic consistency check failed. Existing outputs were generated with different semantics.\n"
                    f"{diff_text}"
                )
            print("[WARNING] Semantic mismatch detected. Existing artifacts will NOT be recovered.")
            for diff in semantic_mismatch_diffs[:20]:
                print(f"[WARNING] semantic diff: {diff}")
    elif existing_snapshot and "semantic_signature" not in existing_snapshot and not args.allow_semantic_mismatch:
        source_label = str(existing_snapshot_path) if existing_snapshot_path is not None else "snapshot"
        raise RuntimeError(
            f"Existing snapshot is missing semantic_signature ({source_label}). "
            "Use --allow_semantic_mismatch true to proceed without artifact recovery."
        )
    elif existing_snapshot is None:
        local_artifacts_exist = any(p.exists() for p in [target_manifest_path, cache_path, stats_path, failure_path])
        drive_artifacts_exist = False
        if sync_manager.drive_manifest_dir is not None:
            drive_candidates = [
                sync_manager.drive_target_path,
                sync_manager.drive_cache_path,
                sync_manager.drive_stats_path,
                sync_manager.drive_failure_path,
            ]
            drive_artifacts_exist = any(p is not None and p.exists() for p in drive_candidates)

        if local_artifacts_exist or drive_artifacts_exist:
            should_recover_artifacts = False
            if not args.allow_semantic_mismatch:
                raise RuntimeError(
                    "Existing artifacts found but semantic snapshot is missing "
                    "(run_config_snapshot.json or run_config_snapshot.partial.json). "
                    "Cannot verify semantic consistency."
                )
            print("[WARNING] Existing artifacts found without snapshot. Proceeding without artifact recovery.")

    if should_recover_artifacts:
        sync_manager.recover_latest_to_local()

    if should_recover_artifacts:
        target_df = read_csv_or_empty(target_manifest_path, TARGET_COLUMNS)
        cache_df = read_csv_or_empty(cache_path, CACHE_COLUMNS)
        failure_df = read_csv_or_empty(failure_path, FAILURE_COLUMNS)
    else:
        target_df = pd.DataFrame(columns=TARGET_COLUMNS)
        cache_df = pd.DataFrame(columns=CACHE_COLUMNS)
        failure_df = pd.DataFrame(columns=FAILURE_COLUMNS)

    target_df["degradation_param_norm"] = target_df["degradation_param"].apply(normalize_param_key)
    target_df = target_df.drop_duplicates(
        subset=["clip_id", "degradation_type", "degradation_param_norm"],
        keep="last",
    ).drop(columns=["degradation_param_norm"])
    cache_df = cache_df.drop_duplicates(subset=["original_clip_id"], keep="last")

    target_records = target_df.to_dict("records")
    cache_records = cache_df.to_dict("records")
    failure_records = failure_df.to_dict("records")

    completed_target_keys = {
        make_target_key(r["clip_id"], r["degradation_type"], r["degradation_param"]) for r in target_records
    }
    baseline_cache: Dict[str, Tuple[float, float]] = {}
    for _, row in cache_df.iterrows():
        original_id = str(row["original_clip_id"])
        baseline_cache[original_id] = (float(row["p_orig_map"]), float(row["p_orig_hota"]))
    original_state_map: Dict[str, str] = {original_id: ORIG_STATE_READY for original_id in baseline_cache.keys()}

    write_run_config_snapshot(
        snapshot_partial_path,
        args,
        trackeval_dir,
        map_definition=map_definition,
        manifest_identity=manifest_identity,
        provenance_identity=provenance_identity,
        semantic_signature=semantic_signature,
        execution_scope=execution_scope,
        is_partial_run=is_partial_run,
        zero_clipped_policy=ZERO_CLIPPED_POLICY,
        stats_mode=STATS_MODE,
        deep_sort_effective_kwargs=deep_sort_effective_kwargs,
    )
    sync_manager.sync_snapshot_partial_to_drive()

    seq_info_cache: Dict[str, dict] = {}
    seq_gt_raw_cache: Dict[str, pd.DataFrame] = {}
    seq_gt_map_cache: Dict[str, pd.DataFrame] = {}

    def get_seq_info(sequence_name: str) -> dict:
        if sequence_name not in seq_info_cache:
            seq_info_cache[sequence_name] = parse_sequence_info(raw_dir / sequence_name)
        return seq_info_cache[sequence_name]

    def get_gt_df_raw(sequence_name: str) -> pd.DataFrame:
        if sequence_name not in seq_gt_raw_cache:
            gt_path = raw_dir / sequence_name / "gt" / "gt.txt"
            gt_df_seq = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)
            seq_gt_raw_cache[sequence_name] = gt_df_seq
        return seq_gt_raw_cache[sequence_name]

    def get_gt_df_map(sequence_name: str) -> pd.DataFrame:
        if sequence_name not in seq_gt_map_cache:
            gt_df_raw = get_gt_df_raw(sequence_name)
            gt_df_map = gt_df_raw[(gt_df_raw["class"] == 1) & (gt_df_raw["conf"] == 1)].copy()
            seq_gt_map_cache[sequence_name] = gt_df_map
        return seq_gt_map_cache[sequence_name]

    dirty_updates = 0
    skip_stats = {
        "original_low_active_trajectories": 0,
        "obfuscated_low_active_trajectories": 0,
        "obfuscated_existing_target": 0,
        "obfuscated_missing_baseline": 0,
        "obfuscated_low_baseline_filtered": 0,
        "contract_violation": 0,
    }
    split_stats: Dict[str, dict] = {}

    target_record_map = {
        make_target_key(r["clip_id"], r["degradation_type"], r["degradation_param"]): r for r in target_records
    }
    original_triplet_index = build_original_triplet_index(clip_df)
    run_completed_successfully = False

    def maybe_flush(force: bool = False) -> None:
        nonlocal dirty_updates
        if force or dirty_updates >= max(1, args.save_every):
            stats_df = build_target_stats_df(split_stats)
            save_state(sync_manager, target_records, cache_records, failure_records)
            atomic_write_csv(stats_df, stats_partial_path, STATS_COLUMNS)
            dirty_updates = 0

    try:
        print("[INFO] Step A: Computing/caching P_orig for original clips.")
        original_df = clip_df[clip_df["degradation_type"] == "original"].copy()
        if args.max_original_clips is not None:
            original_df = original_df.head(args.max_original_clips)

        for row in tqdm(original_df.itertuples(index=False), total=len(original_df), desc="Original"):
            original_clip_id = str(row.clip_id)
            if to_int(getattr(row, "active_trajectories", 0), default=0) < args.min_active_trajectories:
                skip_stats["original_low_active_trajectories"] += 1
                original_state_map[original_clip_id] = ORIG_STATE_LOW_ACTIVE
                baseline_cache.pop(original_clip_id, None)
                cache_records = [r for r in cache_records if str(r.get("original_clip_id")) != original_clip_id]
                continue

            start_frame = int(row.start_frame)
            end_frame = int(row.end_frame)
            if (end_frame - start_frame + 1) != args.expected_clip_length:
                original_state_map[original_clip_id] = ORIG_STATE_CONTRACT_VIOLATION
                baseline_cache.pop(original_clip_id, None)
                cache_records = [r for r in cache_records if str(r.get("original_clip_id")) != original_clip_id]
                append_failure(
                    failure_records=failure_records,
                    stage="contract_violation",
                    clip_id=original_clip_id,
                    original_clip_id=original_clip_id,
                    degradation_type="original",
                    degradation_param=None,
                    exc=ValueError(
                        f"Expected clip length {args.expected_clip_length}, got {end_frame - start_frame + 1}"
                    ),
                )
                dirty_updates += 1
                maybe_flush(force=False)
                continue

            if baseline_cache.get(original_clip_id) is not None and original_state_map.get(original_clip_id) == ORIG_STATE_READY:
                continue

            try:
                seq_info = get_seq_info(row.sequence_name)
                gt_seq_raw = get_gt_df_raw(row.sequence_name)
                gt_seq_map = get_gt_df_map(row.sequence_name)
                gt_map_clip = gt_seq_map[(gt_seq_map["frame"] >= start_frame) & (gt_seq_map["frame"] <= end_frame)].copy()
                gt_hota_clip = gt_seq_raw[(gt_seq_raw["frame"] >= start_frame) & (gt_seq_raw["frame"] <= end_frame)].copy()
                clip_dir = resolve_path(project_root, str(row.file_path))
                validate_clip_frames(clip_dir, start_frame, end_frame, seq_info["im_ext"])

                p_orig_map, p_orig_hota = evaluate_clip(
                    clip_dir=clip_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    im_ext=seq_info["im_ext"],
                    model=model,
                    gt_map_clip_df=gt_map_clip,
                    gt_hota_clip_df=gt_hota_clip,
                    eval_clip_id=original_clip_id,
                    seq_info=seq_info,
                    trackeval_dir=trackeval_dir,
                    yolo_conf=args.yolo_conf,
                    yolo_iou=args.yolo_iou,
                    yolo_imgsz=args.yolo_imgsz,
                    yolo_device=args.yolo_device,
                    yolo_agnostic_nms=args.yolo_agnostic_nms,
                    deepsort_effective_kwargs=deep_sort_effective_kwargs,
                    allow_hota_stdout_fallback=args.allow_hota_stdout_fallback,
                )

                baseline_cache[original_clip_id] = (p_orig_map, p_orig_hota)
                original_state_map[original_clip_id] = ORIG_STATE_READY
                cache_records.append(
                    {
                        "original_clip_id": original_clip_id,
                        "p_orig_map": p_orig_map,
                        "p_orig_hota": p_orig_hota,
                    }
                )
                dirty_updates += 1
                maybe_flush(force=False)
            except Exception as exc:  # noqa: BLE001
                baseline_cache.pop(original_clip_id, None)
                cache_records = [r for r in cache_records if str(r.get("original_clip_id")) != original_clip_id]
                if is_empty_gt_error(exc):
                    original_state_map[original_clip_id] = ORIG_STATE_EMPTY_GT
                    failure_stage = "original_empty_gt"
                elif isinstance(exc, ValueError) and str(exc).startswith("contract_violation"):
                    original_state_map[original_clip_id] = ORIG_STATE_CONTRACT_VIOLATION
                    failure_stage = "contract_violation"
                else:
                    original_state_map[original_clip_id] = ORIG_STATE_EVAL_FAILURE
                    failure_stage = "original_eval"
                append_failure(
                    failure_records=failure_records,
                    stage=failure_stage,
                    clip_id=original_clip_id,
                    original_clip_id=original_clip_id,
                    degradation_type="original",
                    degradation_param=None,
                    exc=exc,
                )
                dirty_updates += 1
                maybe_flush(force=False)

        print("[INFO] Step B: Computing P_anon and deltas for obfuscated clips.")
        obf_df = clip_df[clip_df["degradation_type"] != "original"].copy()
        if args.max_obf_clips is not None:
            obf_df = obf_df.head(args.max_obf_clips)
        for split_name in sorted({normalize_split_name(v) for v in obf_df["split"].tolist()}):
            if split_name not in split_stats:
                split_stats[split_name] = empty_split_stats()

        clip_id_format = config["naming"]["clip_id_format"]

        for row in tqdm(obf_df.itertuples(index=False), total=len(obf_df), desc="Obfuscated"):
            split_name = normalize_split_name(getattr(row, "split", "unknown"))
            increment_split_stat(split_stats, split_name, "total_candidates")

            target_clip_id = build_target_clip_id(
                file_path_value=str(row.file_path),
                fallback_clip_id=str(row.clip_id),
                degradation_type=str(row.degradation_type),
                degradation_param=row.degradation_param,
            )
            target_key = make_target_key(target_clip_id, row.degradation_type, row.degradation_param)
            if target_key in completed_target_keys:
                skip_stats["obfuscated_existing_target"] += 1
                increment_split_stat(split_stats, split_name, "reused_completed_count")
                existing = target_record_map.get(target_key)
                existing_delta_map = to_float(existing.get("delta_map")) if existing else None
                existing_delta_hota = to_float(existing.get("delta_hota")) if existing else None
                if existing_delta_map == 0.0 and existing_delta_hota == 0.0:
                    increment_split_stat(split_stats, split_name, "zero_clipped_count")
                continue

            if to_int(getattr(row, "active_trajectories", 0), default=0) < args.min_active_trajectories:
                skip_stats["obfuscated_low_active_trajectories"] += 1
                increment_split_stat(split_stats, split_name, "excluded_low_active_trajectories")
                continue

            start_frame = int(row.start_frame)
            end_frame = int(row.end_frame)
            actual_len = end_frame - start_frame + 1
            if actual_len != args.expected_clip_length:
                skip_stats["contract_violation"] += 1
                increment_split_stat(split_stats, split_name, "excluded_eval_failure")
                append_failure(
                    failure_records=failure_records,
                    stage="contract_violation",
                    clip_id=target_clip_id,
                    original_clip_id=str(row.clip_id),
                    degradation_type=str(row.degradation_type),
                    degradation_param=row.degradation_param,
                    exc=ValueError(f"Expected clip length {args.expected_clip_length}, got {actual_len}"),
                )
                dirty_updates += 1
                maybe_flush(force=False)
                continue

            try:
                seq_info = get_seq_info(row.sequence_name)
                clip_dir = resolve_path(project_root, str(row.file_path))
                validate_clip_frames(clip_dir, start_frame, end_frame, seq_info["im_ext"])
            except Exception as clip_contract_exc:  # noqa: BLE001
                skip_stats["contract_violation"] += 1
                increment_split_stat(split_stats, split_name, "excluded_eval_failure")
                append_failure(
                    failure_records=failure_records,
                    stage="contract_violation",
                    clip_id=target_clip_id,
                    original_clip_id=str(row.clip_id),
                    degradation_type=str(row.degradation_type),
                    degradation_param=row.degradation_param,
                    exc=clip_contract_exc,
                )
                dirty_updates += 1
                maybe_flush(force=False)
                continue

            original_clip_id, original_resolution_source = resolve_original_clip_id(
                row=row,
                clip_id_format=clip_id_format,
                original_triplet_index=original_triplet_index,
                baseline_cache=baseline_cache,
                original_state_map=original_state_map,
            )
            original_reason = original_state_map.get(original_clip_id)
            if original_clip_id not in baseline_cache:
                skip_stats["obfuscated_missing_baseline"] += 1
                if original_reason == ORIG_STATE_LOW_ACTIVE:
                    increment_split_stat(split_stats, split_name, "excluded_low_active_trajectories")
                    failure_stage = "obfuscated_missing_baseline_from_orig_low_active"
                elif original_reason == ORIG_STATE_EMPTY_GT:
                    increment_split_stat(split_stats, split_name, "excluded_empty_gt")
                    failure_stage = "obfuscated_missing_baseline_from_orig_empty_gt"
                elif original_reason == ORIG_STATE_CONTRACT_VIOLATION:
                    increment_split_stat(split_stats, split_name, "excluded_eval_failure")
                    failure_stage = "obfuscated_missing_baseline_from_orig_contract_violation"
                elif original_reason == ORIG_STATE_EVAL_FAILURE:
                    increment_split_stat(split_stats, split_name, "excluded_eval_failure")
                    failure_stage = "obfuscated_missing_baseline_from_orig_eval_failure"
                else:
                    increment_split_stat(split_stats, split_name, "excluded_missing_baseline")
                    failure_stage = "schema_mismatch_candidate"
                append_failure(
                    failure_records=failure_records,
                    stage=failure_stage,
                    clip_id=target_clip_id,
                    original_clip_id=original_clip_id,
                    degradation_type=str(row.degradation_type),
                    degradation_param=row.degradation_param,
                    exc=RuntimeError(
                        "Missing baseline performance for original clip. "
                        f"resolution_source={original_resolution_source}, original_reason={original_reason}"
                    ),
                )
                dirty_updates += 1
                maybe_flush(force=False)
                continue

            p_orig_map, p_orig_hota = baseline_cache[original_clip_id]
            if p_orig_map < args.min_orig_threshold or p_orig_hota < args.min_orig_threshold:
                skip_stats["obfuscated_low_baseline_filtered"] += 1
                increment_split_stat(split_stats, split_name, "excluded_low_baseline")
                continue

            try:
                gt_seq_raw = get_gt_df_raw(row.sequence_name)
                gt_seq_map = get_gt_df_map(row.sequence_name)
                gt_map_clip = gt_seq_map[(gt_seq_map["frame"] >= start_frame) & (gt_seq_map["frame"] <= end_frame)].copy()
                gt_hota_clip = gt_seq_raw[(gt_seq_raw["frame"] >= start_frame) & (gt_seq_raw["frame"] <= end_frame)].copy()

                p_anon_map, p_anon_hota = evaluate_clip(
                    clip_dir=clip_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    im_ext=seq_info["im_ext"],
                    model=model,
                    gt_map_clip_df=gt_map_clip,
                    gt_hota_clip_df=gt_hota_clip,
                    eval_clip_id=target_clip_id,
                    seq_info=seq_info,
                    trackeval_dir=trackeval_dir,
                    yolo_conf=args.yolo_conf,
                    yolo_iou=args.yolo_iou,
                    yolo_imgsz=args.yolo_imgsz,
                    yolo_device=args.yolo_device,
                    yolo_agnostic_nms=args.yolo_agnostic_nms,
                    deepsort_effective_kwargs=deep_sort_effective_kwargs,
                    allow_hota_stdout_fallback=args.allow_hota_stdout_fallback,
                )

                record = {
                    "clip_id": target_clip_id,
                    "original_clip_id": original_clip_id,
                    "degradation_type": str(row.degradation_type),
                    "degradation_param": to_float(row.degradation_param),
                    "p_orig_map": float(p_orig_map),
                    "p_orig_hota": float(p_orig_hota),
                    "p_anon_map": float(p_anon_map),
                    "p_anon_hota": float(p_anon_hota),
                    "delta_map": max(0.0, float(p_orig_map) - float(p_anon_map)),
                    "delta_hota": max(0.0, float(p_orig_hota) - float(p_anon_hota)),
                }
                target_records.append(record)
                completed_target_keys.add(target_key)
                target_record_map[target_key] = record
                increment_split_stat(split_stats, split_name, "included_targets")
                if record["delta_map"] == 0.0 and record["delta_hota"] == 0.0:
                    increment_split_stat(split_stats, split_name, "zero_clipped_count")
                dirty_updates += 1
                maybe_flush(force=False)

            except Exception as exc:  # noqa: BLE001
                if is_empty_gt_error(exc):
                    failure_stage = "obfuscated_empty_gt"
                    increment_split_stat(split_stats, split_name, "excluded_empty_gt")
                else:
                    failure_stage = "obfuscated_eval"
                    increment_split_stat(split_stats, split_name, "excluded_eval_failure")
                append_failure(
                    failure_records=failure_records,
                    stage=failure_stage,
                    clip_id=target_clip_id,
                    original_clip_id=original_clip_id,
                    degradation_type=str(row.degradation_type),
                    degradation_param=row.degradation_param,
                    exc=exc,
                )
                dirty_updates += 1
                maybe_flush(force=False)
        run_completed_successfully = True
    finally:
        print("[INFO] Final save/sync.")
        maybe_flush(force=True)
        final_stats_df = build_target_stats_df(split_stats)
        if run_completed_successfully:
            if not snapshot_partial_path.exists():
                raise RuntimeError(
                    "Snapshot partial file is missing at completion; cannot promote canonical snapshot: "
                    f"{snapshot_partial_path}"
                )
            atomic_copy_file(snapshot_partial_path, snapshot_path)
            sync_manager.sync_snapshot_to_drive()
            print("[INFO] Canonical run_config_snapshot.json promoted after successful completion.")
            sync_manager.cleanup_partial_snapshot_best_effort()
            sync_manager.save_stats_and_sync(final_stats_df)
            print("[INFO] Canonical target_stats.csv promoted after successful completion.")
        else:
            print(
                "[WARNING] Run did not complete successfully. "
                "Canonical target_stats.csv was not updated; partial stats remain in:"
                f" {stats_partial_path}"
            )
            print(
                "[WARNING] Canonical run_config_snapshot.json was not updated; partial snapshot remains in:"
                f" {snapshot_partial_path}"
            )
        failure_stage_counts: Dict[str, int] = {}
        for record in failure_records:
            stage = str(record.get("stage", "unknown"))
            failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

        print(f"[SUCCESS] Target rows: {len(target_records)}")
        print(f"[SUCCESS] Baseline cache rows: {len(cache_records)}")
        print(f"[INFO] Failure rows: {len(failure_records)}")
        print(f"[INFO] Skip stats: {skip_stats}")
        print(f"[INFO] Failure stage stats: {failure_stage_counts}")
        print(f"[INFO] Target stats rows: {len(final_stats_df)}")
        print(f"[INFO] Target manifest: {target_manifest_path}")
        print(f"[INFO] Target stats: {stats_path}")
        print(f"[INFO] Run snapshot: {snapshot_path}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user.")
        raise
    except Exception as fatal_exc:  # noqa: BLE001
        print(f"[FATAL] {type(fatal_exc).__name__}: {safe_error_message(fatal_exc)}")
        raise
    finally:
        elapsed = time.time() - start_time
        print(f"[INFO] Elapsed: {elapsed:.2f}s")
